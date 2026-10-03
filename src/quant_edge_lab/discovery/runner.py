"""Shared-scan staged discovery. One parquet read per day for all active families."""

from __future__ import annotations

import json
import traceback
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any, Callable

import polars as pl
from rich.console import Console
from rich.table import Table

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
from quant_edge_lab.discovery.catalog import family_by_id
from quant_edge_lab.discovery.eval_batch10 import (
    BATCH10_PRIMARY,
    count_n_gap10,
    eval_family_day,
    expand_batch10_jobs,
    jobs_from_keys,
    med_from_hist,
    typ_from_hist,
)
from quant_edge_lab.discovery.frozen_direction import expected_direction_for_job
from quant_edge_lab.discovery.event_policy import apply_event_policy
from quant_edge_lab.discovery.funnel import decide_stage
from quant_edge_lab.discovery.knowledge import atomic_write_json, record_run
from quant_edge_lab.discovery.models import Checkpoint, FamilyIdea
from quant_edge_lab.discovery.outcomes_vector import forward_returns_vectorized
from quant_edge_lab.discovery.stages import StageName, select_stage_days
from quant_edge_lab.events.engine import evaluate_events
from quant_edge_lab.hypotheses.loader import load_hypothesis
from quant_edge_lab.pipeline_flat import _featured_trigger_window, _update_hist
from quant_edge_lab.universe.filters import add_session_columns

console = Console()

S1_HORIZONS = ["1m", "5m", "15m", "30m", "60m"]


def _batch_dir(root: Path, batch_id: str) -> Path:
    p = Paths(root).derived / "discovery" / "batches" / batch_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_path(root: Path, batch_id: str) -> Path:
    return _batch_dir(root, batch_id) / "state.json"


def load_state(root: Path, batch_id: str) -> dict[str, Any]:
    p = _state_path(root, batch_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(_state_path(root, state["batch_id"]), state)


def _apply_engine(
    fam: FamilyIdea,
    featured: pl.DataFrame,
    instruments: pl.DataFrame,
    spec,
    exp_id: str,
) -> pl.DataFrame:
    if fam.engine in {"yaml_conditions", "yaml_conditions_monday"}:
        ev = evaluate_events(featured, instruments, spec, experiment_id=exp_id)
        if fam.engine == "yaml_conditions_monday" and ev.height:
            ev = ev.filter(pl.col("session_date").dt.weekday() == 1)
        return ev
    raise RuntimeError(f"{fam.id} engine={fam.engine} is not executable")


def _check_availability(featured: pl.DataFrame) -> None:
    if featured.height == 0 or "available_at" not in featured.columns:
        return
    bad = featured.filter(pl.col("available_at") > pl.col("decision_ts"))
    if bad.height:
        raise AssertionError(f"look-ahead: {bad.height} rows available_at > decision_ts")


def _job_key(fid: str, vid: str) -> str:
    return fid if vid == "default" else vid


def _running_state(stage: StageName, passed: bool | None = None, error: bool = False) -> str:
    if error:
        return "ERROR"
    if passed is True:
        return {"fast": "PASSED_STAGE_1", "broad": "PASSED_STAGE_2", "full": "PASSED_DISCOVERY"}[stage]
    if passed is False:
        return {"fast": "KILLED_STAGE_1", "broad": "KILLED_STAGE_2", "full": "KILLED_DISCOVERY"}[stage]
    return {"fast": "RUNNING_STAGE_1", "broad": "RUNNING_STAGE_2", "full": "RUNNING_FULL_DISCOVERY"}[stage]


def run_batch(
    root: Path,
    family_ids: list[str],
    *,
    batch_id: str,
    stage: StageName = "fast",
    resume: bool = False,
    day_loader: Callable[[str], pl.DataFrame] | None = None,
    all_days: list[str] | None = None,
    instruments: pl.DataFrame | None = None,
    print_every: int = 5,
    expand_variants: bool = False,
    job_keys: list[str] | None = None,
    extra_jobs: list[tuple[str, str, dict]] | None = None,
    compact_returns: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    Paths(root).ensure()
    if extra_jobs:
        job_list = extra_jobs
        family_ids = list(dict.fromkeys(f for f, _, _ in job_list))
    elif job_keys:
        job_list = jobs_from_keys(job_keys)
        family_ids = list(dict.fromkeys(f for f, _, _ in job_list))
    else:
        job_list = expand_batch10_jobs(family_ids, expand_variants=expand_variants)
    fam_map = {i: family_by_id(i) for i in family_ids}
    for fam in fam_map.values():
        if not fam.executable:
            raise RuntimeError(f"{fam.id} is not executable: {fam.idea_only_reason}")

    if all_days is None:
        man = load_manifest(root)
        all_days = sorted(
            d
            for d, rec in man.get("files", {}).items()
            if rec.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
        )
    panel = select_stage_days(all_days, stage)
    inst = instruments
    if inst is None:
        inst_path = Paths(root).normalized / "massive-flat" / "instruments.parquet"
        inst = pl.read_parquet(inst_path)

    state = load_state(root, batch_id) if resume else {}
    if not state:
        state = {
            "batch_id": batch_id,
            "stage": stage,
            "panel": panel,
            "stage_criteria_version": next(iter(fam_map.values())).stage_criteria.version,
            "frozen_spec": "knowledge/families/batch10_frozen_v1.yaml" if expand_variants else None,
            "jobs": {},
            "created_at": datetime.now(UTC).isoformat(),
        }
    for fid, vid, params in job_list:
        key = _job_key(fid, vid)
        if key not in state["jobs"]:
            state["jobs"][key] = Checkpoint(
                family_id=fid,
                variant_id=vid,
                state="RUNNING_STAGE_1" if stage == "fast" else ("RUNNING_STAGE_2" if stage == "broad" else "RUNNING_FULL_DISCOVERY"),
                stage=stage,
                days_total=len(panel),
            ).model_dump()
            state["jobs"][key]["processed_days"] = []
            if not compact_returns:
                state["jobs"][key]["ret_parts"] = []
            state["jobs"][key]["ticker_counts"] = {}
            state["jobs"][key]["params"] = params
            state["jobs"][key]["family_id"] = fid

    specs: dict[str, Any] = {}
    need_h001 = any(fid in {"H001", "H022"} for fid, _, _ in job_list)
    need_h040 = any(fid == "H040" for fid, _, _ in job_list)
    if need_h001:
        specs["H001"] = load_hypothesis(root / "hypotheses/discovery/H001_gap_fade.yaml")
    if need_h040:
        specs["H040"] = load_hypothesis(root / "hypotheses/discovery/H040_monday_gap.yaml")
    for fid in family_ids:
        fam = fam_map[fid]
        if fid not in BATCH10_PRIMARY and fam.hypothesis_yaml:
            specs[fid] = load_hypothesis(root / fam.hypothesis_yaml)

    prev_close: dict[str, float] = {}
    vol_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    range_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    ah_prev: dict[str, float] = {}
    yday_high: dict[str, float] = {}
    yday_low: dict[str, float] = {}
    t0 = datetime.now(UTC)

    def loader(day: str) -> pl.DataFrame:
        if day_loader:
            return day_loader(day)
        return pl.read_parquet(local_parquet_path(root, day))

    for i, day in enumerate(panel):
        bars = loader(day)
        sessioned = add_session_columns(bars)
        featured = _featured_trigger_window(sessioned, prev_close, vol_hist)
        _check_availability(featured)
        typ = typ_from_hist(vol_hist)
        med20 = med_from_hist(range_hist)
        n_gap10 = count_n_gap10(sessioned, inst, prev_close)

        for fid, vid, params in job_list:
            key = _job_key(fid, vid)
            job = state["jobs"][key]
            if day in job.get("processed_days", []):
                continue
            if job.get("state") == "ERROR":
                continue
            fam = fam_map[fid]
            try:
                if fid in BATCH10_PRIMARY:
                    spec = specs.get("H001") if fid in {"H001", "H022"} else specs.get("H040") if fid == "H040" else None
                    ev = eval_family_day(
                        fid,
                        featured=featured,
                        sessioned=sessioned,
                        instruments=inst,
                        spec=spec,
                        exp_id=f"{batch_id}-{key}",
                        prev_close=prev_close,
                        typ=typ,
                        yday_high=yday_high,
                        yday_low=yday_low,
                        ah_ret=ah_prev,
                        med20_range=med20,
                        n_gap10=n_gap10,
                        params=params,
                    )
                else:
                    spec = specs[fid]
                    ev = _apply_engine(fam, featured, inst, spec, f"{batch_id}-{key}")
                    ev = apply_event_policy(ev, fam.event_policy, fam.cooldown_minutes)
                if ev.height:
                    horizons = S1_HORIZONS
                    side_fallback = fam.expected_direction if fam.expected_direction in {"long", "short"} else "long"
                    if params.get("side") in {"long", "short"}:
                        side_fallback = params["side"]
                    fwd = forward_returns_vectorized(ev, sessioned, horizons, side_fallback)
                    use_abs = fam.id == "H004"
                    if fwd.height:
                        keep = [
                            c
                            for c in ["event_id", "ticker", "session_date", "instrument_id"]
                            if c in ev.columns
                        ]
                        meta = ev.select(keep).unique(subset=["event_id"]) if "event_id" in ev.columns else ev.select(keep)
                        fwd = fwd.join(meta, on="event_id", how="left")
                        if "session_date" in fwd.columns:
                            fwd = fwd.with_columns(pl.col("session_date").cast(pl.Utf8).alias("trading_day"))
                        col = pl.col("forward_return").abs() if use_abs else pl.col("forward_return")
                        g15 = fwd.filter((pl.col("horizon") == "15m") & pl.col("forward_return").is_not_null())
                        if g15.height:
                            metric = g15.select(col.alias("m"))["m"]
                            job["sum_15"] = float(job.get("sum_15") or 0) + float(metric.sum())
                            job["n_15"] = int(job.get("n_15") or 0) + metric.len()
                            job["interim_means"] = {"15m": job["sum_15"] / job["n_15"] if job["n_15"] else None}
                        if compact_returns:
                            rdir = _batch_dir(root, batch_id) / "returns" / key
                            rdir.mkdir(parents=True, exist_ok=True)
                            fwd.write_parquet(rdir / f"{day}.parquet")
                        else:
                            part: dict[str, list[float]] = {}
                            g = (
                                fwd.filter(pl.col("forward_return").is_not_null())
                                .with_columns(col.alias("m"))
                                .group_by("horizon")
                                .agg(pl.col("m"))
                            )
                            for row in g.iter_rows(named=True):
                                part[str(row["horizon"])] = [float(x) for x in row["m"]]
                            job.setdefault("ret_parts", []).append(part)
                            job["interim_means"] = _flatten_parts(job["ret_parts"])
                    job["events"] = int(job.get("events") or 0) + ev.height
                    counts = job.setdefault("ticker_counts", {})
                    if "ticker" in ev.columns:
                        for t, n in ev.group_by("ticker").len().iter_rows():
                            counts[str(t)] = int(counts.get(str(t), 0)) + int(n)
                    job.setdefault("dayset", [])
                    job["dayset"] = list(set(job["dayset"]) | {day})
                    job["unsigned_metric"] = use_abs
                job["processed_days"] = list(set(job.get("processed_days") or []) | {day})
                job["days_done"] = len(job["processed_days"])
                job["ticker_days"] = int(job.get("events") or 0)
                job["unique_tickers"] = len(job.get("ticker_counts") or {})
                job["trading_days_with_events"] = len(job.get("dayset") or [])
                job["updated_at"] = datetime.now(UTC).isoformat()
                job["interim"] = True
                job["state"] = _running_state(stage)
            except Exception as exc:
                job["state"] = "ERROR"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["decision_reason"] = traceback.format_exc()[-500:]

        ah = sessioned.filter((pl.col("time_et") >= time(16, 0)) & (pl.col("time_et") < time(20, 0)))
        rth = sessioned.filter(pl.col("is_rth"))
        if rth.height:
            daily = rth.group_by("instrument_id").agg(
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                pl.col("close").last().alias("c"),
            )
            for row in daily.iter_rows(named=True):
                iid = row["instrument_id"]
                pc = prev_close.get(iid)
                if pc and pc > 0:
                    range_hist[iid].append((float(row["h"]) - float(row["l"])) / pc)
                yday_high[iid] = float(row["h"])
                yday_low[iid] = float(row["l"])
            if ah.height:
                rth_c = {r["instrument_id"]: r["rth_close"] for r in daily.rename({"c": "rth_close"}).iter_rows(named=True)}
                ah_c = {
                    r["instrument_id"]: r["ah_close"]
                    for r in ah.group_by("instrument_id").agg(pl.col("close").last().alias("ah_close")).iter_rows(named=True)
                }
                for iid, ac in ah_c.items():
                    rc = rth_c.get(iid)
                    if rc:
                        ah_prev[iid] = ac / rc - 1.0
        _update_hist(sessioned, prev_close, vol_hist)
        if i % print_every == 0:
            _print_status(batch_id, t0, state, panel, i)
            save_state(root, state)

    from quant_edge_lab.discovery.stage_stats import load_job_returns, summarize_forwards
    from quant_edge_lab.statistics.robustness import robustness_tables

    for fid, vid, params in job_list:
        key = _job_key(fid, vid)
        job = state["jobs"][key]
        if job.get("state") == "ERROR":
            continue
        fam = fam_map[fid]
        use_abs = fam.id == "H004"
        if compact_returns:
            fwd = load_job_returns(_batch_dir(root, batch_id) / "returns", key)
            stats = summarize_forwards(fwd, primary=fam.stage_criteria.primary_horizon, abs_primary=use_abs)
            job["stage_stats"] = stats
            means = {h["horizon"]: h["mean"] for h in stats.get("horizons") or []}
            job["interim_means"] = means
            primary = (stats.get("primary") or {}).get("mean")
            n_td = int((stats.get("sample") or {}).get("ticker_days") or job.get("events") or 0)
            n_days = int((stats.get("sample") or {}).get("trading_days") or len(job.get("dayset") or []))
            top_share = (stats.get("sample") or {}).get("top_ticker_share")
            if fwd.height:
                evs = fwd.filter(pl.col("horizon") == fam.stage_criteria.primary_horizon)
                if "event_id" in evs.columns:
                    evs = evs.unique(subset=["event_id"])
                try:
                    job["robustness"] = robustness_tables(evs, fwd)
                except Exception as exc:
                    job["robustness_error"] = f"{type(exc).__name__}: {exc}"
        else:
            means = _flatten_parts(job.get("ret_parts") or [])
            job["interim_means"] = means
            primary = means.get(fam.stage_criteria.primary_horizon)
            n_td = int(job.get("ticker_days") or len(job.get("tdays") or []) or job.get("events") or 0)
            n_days = len(job.get("dayset") or [])
            top_share = None
            counts = job.get("ticker_counts") or {}
            if counts:
                tot = sum(int(v) for v in counts.values())
                top_share = (max(int(v) for v in counts.values()) / tot) if tot else None
            elif job.get("tdays"):
                names = [str(x).split("|")[0] for x in job["tdays"]]
                top_share = max(Counter(names).values()) / len(names)
        if top_share is None:
            counts = job.get("ticker_counts") or {}
            if counts:
                tot = sum(int(v) for v in counts.values())
                top_share = (max(int(v) for v in counts.values()) / tot) if tot else None
        job["concentration_top_ticker_share"] = top_share
        job["ticker_days"] = n_td
        job["trading_days_with_events"] = n_days
        try:
            exp_dir = expected_direction_for_job(fid, vid, root=root)
        except KeyError:
            exp_dir = params.get("expected_direction") or fam.expected_direction
        job["expected_direction"] = exp_dir
        decision, reason = decide_stage(
            expected_direction=exp_dir,
            ticker_days=n_td,
            trading_days=n_days,
            mean_primary=primary,
            top_ticker_share=top_share,
            criteria=fam.stage_criteria,
            stage_complete=True,
        )
        job["decision"] = decision
        job["decision_reason"] = reason
        job["interim"] = False
        job["state"] = _running_state(stage, passed=(decision == "PASS"))
        record_run(
            root,
            {
                "type": "stage_result",
                "batch_id": batch_id,
                "family_id": fid,
                "variant_id": vid,
                "stage": stage,
                "decision": decision,
                "reason": reason,
                "events": job.get("events"),
                "ticker_days": n_td,
                "trading_days": n_days,
                "means": job.get("interim_means"),
                "unsigned_metric": use_abs,
                "expected_direction": exp_dir,
                "concentration_top_ticker_share": top_share,
                "primary_stats": (job.get("stage_stats") or {}).get("primary") if compact_returns else None,
                "note": "SIGNAL_ONLY; not an edge; frozen criteria only",
            },
        )
    state["finished_at"] = datetime.now(UTC).isoformat()
    save_state(root, state)
    _print_status(batch_id, t0, state, panel, len(panel))
    return state


def _flatten_parts(parts: list[dict]) -> dict[str, float | None]:
    acc: dict[str, list[float]] = defaultdict(list)
    for p in parts:
        for h, vals in p.items():
            if isinstance(vals, list):
                acc[h].extend([x for x in vals if x is not None])
            elif vals is not None:
                acc[h].append(float(vals))
    return {h: (sum(v) / len(v) if v else None) for h, v in acc.items()}


def _print_status(batch_id: str, t0: datetime, state: dict, panel: list[str], i: int) -> None:
    elapsed = datetime.now(UTC) - t0
    console.print(f"\nBATCH {batch_id}  Elapsed: {elapsed}  days {i}/{len(panel)}  [INTERIM / NOT FINAL]")
    table = Table()
    for col in ("ID", "STAGE", "PROGRESS", "EVENTS", "TICKER-DAYS", "EFFECT 15m", "STATUS"):
        table.add_column(col)
    n_run = n_kill = n_pass = n_err = 0
    for fid, job in state.get("jobs", {}).items():
        st = job.get("state") or ""
        if st == "ERROR":
            n_err += 1
        elif str(st).startswith("KILLED"):
            n_kill += 1
        elif str(st).startswith("PASSED"):
            n_pass += 1
        else:
            n_run += 1
        means = job.get("interim_means") or {}
        eff = means.get("15m")
        eff_s = f"{eff:.4%}" if isinstance(eff, float) else "—"
        tot = job.get("days_total") or len(panel)
        done = job.get("days_done") or 0
        table.add_row(
            fid,
            str(job.get("stage")),
            f"{done}/{tot}",
            str(job.get("events") or 0),
            str(job.get("ticker_days") or 0),
            eff_s,
            f"{st} {job.get('decision') or ''}",
        )
    console.print(table)
    console.print(f"SUMMARY running={n_run} killed={n_kill} passed={n_pass} errors={n_err}")
