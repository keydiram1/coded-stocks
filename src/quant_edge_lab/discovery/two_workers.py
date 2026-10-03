"""Two logical workers, one shared daily scan. No sealed OOS, no broker."""

from __future__ import annotations

import traceback
from collections import defaultdict, deque
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.table import Table

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
from quant_edge_lab.discovery.dir_gates import (
    BATCH_REL,
    GATES_REL,
    decide_directional,
    frac_days_agree,
    freeze_hashes,
    load_gates,
)
from quant_edge_lab.discovery.eval_batch10 import med_from_hist, typ_from_hist
from quant_edge_lab.discovery.eval_directional import WORKER_A_IDS, WORKER_B_IDS, eval_job, prepare_rth
from quant_edge_lab.discovery.h004_direction import write_h004_signed_baseline
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json, knowledge_dir, record_run
from quant_edge_lab.discovery.outcomes_vector import forward_returns_vectorized
from quant_edge_lab.discovery.runner import _batch_dir, _check_availability, load_state, save_state
from quant_edge_lab.discovery.stage_stats import load_job_returns, summarize_forwards
from quant_edge_lab.discovery.stages import select_falsify_days, select_stage_days
from quant_edge_lab.hashing import git_sha
from quant_edge_lab.pipeline_flat import _featured_trigger_window, _update_hist
from quant_edge_lab.statistics.robustness import robustness_tables
from quant_edge_lab.universe.filters import add_session_columns

console = Console()
HORIZONS = ["1m", "5m", "15m", "30m", "60m"]
BATCH_ID = "dir-two-workers-v1"


def _job_catalog() -> list[dict[str, Any]]:
    jobs = []
    for jid in WORKER_A_IDS:
        jobs.append({"id": jid, "worker": "A", "family_id": "H004", "invert": False})
    for jid in WORKER_B_IDS:
        jobs.append({"id": jid, "worker": "B", "family_id": jid, "invert": False})
    return jobs


def _init_state(root: Path, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "batch_id": BATCH_ID,
        "freeze": hashes,
        "git": git_sha(root),
        "created_at": datetime.now(UTC).isoformat(),
        "jobs": {},
        "derived": [],
        "h004_no_directional_edge": False,
        "note": "SIGNAL_ONLY. Logical workers A/B share one parquet scan.",
    }


def _ensure_jobs(state: dict[str, Any], catalog: list[dict[str, Any]], days_total: int, stage: str) -> None:
    for spec in catalog:
        jid = spec["id"]
        if jid not in state["jobs"]:
            state["jobs"][jid] = {
                "id": jid,
                "worker": spec["worker"],
                "family_id": spec["family_id"],
                "invert": spec.get("invert", False),
                "parent": spec.get("parent"),
                "state": "REGISTERED",
                "processed_days": [],
                "ticker_counts": {},
                "dayset": [],
                "events": 0,
                "params": spec,
            }
        job = state["jobs"][jid]
        if job.get("decision") == "KILL" and not spec.get("invert"):
            continue
        job["stage"] = stage
        job["days_total"] = days_total
        job["interim"] = True
        if job.get("decision") != "KILL":
            job["state"] = "RUNNING_STAGE_1"


def _panel_for(stage: str, all_days: list[str]) -> list[str]:
    if stage == "fast":
        return select_stage_days(all_days, "fast")
    if stage == "falsify":
        return select_falsify_days(all_days, n=100)
    if stage == "broad":
        return select_stage_days(all_days, "broad")
    if stage == "full":
        return select_stage_days(all_days, "full")
    raise ValueError(stage)


def _print_dashboard(state: dict[str, Any], stage: str, i: int, n: int, t0: datetime) -> None:
    elapsed = datetime.now(UTC) - t0
    rate = i / elapsed.total_seconds() if elapsed.total_seconds() > 1 and i else None
    eta = None
    if rate and n > i:
        eta = int((n - i) / rate)
    console.print(
        f"\n=== TWO-WORKER  stage={stage}  days {i}/{n}  elapsed={elapsed}  "
        f"ETA_s={eta}  [INTERIM]  batch={BATCH_ID} ==="
    )
    for worker, title in (("A", "WORKER A — H004 DIRECTION"), ("B", "WORKER B — GENERAL DISCOVERY")):
        table = Table(title=title)
        for col in ("ID", "PROGRESS", "EVENTS", "MEAN15", "MED", "WIN", "STATUS"):
            table.add_column(col)
        n_run = n_kill = n_pass = n_err = 0
        for jid, job in state["jobs"].items():
            if job.get("worker") != worker:
                continue
            st = str(job.get("state") or "")
            if st == "ERROR":
                n_err += 1
            elif st.startswith("KILLED") or job.get("decision") == "KILL":
                n_kill += 1
            elif st.startswith("PASSED") or job.get("decision") == "PASS":
                n_pass += 1
            else:
                n_run += 1
            prim = ((job.get("stage_stats") or {}).get("primary") or {})
            mean = (job.get("interim_means") or {}).get("15m")
            if mean is None:
                mean = prim.get("mean")
            med = prim.get("median")
            win = prim.get("win_rate")
            table.add_row(
                jid,
                f"{job.get('days_done') or 0}/{job.get('days_total') or n}",
                str(job.get("events") or 0),
                f"{mean:.4%}" if isinstance(mean, float) else "—",
                f"{med:.4%}" if isinstance(med, float) else "—",
                f"{win:.3f}" if isinstance(win, float) else "—",
                str(job.get("decision") or st),
            )
        console.print(table)
        console.print(f"{title}: running={n_run} killed={n_kill} passed={n_pass} errors={n_err} [INTERIM]")
        if worker == "A" and state.get("h004_no_directional_edge"):
            console.print("H004 = VALIDATED VOLATILITY PHENOMENON / NO DIRECTIONAL EDGE FOUND")


def _active_ids(state: dict[str, Any], catalog: list[dict[str, Any]]) -> list[str]:
    out = []
    for spec in catalog:
        jid = spec["id"]
        job = state["jobs"].get(jid) or {}
        if job.get("decision") == "KILL":
            continue
        if job.get("state") == "ERROR":
            continue
        out.append(jid)
    return out


def run_shared_stage(
    root: Path,
    state: dict[str, Any],
    *,
    stage: str,
    panel: list[str],
    catalog: list[dict[str, Any]],
    instruments: pl.DataFrame,
    print_every: int = 5,
) -> dict[str, Any]:
    gates = load_gates(root)
    gate = gates.stages[stage if stage in gates.stages else "fast"]
    _ensure_jobs(state, catalog, len(panel), stage)
    active = _active_ids(state, catalog)
    if not active:
        return state
    resuming = any(
        (state["jobs"][jid].get("stage") == stage)
        and (state["jobs"][jid].get("processed_days"))
        and (state["jobs"][jid].get("decision") is None)
        for jid in active
    )
    for jid in active:
        job = state["jobs"][jid]
        job["stage"] = stage
        job["days_total"] = len(panel)
        job["interim"] = True
        rdir = _batch_dir(root, BATCH_ID) / "returns" / f"{stage}_{jid}"
        rdir.mkdir(parents=True, exist_ok=True)
        if not resuming:
            job["processed_days"] = []
            job["events"] = 0
            job["ticker_counts"] = {}
            job["dayset"] = []
            job["sum_15"] = 0.0
            job["n_15"] = 0
            job["decision"] = None
            job["interim_means"] = {}
            job["days_done"] = 0

    prev_close: dict[str, float] = {}
    yday_ret: dict[str, float] = {}
    vol_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    range_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    ah_prev: dict[str, float] = {}
    t0 = datetime.now(UTC)

    for i, day in enumerate(panel):
        bars = pl.read_parquet(local_parquet_path(root, day))
        sessioned = add_session_columns(bars)
        featured = _featured_trigger_window(sessioned, prev_close, vol_hist)
        _check_availability(featured)
        typ = typ_from_hist(vol_hist)
        med20 = med_from_hist(range_hist)
        rth = prepare_rth(sessioned, instruments, prev_close, typ, yday_ret, ah_prev, med20)

        for jid in active:
            job = state["jobs"][jid]
            if day in job.get("processed_days", []):
                continue
            try:
                eval_id = jid[:-4] if jid.endswith(".opp") else jid
                ev = eval_job(
                    eval_id,
                    rth,
                    instruments=instruments,
                    prev_close=prev_close,
                    typ=typ,
                    exp_id=f"{BATCH_ID}-{jid}",
                )
                if job.get("invert") and ev.height and "side" in ev.columns:
                    ev = ev.with_columns(
                        pl.when(pl.col("side") == "long").then(pl.lit("short")).otherwise(pl.lit("long")).alias("side")
                    )
                if ev.height:
                    fwd = forward_returns_vectorized(ev, sessioned, HORIZONS, "long")
                    keep = [c for c in ["event_id", "ticker", "session_date", "instrument_id"] if c in ev.columns]
                    meta = ev.select(keep).unique(subset=["event_id"]) if "event_id" in ev.columns else ev.select(keep)
                    fwd = fwd.join(meta, on="event_id", how="left")
                    if "session_date" in fwd.columns:
                        fwd = fwd.with_columns(pl.col("session_date").cast(pl.Utf8).alias("trading_day"))
                    g15 = fwd.filter((pl.col("horizon") == "15m") & pl.col("forward_return").is_not_null())
                    if g15.height:
                        job["sum_15"] = float(job.get("sum_15") or 0) + float(g15["forward_return"].sum())
                        job["n_15"] = int(job.get("n_15") or 0) + g15.height
                        job["interim_means"] = {"15m": job["sum_15"] / job["n_15"]}
                    rdir = _batch_dir(root, BATCH_ID) / "returns" / f"{stage}_{jid}"
                    fwd.write_parquet(rdir / f"{day}.parquet")
                    job["events"] = int(job.get("events") or 0) + ev.height
                    counts = job.setdefault("ticker_counts", {})
                    if "ticker" in ev.columns:
                        for t, n in ev.group_by("ticker").len().iter_rows():
                            counts[str(t)] = int(counts.get(str(t), 0)) + int(n)
                    job["dayset"] = list(set(job.get("dayset") or []) | {day})
                job["processed_days"] = list(set(job.get("processed_days") or []) | {day})
                job["days_done"] = len(job["processed_days"])
                job["ticker_days"] = int(job.get("events") or 0)
                job["updated_at"] = datetime.now(UTC).isoformat()
            except Exception as exc:
                job["state"] = "ERROR"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["decision_reason"] = traceback.format_exc()[-800:]

        rth_bars = sessioned.filter(pl.col("is_rth"))
        ah = sessioned.filter((pl.col("time_et") >= time(16, 0)) & (pl.col("time_et") < time(20, 0)))
        if rth_bars.height:
            daily = rth_bars.group_by("instrument_id").agg(
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                pl.col("close").last().alias("c"),
            )
            for row in daily.iter_rows(named=True):
                iid = row["instrument_id"]
                pc = prev_close.get(iid)
                if pc and pc > 0:
                    range_hist[iid].append((float(row["h"]) - float(row["l"])) / pc)
                    yday_ret[iid] = float(row["c"]) / pc - 1.0
            if ah.height:
                rth_c = {r["instrument_id"]: r["c"] for r in daily.iter_rows(named=True)}
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
            _print_dashboard(state, stage, i, len(panel), t0)
            save_state(root, state)

    for jid in active:
        job = state["jobs"][jid]
        if job.get("state") == "ERROR":
            continue
        fwd = load_job_returns(_batch_dir(root, BATCH_ID) / "returns", f"{stage}_{jid}")
        stats = summarize_forwards(fwd, primary="15m", abs_primary=False)
        job["stage_stats"] = stats
        job["interim_means"] = {h["horizon"]: h["mean"] for h in stats.get("horizons") or []}
        prim = stats.get("primary") or {}
        sample = stats.get("sample") or {}
        n_td = int(sample.get("ticker_days") or job.get("events") or 0)
        n_days = int(sample.get("trading_days") or len(job.get("dayset") or []))
        top = sample.get("top_ticker_share")
        agree = frac_days_agree(fwd)
        job["frac_days_agree"] = agree
        job["ticker_days"] = n_td
        job["trading_days_with_events"] = n_days
        job["concentration_top_ticker_share"] = top
        if fwd.height:
            evs = fwd.filter(pl.col("horizon") == "15m")
            if "event_id" in evs.columns:
                evs = evs.unique(subset=["event_id"])
            try:
                job["robustness"] = robustness_tables(evs, fwd)
            except Exception as exc:
                job["robustness_error"] = str(exc)
        decision, reason = decide_directional(
            mean_primary=prim.get("mean"),
            median_primary=prim.get("median"),
            win_rate=prim.get("win_rate"),
            ticker_days=n_td,
            trading_days=n_days,
            top_ticker_share=top,
            frac_days_agree=agree,
            ci=prim.get("bootstrap_ci_95"),
            gate=gate,
            stage_complete=True,
        )
        job["decision"] = decision
        job["decision_reason"] = reason
        job["interim"] = False
        job["state"] = (
            "PASSED_STAGE_1" if decision == "PASS" else "KILLED_STAGE_1"
        ) if stage == "fast" else (
            "PASSED_STAGE_2" if decision == "PASS" else "KILLED_STAGE_2"
        ) if stage in {"broad", "falsify"} else (
            "PASSED_DISCOVERY" if decision == "PASS" else "KILLED_DISCOVERY"
        )
        record_run(
            root,
            {
                "type": "directional_stage_result",
                "batch_id": BATCH_ID,
                "job": jid,
                "worker": job.get("worker"),
                "stage": stage,
                "decision": decision,
                "reason": reason,
                "events": job.get("events"),
                "ticker_days": n_td,
                "primary": prim,
                "freeze": state.get("freeze"),
                "note": "SIGNAL_ONLY; directional gates dir_gates_v1",
            },
        )
        mean = prim.get("mean")
        if (
            decision == "KILL"
            and mean is not None
            and mean < 0
            and abs(mean) >= 2 * gate.min_abs_mean_15m
            and not job.get("invert")
            and not jid.endswith(".opp")
        ):
            opp = f"{jid}.opp"
            if opp not in {c["id"] for c in catalog}:
                catalog.append(
                    {
                        "id": opp,
                        "worker": job["worker"],
                        "family_id": job["family_id"] if job["worker"] == "B" else jid,
                        "invert": True,
                        "parent": jid,
                    }
                )
                state["derived"].append(
                    {
                        "id": opp,
                        "parent": jid,
                        "origin": "sign_disagreement_large_stable",
                        "queued_for_next_unseen_panel": True,
                        "parent_mean_15m": mean,
                    }
                )
    save_state(root, state)
    _print_dashboard(state, stage, len(panel), len(panel), t0)
    return state


def write_consolidated_report(root: Path, state: dict[str, Any], baseline: dict[str, Any]) -> Path:
    reports = root / "reports" / "discovery"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "directional_two_workers_v1.md"
    jobs = state.get("jobs") or {}
    a_pass = [k for k, j in jobs.items() if j.get("worker") == "A" and j.get("decision") == "PASS"]
    b_pass = [k for k, j in jobs.items() if j.get("worker") == "B" and j.get("decision") == "PASS"]
    killed = [k for k, j in jobs.items() if j.get("decision") == "KILL"]
    lines = [
        "# Directional two-worker funnel (SIGNAL_ONLY)",
        "",
        "Not sealed OOS. Not trading. Frozen gates: `knowledge/families/directional_gates_v1.yaml`.",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Git: {state.get('git')}",
        f"Freeze hashes: {state.get('freeze')}",
        "",
        "## VALIDATED PHENOMENA",
        "",
        "- H004 unsigned volume-before-price (prior batch10-full) remains a validated **volatility** phenomenon.",
        "",
        "## H004 SIGNED BASELINE (cached full panel)",
        "",
    ]
    for key, rec in (baseline.get("jobs") or {}).items():
        prim = rec.get("signed_primary") or {}
        up = rec.get("unsigned_primary") or {}
        lines.append(
            f"- **{key}** signed 15m mean={prim.get('mean')} median={prim.get('median')} "
            f"win={prim.get('win_rate')} CI={prim.get('bootstrap_ci_95')}; "
            f"|15m| mean={up.get('mean')} sample={rec.get('sample')}"
        )
    lines += [
        "",
        "## DIRECTIONAL CANDIDATES (last completed stage PASS)",
        "",
    ]
    if not a_pass and not b_pass:
        lines.append("None. PASS is not an edge.")
    for k in a_pass + b_pass:
        j = jobs[k]
        prim = (j.get("stage_stats") or {}).get("primary") or {}
        lines += [
            f"### {k} (worker {j.get('worker')})",
            f"- stage={j.get('stage')} events={j.get('events')} ticker_days={j.get('ticker_days')}",
            f"- signed-to-decision 15m mean={prim.get('mean')} median={prim.get('median')} win={prim.get('win_rate')}",
            f"- CI={prim.get('bootstrap_ci_95')} concentration={j.get('concentration_top_ticker_share')}",
            f"- reason: {j.get('decision_reason')}",
            "",
        ]
    if state.get("h004_no_directional_edge"):
        lines += [
            "## H004 DIRECTIONAL CONCLUSION",
            "",
            "H004 = VALIDATED VOLATILITY PHENOMENON / NO DIRECTIONAL EDGE FOUND",
            "",
        ]
    lines += ["## FAILED HYPOTHESES", ""]
    for k in killed:
        j = jobs[k]
        lines.append(f"- {k}: {j.get('decision_reason')}")
    lines += [
        "",
        "## DERIVED HYPOTHESES",
        "",
        str(state.get("derived") or []),
        "",
        "## DATA LIMITATIONS",
        "",
        "- as-printed prices; no NBBO; SIGNAL_ONLY next_bar_open; no sector ETFs in dataset",
        "- cross-section uses same-day RTH names in the converted universe only",
        "",
        "## EXECUTION LIMITATIONS",
        "",
        "- No fills, borrow, or fees. Do not trade from this report.",
        "",
        "## NEXT DATA NEEDED",
        "",
        "- Official sector mapping / liquid ETF bars if relative-sector tests are required",
        "",
        "## NEXT RESEARCH DIRECTIONS",
        "",
        "- Only after this funnel exhausts: new frozen families, not parameter mining of KILLs.",
        "",
        "No sealed OOS. No orders.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(
        knowledge_dir(root) / "experiments" / "directional_two_workers_summary.json",
        {
            "report": str(path),
            "freeze": state.get("freeze"),
            "h004_no_directional_edge": state.get("h004_no_directional_edge"),
            "jobs": {k: {"decision": j.get("decision"), "reason": j.get("decision_reason"), "worker": j.get("worker")} for k, j in jobs.items()},
        },
    )
    return path


def run_two_workers(root: Path) -> dict[str, Any]:
    root = Path(root)
    Paths(root).ensure()
    hashes = freeze_hashes(root)
    freeze_rec = {
        "type": "directional_freeze",
        "created_at": datetime.now(UTC).isoformat(),
        "hashes": hashes,
        "gates": str(GATES_REL),
        "batch": str(BATCH_REL),
        "git": git_sha(root),
        "note": "Frozen before Worker A/B results.",
    }
    atomic_write_json(knowledge_dir(root) / "experiments" / "directional_freeze.json", freeze_rec)
    append_jsonl(knowledge_dir(root) / "index.jsonl", freeze_rec)

    console.print("WORKER A: H004 signed baseline from cache (no tape reread)")
    baseline = write_h004_signed_baseline(root)
    console.print(f"baseline → {baseline.get('path')}")

    man = load_manifest(root)
    all_days = sorted(
        d
        for d, rec in man.get("files", {}).items()
        if rec.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
    )
    inst = pl.read_parquet(Paths(root).normalized / "massive-flat" / "instruments.parquet")

    state = load_state(root, BATCH_ID)
    if state and state.get("freeze") and state["freeze"] != hashes:
        raise RuntimeError("Frozen YAML changed after launch; refusing to resume. New batch required.")
    if not state:
        state = _init_state(root, hashes)
    catalog = _job_catalog()
    for d in state.get("derived") or []:
        if d["id"] not in {c["id"] for c in catalog}:
            parent = state["jobs"].get(d["parent"]) or {}
            catalog.append(
                {
                    "id": d["id"],
                    "worker": parent.get("worker", "B"),
                    "family_id": parent.get("family_id", d["parent"]),
                    "invert": True,
                    "parent": d["parent"],
                }
            )

    stages = ["fast", "falsify", "broad", "full"]
    for stage in stages:
        if stage != "fast":
            survivors = [c for c in catalog if (state["jobs"].get(c["id"]) or {}).get("decision") == "PASS"]
            pending_derived = [
                c
                for c in catalog
                if c.get("invert")
                and (state["jobs"].get(c["id"]) or {}).get("decision") not in {"PASS", "KILL"}
            ]
            catalog_stage = survivors + pending_derived
            if not catalog_stage:
                console.print(f"No survivors for stage={stage}")
                if stage == "falsify":
                    continue
                break
        else:
            catalog_stage = [c for c in catalog if not c.get("invert")]
        panel = _panel_for(stage, all_days)
        console.print(f"\n=== STAGE {stage} n_days={len(panel)} jobs={[c['id'] for c in catalog_stage]} ===")
        run_shared_stage(root, state, stage=stage, panel=panel, catalog=catalog_stage, instruments=inst)
        if stage == "fast":
            a_alive = [
                jid
                for jid, j in state["jobs"].items()
                if j.get("worker") == "A" and j.get("decision") == "PASS"
            ]
            if not a_alive:
                state["h004_no_directional_edge"] = True
                console.print("H004 = VALIDATED VOLATILITY PHENOMENON / NO DIRECTIONAL EDGE FOUND")
                for jid, j in state["jobs"].items():
                    if j.get("worker") == "A" and j.get("decision") != "PASS":
                        j["h004_stop"] = True
                catalog = [c for c in catalog if c.get("worker") != "A"]
        save_state(root, state)

    report = write_consolidated_report(root, state, baseline)
    console.print(f"Report: {report}")
    console.print("Funnel exhausted. No sealed OOS. No orders.")
    state["report"] = str(report)
    save_state(root, state)
    return state
