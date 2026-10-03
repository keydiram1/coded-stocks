"""Directional campaign v3. News PIT + V2 states + XS residuals. Does not modify V2."""

from __future__ import annotations

import traceback
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta, time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from rich.table import Table

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
from quant_edge_lab.data.massive.news import (
    CANONICAL_UTC,
    PRIMARY_LAG_MIN,
    attach_cross_section,
    ensure_news_cache,
    load_insights,
    validate_news_cache,
)
from quant_edge_lab.discovery.campaign_v2_runner import _ingest_hist
from quant_edge_lab.discovery.campaign_v3_eval import eval_v3_job, load_v3_jobs
from quant_edge_lab.discovery.dir_gates import decide_directional, frac_days_agree, load_gates_from
from quant_edge_lab.discovery.eval_batch10 import med_from_hist, typ_from_hist
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json, knowledge_dir
from quant_edge_lab.discovery.outcomes_vector import forward_returns_vectorized
from quant_edge_lab.discovery.runner import _batch_dir, load_state, save_state
from quant_edge_lab.discovery.stage_stats import load_job_returns, summarize_forwards
from quant_edge_lab.discovery.stages import select_falsify_days, select_stage_days
from quant_edge_lab.features.causal_store import compute_day_features, day_path
from quant_edge_lab.hashing import git_sha, sha256_file
from quant_edge_lab.universe.filters import add_session_columns

console = Console()
BATCH_ID = "dir-campaign-v3"
MANIFEST_REL = Path("knowledge/campaigns/directional_v3_manifest.yaml")
GATES_REL = Path("knowledge/campaigns/directional_v3_gates.yaml")
HORIZONS = ["15m", "30m", "60m", "120m"]
EXPECTED_MANIFEST = "a2b513f282e0b4cb70627cc724ef05cc28d2d892e70bb93e8a145e8f888fcfa6"
EXPECTED_GATES = "3e41f0222ccc38d168b862169a384f1055f60c68f2934889959cdd40d1e5ee0f"
XR_FROZEN_IDS = {
    "XR.strong_resid.15m",
    "XR.weak_resid.15m",
    "XR.A.impulse_pb.xs_agree",
    "XR.A.impulse_pb.xs_against",
}
SMOKE_NEWS_JOB_IDS = (
    "VA.A.impulse_pb.pos",
    "N1.accept.15m",
    "NI.multi_pos.15m",
    "NR.pos_strong.15m",
    "BL.news_pos.15m",
)


class SharedInfrastructureError(RuntimeError):
    """Same unexpected exception hit multiple jobs; campaign must stop."""


def assert_frozen_hashes(freeze: dict[str, str]) -> None:
    if freeze["manifest"] != EXPECTED_MANIFEST or freeze["gates"] != EXPECTED_GATES:
        raise RuntimeError(
            f"V3 freeze hash changed; refusing run. got manifest={freeze['manifest']} gates={freeze['gates']}"
        )


def freeze_campaign(root: Path) -> dict[str, str]:
    return {
        "manifest": sha256_file(root / MANIFEST_REL),
        "gates": sha256_file(root / GATES_REL),
        "git": git_sha(root),
    }


def _economic_floor(gates_raw: dict, horizon: str, stage: str) -> float:
    floors = gates_raw.get("economic_floors") or {}
    if horizon in floors:
        base = float(floors[horizon])
    else:
        base = float(gates_raw["stages"][stage]["min_abs_mean_15m"])
    stage_boost = {"fast": 0.0, "falsify": 0.0, "broad": 0.0003, "full": 0.0006}
    if horizon == "15m":
        return float(gates_raw["stages"][stage]["min_abs_mean_15m"])
    return base + stage_boost.get(stage, 0.0)


def _outlier_floor(gates_raw: dict, horizon: str, stage_gate) -> float:
    of = (gates_raw.get("outlier_floors") or {}).get(horizon)
    if of is not None:
        return float(of)
    return float(getattr(stage_gate, "outlier_floor", 0.0008))


def _panel(stage: str, days: list[str]) -> list[str]:
    if stage == "fast":
        return select_stage_days(days, "fast")
    if stage == "falsify":
        return select_falsify_days(days, n=100)
    if stage == "broad":
        return select_stage_days(days, "broad")
    return select_stage_days(days, "full")


def _dashboard(state: dict, stage: str, i: int, n: int, t0: datetime) -> None:
    elapsed = datetime.now(UTC) - t0
    console.print(f"\n=== CAMPAIGN V3  stage={stage}  {i}/{n}  elapsed={elapsed}  [INTERIM] ===")
    table = Table()
    for c in ("ID", "HZ", "EVENTS", "MEAN", "STATUS"):
        table.add_column(c)
    n_run = n_kill = n_pass = n_err = 0
    for jid, job in list(state.get("jobs", {}).items())[:70]:
        st = str(job.get("decision") or job.get("state") or "")
        if job.get("state") == "ERROR":
            n_err += 1
        elif str(st).startswith("KILL"):
            n_kill += 1
        elif str(st).startswith("PASS"):
            n_pass += 1
        else:
            n_run += 1
        hz = job.get("horizon") or "15m"
        m = (job.get("interim_means") or {}).get(hz)
        table.add_row(jid, hz, str(job.get("events") or 0), f"{m:.4%}" if isinstance(m, float) else "—", st[:18])
    console.print(table)
    console.print(f"SUMMARY running={n_run} killed={n_kill} passed={n_pass} errors={n_err} [INTERIM]")


def _outlier_stats(fwd: pl.DataFrame, primary: str) -> dict[str, float | None]:
    g = fwd.filter((pl.col("horizon") == primary) & pl.col("forward_return").is_not_null())
    if g.height < 20:
        return {"outlier_mean": None, "drop_best_day_mean": None}
    r = g["forward_return"].to_numpy()
    q = np.quantile(r, 0.99)
    om = float(np.mean(r[r <= q])) if np.any(r <= q) else float(np.mean(r))
    if "trading_day" in g.columns:
        by = g.group_by("trading_day").agg(pl.col("forward_return").mean().alias("m"))
        if by.height:
            best = by.sort("m", descending=True)["trading_day"][0]
            dropped = g.filter(pl.col("trading_day") != best)
            dm = float(dropped["forward_return"].mean()) if dropped.height else None
        else:
            dm = om
    else:
        dm = om
    return {"outlier_mean": om, "drop_best_day_mean": dm}


def _finalize_job(root: Path, state: dict, spec: dict, stage: str, gate, gates_raw: dict, *, batch_id: str = BATCH_ID) -> None:
    jid = spec["id"]
    job = state["jobs"][jid]
    hz = spec.get("horizon") or "15m"
    if job.get("state") == "ERROR":
        job["decision"] = None
        return
    if jid in XR_FROZEN_IDS and job.get("decision") == "KILL" and stage != "fast":
        return
    fwd = load_job_returns(_batch_dir(root, batch_id) / "returns", f"{stage}_{jid}")
    stats = summarize_forwards(fwd, primary=hz, abs_primary=False, n_boot=400, bootstrap_primary_only=True)
    job["stage_stats"] = stats
    job["horizon"] = hz
    job["interim_means"] = {h["horizon"]: h["mean"] for h in stats.get("horizons") or []}
    prim = stats.get("primary") or {}
    sample = stats.get("sample") or {}
    extras = _outlier_stats(fwd, hz) if fwd.height else {}
    extras["economic_floor"] = _economic_floor(gates_raw, hz, stage)
    of = _outlier_floor(gates_raw, hz, gate)
    gate2 = gate.model_copy(update={"outlier_floor": of})
    decision, reason = decide_directional(
        mean_primary=prim.get("mean"),
        median_primary=prim.get("median"),
        win_rate=prim.get("win_rate"),
        ticker_days=int(sample.get("ticker_days") or job.get("events") or 0),
        trading_days=len(job.get("dayset") or []),
        top_ticker_share=(max((job.get("ticker_counts") or {}).values()) / max(job.get("events") or 1, 1))
        if job.get("ticker_counts")
        else None,
        frac_days_agree=frac_days_agree(fwd, primary=hz) if fwd.height else None,
        ci=(prim.get("ci95") or prim.get("block_ci95")),
        gate=gate2,
        stage_complete=True,
        extras=extras,
    )
    job["decision"] = decision
    job["decision_reason"] = reason
    job["state"] = decision
    append_jsonl(
        knowledge_dir(root) / "index.jsonl",
        {
            "type": "stage_result",
            "batch_id": BATCH_ID,
            "family_id": spec.get("family"),
            "variant_id": jid,
            "stage": stage,
            "decision": decision,
            "reason": reason,
            "horizon": hz,
            "events": job.get("events"),
            "note": "SIGNAL_ONLY; not an edge",
            "recorded_at": datetime.now(UTC).isoformat(),
        },
    )


def run_stage(
    root: Path,
    state: dict,
    *,
    stage: str,
    panel: list[str],
    jobs: list[dict],
    instruments: pl.DataFrame,
    all_days: list[str],
    insights: pl.DataFrame,
    gates_raw: dict,
    batch_id: str = BATCH_ID,
) -> None:
    gates = load_gates_from(root, GATES_REL)
    gate = gates.stages[stage if stage in gates.stages else "fast"]
    for spec in jobs:
        jid = spec["id"]
        state["jobs"].setdefault(jid, {"id": jid, "kind": spec.get("kind"), "family": spec.get("family"), "horizon": spec.get("horizon")})
        job = state["jobs"][jid]
        if job.get("decision") == "KILL" and stage != "fast":
            continue
        if stage == "fast" or job.get("decision") == "PASS":
            job["stage"] = stage
            job["decision"] = None
            job["processed_days"] = []
            job["events"] = 0
            job["sum_p"] = 0.0
            job["n_p"] = 0
            job["ticker_counts"] = {}
            job["dayset"] = []
            (_batch_dir(root, batch_id) / "returns" / f"{stage}_{jid}").mkdir(parents=True, exist_ok=True)
    hist: dict[str, Any] = {
        "prev_close": {},
        "typ": {},
        "yday_ret": {},
        "yday_high": {},
        "yday_low": {},
        "yday_range": {},
        "yday_clv": {},
        "ret_2d": {},
        "ret_3d": {},
        "ret_5d": {},
        "ah_ret": {},
        "med20": {},
        "pm_ret": {},
        "closes": defaultdict(lambda: deque(maxlen=5)),
        "vol_hist": defaultdict(lambda: deque(maxlen=20)),
        "range_hist": defaultdict(lambda: deque(maxlen=20)),
    }
    t0 = datetime.now(UTC)
    eval_set = set(panel)
    days_sorted = [d for d in all_days if panel and d <= panel[-1]]
    prior = [d for d in days_sorted if d < panel[0]][-25:]
    walk = (prior + [d for d in days_sorted if d >= panel[0]]) if panel else []
    console.print(f"V3 hist walk {len(walk)} calendar days; evaluate {len(panel)}")
    ins_day = insights
    eval_i = 0
    active = [j for j in jobs if (state["jobs"].get(j["id"]) or {}).get("decision") != "KILL"]
    for day in walk:
        bars = pl.read_parquet(local_parquet_path(root, day))
        sessioned = add_session_columns(bars)
        if day in eval_set:
            hist["typ"] = typ_from_hist(hist["vol_hist"])
            hist["med20"] = med_from_hist(hist["range_hist"])
            feat = compute_day_features(sessioned, instruments, hist)
            feat = attach_cross_section(feat)
            d0 = datetime.fromisoformat(day).replace(tzinfo=UTC)
            day_ins = (
                ins_day.filter(
                    (pl.col("published_ts") >= d0 - timedelta(days=2))
                    & (pl.col("published_ts") < d0 + timedelta(days=2))
                )
                if ins_day.height
                else ins_day
            )
            cache = day_path(root, day)
            if feat.height and not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                feat.write_parquet(cache)
            infra: dict[str, list[str]] = {}
            for spec in active:
                jid = spec["id"]
                job = state["jobs"][jid]
                if job.get("state") == "ERROR":
                    continue
                if day in job.get("processed_days", []):
                    continue
                try:
                    ev = eval_v3_job(spec, feat, day_ins, PRIMARY_LAG_MIN, f"{batch_id}-{jid}")
                    if "sentiment_reasoning" in (ev.columns if ev.height else []):
                        raise AssertionError("sentiment_reasoning leaked into events")
                    if ev.height:
                        hz = spec.get("horizon") or "15m"
                        fwd = forward_returns_vectorized(ev, sessioned, HORIZONS, "long")
                        keep = [c for c in ["event_id", "ticker", "session_date", "instrument_id"] if c in ev.columns]
                        meta = ev.select(keep).unique(subset=["event_id"]) if "event_id" in ev.columns else ev.select(keep)
                        fwd = fwd.join(meta, on="event_id", how="left")
                        if "session_date" in fwd.columns:
                            fwd = fwd.with_columns(pl.col("session_date").cast(pl.Utf8).alias("trading_day"))
                        gp = fwd.filter((pl.col("horizon") == hz) & pl.col("forward_return").is_not_null())
                        if gp.height:
                            job["sum_p"] = float(job.get("sum_p") or 0) + float(gp["forward_return"].sum())
                            job["n_p"] = int(job.get("n_p") or 0) + gp.height
                            job["interim_means"] = {hz: job["sum_p"] / job["n_p"]}
                        fwd.write_parquet(_batch_dir(root, batch_id) / "returns" / f"{stage}_{jid}" / f"{day}.parquet")
                        job["events"] = int(job.get("events") or 0) + ev.height
                        if "ticker" in ev.columns:
                            counts = job.setdefault("ticker_counts", {})
                            for t, n in ev.group_by("ticker").len().iter_rows():
                                counts[str(t)] = int(counts.get(str(t), 0)) + int(n)
                        job["dayset"] = list(set(job.get("dayset") or []) | {day})
                    job["processed_days"] = list(set(job.get("processed_days") or []) | {day})
                    job["state"] = "RUNNING"
                except Exception as exc:
                    job["state"] = "ERROR"
                    job["decision"] = None
                    job["error"] = f"{type(exc).__name__}: {exc}"
                    job["decision_reason"] = traceback.format_exc()[-800:]
                    fp = f"{type(exc).__name__}:{type(exc).__name__}"
                    msg = str(exc)
                    if "join_asof" in msg or "SchemaError" in type(exc).__name__ or "join keys" in msg.lower():
                        fp = f"{type(exc).__name__}:join"
                    elif "look-ahead" in msg or "sentiment_reasoning" in msg:
                        fp = f"{type(exc).__name__}:{msg[:80]}"
                    else:
                        fp = f"{type(exc).__name__}:{msg[:120]}"
                    infra.setdefault(fp, []).append(jid)
                    console.print(f"ERROR {jid} day={day}: {job['error']}")
            for fp, ids in infra.items():
                if len(ids) >= 2:
                    save_state(root, state)
                    raise SharedInfrastructureError(
                        f"STOP: shared infrastructure exception {fp} hit {len(ids)} jobs this day ({day}): {ids[:8]}"
                    )
            if eval_i % 5 == 0:
                _dashboard(state, stage, eval_i, len(panel), t0)
                save_state(root, state)
            eval_i += 1
        _ingest_hist(sessioned, hist)
    for spec in active:
        _finalize_job(root, state, spec, stage, gate, gates_raw, batch_id=batch_id)
    save_state(root, state)
    _dashboard(state, stage, len(panel), len(panel), t0)


def write_report(root: Path, state: dict, news_meta: dict, man: dict) -> Path:
    jobs = state.get("jobs") or {}
    survivors = [j for j, v in jobs.items() if v.get("decision") == "PASS"]
    first = state.get("first_attempt") or {}
    lines = [
        "# Directional campaign v3 (SIGNAL_ONLY)",
        "",
        "V2 was not modified. Not sealed OOS. Not trading.",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Freeze: {state.get('freeze')}",
        f"Canonical join dtype: {CANONICAL_UTC}",
        "",
        "## EXECUTIVE SUMMARY",
        "",
        f"Jobs registered: {len(man.get('jobs') or [])}. Survivors last stage: {survivors or 'none'}.",
        "",
        "## FIRST ATTEMPT",
        "",
        first.get(
            "note",
            "53 News jobs ERROR because timezone dtypes differed (decision_ts naive vs latest_news_available_at UTC).",
        ),
        "",
        "XR jobs completed in the first attempt and were not rerun:",
        "",
    ]
    xr_snap = (first.get("jobs") or {}) if first else jobs
    for jid in sorted(XR_FROZEN_IDS):
        v = xr_snap.get(jid) or jobs.get(jid) or {}
        lines.append(f"- {jid}: {v.get('decision')} n={v.get('events')} {v.get('decision_reason')}")
    n_err1 = sum(1 for jid, v in (first.get("jobs") or jobs).items() if jid not in XR_FROZEN_IDS and (v.get("state") == "ERROR" or not v.get("decision")))
    lines += [
        "",
        f"News-dependent jobs ERROR / not evaluated: {n_err1}.",
        "",
        "## FIX",
        "",
        "Timestamp normalization: both `decision_ts` and `latest_news_available_at` are `datetime[μs, UTC]`",
        "via replace_time_zone on naive UTC instants (no clock shift) and convert_time_zone on aware values.",
        "Regression tests cover join dtypes, PIT boundaries, future-news poison, timezone instant preservation,",
        "and a real-cache smoke join. Shared infrastructure exceptions now STOP the campaign (ERROR, not KILL).",
        "",
        "## RESUMED RUN",
        "",
        "Original frozen hypotheses/gates/stage calendars. Only previously unevaluated news jobs were resumed.",
        "",
        "## DATA / NEWS COVERAGE",
        "",
        f"Insights onset (research period start): {news_meta.get('onset_gte')}",
        f"Insight rows: {news_meta.get('n_insight_rows')}",
        f"Sector PIT: {news_meta.get('sector_note')}",
        f"Cache validation: {yaml.safe_dump(state.get('news_cache_validation') or {}, sort_keys=False)}",
        "",
        yaml.safe_dump({"monthly": news_meta.get("monthly"), "probe": news_meta.get("probe")}, sort_keys=False),
        "",
        "## PIT CONTRACT",
        "",
        "news_available_at = published_utc + 60 minutes (primary). Sensitivity 90/120 reserved for survivors.",
        "sentiment_reasoning stored in news cache only; not used as a feature.",
        "Feature available_at on bars = ts_utc+1m. Decision at confirmation close+1m. Entry next open.",
        "",
        "## REGISTERED HYPOTHESES",
        "",
        f"n={len(man.get('jobs') or [])} in knowledge/campaigns/directional_v3_manifest.yaml",
        "",
        "## STAGE RESULTS (resumed)",
        "",
    ]
    for jid, job in jobs.items():
        tag = " [XR frozen from first attempt]" if jid in XR_FROZEN_IDS else ""
        lines.append(f"- {jid}: {job.get('decision')} hz={job.get('horizon')} n={job.get('events')} {job.get('decision_reason')}{tag}")
    lines += [
        "",
        "## SURVIVORS",
        "",
        (yaml.safe_dump(survivors) if survivors else "None passed frozen dir_gates_v3. Acceptable."),
        "",
        "## FAILURES",
        "",
        "See stage results. Failed variants are not hidden. ERROR is not KILL.",
        "",
        "## NEWS INCREMENTAL VALUE",
        "",
        "Compare VA.*.none vs VA.*.pos/neg and BL.news_* vs N* families in resumed stage results.",
        "",
        "## CROSS-SECTIONAL INCREMENTAL VALUE",
        "",
        "XR.* results are from the first attempt (valid). No PIT sectors.",
        "",
        "## ROBUSTNESS",
        "",
        "Outlier-robust and concentration gates applied from falsify onward. Lag 90/120 only if a job PASSed fast.",
        "",
        "## DATA LIMITATIONS",
        "",
        "No first_seen_at. Hourly feed recency. Insights era shorter than 2021–2026 bar panel. No PIT sectors.",
        "Massive tagging is multi-ticker-heavy. Reasoning may cite already-printed prices (audit only).",
        "",
        "## NEXT RESEARCH STEP",
        "",
        "Do not weaken gates. Do not LLM-classify the corpus unless a survivor is timing-robust at +60m.",
        "",
        "No sealed OOS. No orders.",
    ]
    path = root / "reports" / "discovery" / "directional_campaign_v3.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(
        knowledge_dir(root) / "experiments" / "directional_campaign_v3_summary.json",
        {
            "report": str(path),
            "freeze": state.get("freeze"),
            "survivors": survivors,
            "news_onset": news_meta.get("onset_gte"),
            "resumed_news_jobs": True,
        },
    )
    return path


def _snapshot_first_attempt(state: dict) -> None:
    if state.get("first_attempt"):
        return
    jobs = state.get("jobs") or {}
    state["first_attempt"] = {
        "note": (
            "FIRST ATTEMPT: 53 News jobs ERROR because timezone dtypes differed "
            "(decision_ts datetime[μs] naive vs latest_news_available_at datetime[μs, UTC]). "
            "XR jobs completed and remain valid."
        ),
        "jobs": {
            jid: {
                "decision": v.get("decision"),
                "state": v.get("state"),
                "events": v.get("events"),
                "error": v.get("error"),
                "decision_reason": (v.get("decision_reason") or "")[:500],
            }
            for jid, v in jobs.items()
        },
    }


def _reset_unevaluated_job(job: dict) -> None:
    job["stage"] = "fast"
    job["decision"] = None
    job["state"] = None
    job["error"] = None
    job["decision_reason"] = None
    job["processed_days"] = []
    job["events"] = 0
    job["sum_p"] = 0.0
    job["n_p"] = 0
    job["ticker_counts"] = {}
    job["dayset"] = []
    job["interim_means"] = {}
    job["stage_stats"] = None


def run_campaign_v3(
    root: Path,
    *,
    smoke_days: int = 0,
    job_ids: list[str] | None = None,
    batch_id: str | None = None,
    resume_unevaluated: bool = False,
) -> dict[str, Any]:
    bid = batch_id or BATCH_ID
    root = Path(root)
    Paths(root).ensure()
    freeze = freeze_campaign(root)
    assert_frozen_hashes(freeze)
    man, jobs = load_v3_jobs(root)
    console.print(f"FROZEN v3 n_jobs={len(jobs)} manifest={freeze['manifest'][:16]}… gates={freeze['gates'][:16]}… git={freeze['git']}")
    console.print(f"Canonical timestamp dtype={CANONICAL_UTC}. Hashes unchanged vs original V3 freeze.")
    news_meta = ensure_news_cache(root)
    if not news_meta.get("ok"):
        console.print(f"STOP: {news_meta.get('reason')}")
        path = root / "reports" / "discovery" / "directional_campaign_v3.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# V3 STOP\n\n{news_meta.get('reason')}\n\n{yaml.safe_dump(news_meta)}\n", encoding="utf-8")
        return {"stop": True, "news": news_meta, "report": str(path)}
    cache_stats = validate_news_cache(root)
    insights = load_insights(root)
    if insights.height == 0:
        return {"stop": True, "reason": "empty insights cache"}
    gates_raw = yaml.safe_load((root / GATES_REL).read_text(encoding="utf-8"))
    man_days = load_manifest(root)
    all_days = sorted(
        d for d, r in man_days.get("files", {}).items() if r.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
    )
    onset = news_meta["onset_gte"]
    news_days = [d for d in all_days if d >= onset]
    console.print(f"News period {onset}→{news_days[-1] if news_days else None} n_days={len(news_days)}")
    console.print(f"Stage sizes fast={len(select_stage_days(news_days,'fast'))} falsify={len(select_falsify_days(news_days,n=100))} broad={len(select_stage_days(news_days,'broad'))} full={len(news_days)}")
    if len(news_days) < 40 and not smoke_days:
        return {"stop": True, "reason": f"news-period trading days {len(news_days)} too small"}
    inst = pl.read_parquet(Paths(root).normalized / "massive-flat" / "instruments.parquet")
    state = load_state(root, bid)
    if state and state.get("freeze") and state["freeze"] != freeze:
        raise RuntimeError("V3 freeze hash changed; refusing resume.")
    if not state:
        state = {"batch_id": bid, "freeze": freeze, "jobs": {}, "news_onset": onset, "created_at": datetime.now(UTC).isoformat()}
    state["freeze"] = freeze
    state["news_cache_validation"] = cache_stats
    catalog = list(jobs)
    if job_ids is not None:
        want = set(job_ids)
        catalog = [j for j in catalog if j["id"] in want]
    elif resume_unevaluated:
        _snapshot_first_attempt(state)
        uneval = []
        for spec in catalog:
            jid = spec["id"]
            if jid in XR_FROZEN_IDS:
                continue
            job = state["jobs"].setdefault(jid, {"id": jid, "kind": spec.get("kind"), "family": spec.get("family"), "horizon": spec.get("horizon")})
            if job.get("decision") == "KILL" and job.get("state") != "ERROR":
                continue
            _reset_unevaluated_job(job)
            uneval.append(spec)
        catalog = uneval
        console.print(f"Resuming {len(catalog)} unevaluated news jobs; XR frozen={sorted(XR_FROZEN_IDS)}")
        if len(catalog) != 53 and not smoke_days:
            console.print(f"WARNING: expected 53 unevaluated jobs, got {len(catalog)}")
    stages = ("fast",) if smoke_days else ("fast", "falsify", "broad", "full")
    for stage in stages:
        if stage == "fast":
            run_jobs = catalog
        else:
            run_jobs = [j for j in catalog if (state["jobs"].get(j["id"]) or {}).get("decision") == "PASS"]
            if not run_jobs:
                console.print(f"No survivors for {stage}")
                continue
        panel = _panel(stage, news_days)
        if smoke_days:
            panel = panel[:smoke_days]
        console.print(f"\n=== STAGE {stage} days={len(panel)} jobs={len(run_jobs)} ===")
        run_stage(
            root,
            state,
            stage=stage,
            panel=panel,
            jobs=run_jobs,
            instruments=inst,
            all_days=all_days,
            insights=insights,
            gates_raw=gates_raw,
            batch_id=bid,
        )
        err_n = sum(1 for j in run_jobs if (state["jobs"].get(j["id"]) or {}).get("state") == "ERROR")
        if smoke_days and err_n:
            raise SharedInfrastructureError(f"STOP: {err_n} jobs ERROR during smoke; not converting to KILL")
    if smoke_days:
        console.print("Smoke complete. SIGNAL_ONLY. Main V3 report not overwritten.")
        save_state(root, state)
        return state
    report = write_report(root, state, news_meta, man)
    state["report"] = str(report)
    save_state(root, state)
    console.print(f"Report: {report}")
    console.print("Campaign v3 exhausted. SIGNAL_ONLY. No sealed OOS.")
    return state
