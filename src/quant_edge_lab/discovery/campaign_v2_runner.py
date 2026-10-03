"""Directional campaign v2 autonomous runner. Shared feature scan. No OOS."""

from __future__ import annotations

import json
import traceback
from collections import defaultdict, deque
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from rich.table import Table

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
from quant_edge_lab.discovery.campaign_v2_families import MODEL_FEATS, eval_job, snapshot_matrix
from quant_edge_lab.discovery.campaign_v2_models import (
    fit_gbm,
    fit_logit,
    fit_ridge,
    predict_gbm,
    predict_logit_proba,
    predict_ridge,
    rank_tails,
    select_coverage,
)
from quant_edge_lab.discovery.dir_gates import decide_directional, frac_days_agree, load_gates_from
from quant_edge_lab.discovery.eval_batch10 import _stamp, med_from_hist, typ_from_hist
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json, knowledge_dir, record_run
from quant_edge_lab.discovery.outcomes_vector import forward_returns_vectorized
from quant_edge_lab.discovery.runner import _batch_dir, load_state, save_state
from quant_edge_lab.discovery.stage_stats import load_job_returns, summarize_forwards
from quant_edge_lab.discovery.stages import select_falsify_days, select_stage_days
from quant_edge_lab.features.causal_store import assert_inputs_causal, compute_day_features, day_path
from quant_edge_lab.hashing import git_sha, sha256_file
from quant_edge_lab.pipeline_flat import _update_hist
from quant_edge_lab.statistics.robustness import robustness_tables
from quant_edge_lab.universe.filters import add_session_columns

console = Console()
HORIZONS = ["1m", "5m", "15m", "30m", "60m"]
BATCH_ID = "dir-campaign-v2"
MANIFEST_REL = Path("knowledge/campaigns/directional_v2_manifest.yaml")
GATES_REL = Path("knowledge/campaigns/directional_v2_gates.yaml")
SNAP_TIMES = [time(10, 0), time(11, 0), time(12, 0), time(13, 0), time(14, 0)]


def freeze_campaign(root: Path) -> dict[str, str]:
    return {
        "manifest": sha256_file(root / MANIFEST_REL),
        "gates": sha256_file(root / GATES_REL),
        "git": git_sha(root),
    }


def load_manifest_jobs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = yaml.safe_load((root / MANIFEST_REL).read_text(encoding="utf-8"))
    return raw, list(raw["jobs"])


def _panel(stage: str, all_days: list[str]) -> list[str]:
    if stage == "fast":
        return select_stage_days(all_days, "fast")
    if stage == "falsify":
        return select_falsify_days(all_days, n=100)
    if stage == "broad":
        return select_stage_days(all_days, "broad")
    return select_stage_days(all_days, "full")


def _ingest_hist(sessioned: pl.DataFrame, hist: dict[str, Any]) -> None:
    """Advance causal previous-session state from a completed calendar day."""
    rth = sessioned.filter(pl.col("is_rth"))
    ah = sessioned.filter((pl.col("time_et") >= time(16, 0)) & (pl.col("time_et") < time(20, 0)))
    pm = sessioned.filter(pl.col("is_premarket"))
    if rth.height:
        daily = rth.group_by("instrument_id").agg(
            pl.col("high").max().alias("h"),
            pl.col("low").min().alias("l"),
            pl.col("close").last().alias("c"),
        )
        pc_map = hist["prev_close"]
        for row in daily.iter_rows(named=True):
            iid = row["instrument_id"]
            pc = pc_map.get(iid)
            if pc and pc > 0:
                hist["range_hist"][iid].append((float(row["h"]) - float(row["l"])) / pc)
                hist.setdefault("yday_ret", {})[iid] = float(row["c"]) / pc - 1.0
            _update_multi_day(hist, iid, float(row["c"]), float(row["h"]), float(row["l"]), pc)
        if pm.height:
            pmc = pm.group_by("instrument_id").agg(pl.col("close").last().alias("p"))
            for row in pmc.iter_rows(named=True):
                pc = pc_map.get(row["instrument_id"])
                if pc:
                    hist.setdefault("pm_ret", {})[row["instrument_id"]] = float(row["p"]) / pc - 1
        if ah.height:
            ahc = ah.group_by("instrument_id").agg(pl.col("close").last().alias("a"))
            rc = {r["instrument_id"]: r["c"] for r in daily.iter_rows(named=True)}
            for row in ahc.iter_rows(named=True):
                rcv = rc.get(row["instrument_id"])
                if rcv:
                    hist.setdefault("ah_ret", {})[row["instrument_id"]] = float(row["a"]) / rcv - 1
    _update_hist(sessioned, hist["prev_close"], hist["vol_hist"])


def _update_multi_day(hist: dict[str, Any], iid: str, close: float, high: float, low: float, prev: float | None) -> None:
    hist.setdefault("closes", defaultdict(lambda: deque(maxlen=5)))
    q = hist["closes"][iid]
    q.append(close)
    hist.setdefault("yday_high", {})[iid] = high
    hist.setdefault("yday_low", {})[iid] = low
    hist.setdefault("yday_range", {})[iid] = (high - low) / close if close else None
    hist.setdefault("yday_clv", {})[iid] = (close - low) / (high - low + 1e-12)
    hist.setdefault("prev_close", {})[iid] = close
    cl = list(q)
    if len(cl) >= 2 and cl[-2]:
        hist.setdefault("ret_2d", {})[iid] = cl[-1] / cl[-2] - 1
    if len(cl) >= 3 and cl[-3]:
        hist.setdefault("ret_3d", {})[iid] = cl[-1] / cl[-3] - 1
    if len(cl) >= 5 and cl[-5]:
        hist.setdefault("ret_5d", {})[iid] = cl[-1] / cl[-5] - 1


def _dashboard(state: dict, stage: str, i: int, n: int, t0: datetime) -> None:
    elapsed = datetime.now(UTC) - t0
    console.print(f"\n=== CAMPAIGN V2  stage={stage}  {i}/{n}  elapsed={elapsed}  [INTERIM] ===")
    table = Table()
    for c in ("ID", "KIND", "EVENTS", "MEAN15", "STATUS"):
        table.add_column(c)
    n_run = n_kill = n_pass = n_err = 0
    for jid, job in list(state.get("jobs", {}).items())[:80]:
        st = str(job.get("decision") or job.get("state") or "")
        if st == "ERROR" or job.get("state") == "ERROR":
            n_err += 1
        elif str(st).startswith("KILL"):
            n_kill += 1
        elif str(st).startswith("PASS"):
            n_pass += 1
        else:
            n_run += 1
        m = (job.get("interim_means") or {}).get("15m")
        table.add_row(jid, str(job.get("kind") or ""), str(job.get("events") or 0), f"{m:.4%}" if isinstance(m, float) else "—", st[:18])
    console.print(table)
    console.print(f"SUMMARY running={n_run} killed={n_kill} passed={n_pass} errors={n_err} [INTERIM]")


def _algo_jobs(jobs: list[dict]) -> list[dict]:
    return [j for j in jobs if j.get("kind") in {"algo", "baseline"} and not str(j["id"]).startswith("K1") and not str(j["id"]).startswith("K2") and not str(j["id"]).startswith("K3") and not str(j["id"]).startswith("L.")]


def _is_matrix_job(j: dict) -> bool:
    k = j.get("kind")
    jid = j["id"]
    if jid.startswith("K1.") or jid.startswith("K2.") or jid.startswith("K3.") or jid.startswith("L."):
        return True
    return False


def run_stage(root: Path, state: dict, *, stage: str, panel: list[str], jobs: list[dict], instruments: pl.DataFrame, all_days: list[str]) -> None:
    gates = load_gates_from(root, GATES_REL)
    gate = gates.stages[stage if stage in gates.stages else "fast"]
    algo = [j for j in jobs if not _is_matrix_job(j)]
    matrix_jobs = [j for j in jobs if _is_matrix_job(j)]
    for spec in jobs:
        jid = spec["id"]
        state["jobs"].setdefault(
            jid,
            {"id": jid, "kind": spec.get("kind"), "family": spec.get("family"), "processed_days": [], "events": 0, "ticker_counts": {}, "dayset": [], "params": spec},
        )
        job = state["jobs"][jid]
        if job.get("decision") == "KILL":
            continue
        job["stage"] = stage
        job["days_total"] = len(panel)
        job["decision"] = None
        job["processed_days"] = []
        job["events"] = 0
        job["sum_15"] = 0.0
        job["n_15"] = 0
        job["ticker_counts"] = {}
        job["dayset"] = []
        (_batch_dir(root, BATCH_ID) / "returns" / f"{stage}_{jid}").mkdir(parents=True, exist_ok=True)
    mat_dir = _batch_dir(root, BATCH_ID) / "matrix" / stage
    mat_dir.mkdir(parents=True, exist_ok=True)

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
    console.print(f"causal hist walk {len(walk)} calendar days; evaluate {len(panel)} panel days")
    active_algo = list(algo)
    eval_i = 0

    for day in walk:
        bars = pl.read_parquet(local_parquet_path(root, day))
        sessioned = add_session_columns(bars)
        if day in eval_set:
            hist["typ"] = typ_from_hist(hist["vol_hist"])
            hist["med20"] = med_from_hist(hist["range_hist"])
            feat = compute_day_features(sessioned, instruments, hist)
            assert_inputs_causal(feat, MODEL_FEATS)
            cache = day_path(root, day)
            cache.parent.mkdir(parents=True, exist_ok=True)
            if feat.height:
                feat.write_parquet(cache)

            for spec in active_algo:
                jid = spec["id"]
                job = state["jobs"][jid]
                if day in job.get("processed_days", []):
                    continue
                try:
                    ev = eval_job(jid, feat, f"{BATCH_ID}-{jid}")
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
                        fwd.write_parquet(_batch_dir(root, BATCH_ID) / "returns" / f"{stage}_{jid}" / f"{day}.parquet")
                        job["events"] = int(job.get("events") or 0) + ev.height
                        if "ticker" in ev.columns:
                            counts = job.setdefault("ticker_counts", {})
                            for t, n in ev.group_by("ticker").len().iter_rows():
                                counts[str(t)] = int(counts.get(str(t), 0)) + int(n)
                        job["dayset"] = list(set(job.get("dayset") or []) | {day})
                    job["processed_days"] = list(set(job.get("processed_days") or []) | {day})
                    job["days_done"] = len(job["processed_days"])
                    job["state"] = "RUNNING"
                except Exception as exc:
                    job["state"] = "ERROR"
                    job["error"] = f"{type(exc).__name__}: {exc}"
                    job["decision_reason"] = traceback.format_exc()[-800:]

            if matrix_jobs and feat.height:
                snaps = snapshot_matrix(feat, SNAP_TIMES)
                if snaps.height:
                    stamped = _stamp(snaps.with_columns(pl.lit("long").alias("side")), f"{BATCH_ID}-matrix")
                    fwd = forward_returns_vectorized(stamped, sessioned, ["15m"], "long")
                    y = fwd.filter(pl.col("horizon") == "15m").select(["event_id", "forward_return"])
                    cols = [c for c in MODEL_FEATS if c in stamped.columns]
                    mx = stamped.select(["event_id", "ticker", "session_date", "instrument_id", "ts_utc", "time_et"] + cols)
                    mx = mx.join(y, on="event_id", how="inner")
                    mx = mx.with_columns(pl.lit(day).alias("panel_day"))
                    mx.write_parquet(mat_dir / f"{day}.parquet")
            if eval_i % 5 == 0:
                _dashboard(state, stage, eval_i, len(panel), t0)
                save_state(root, state)
            eval_i += 1
        _ingest_hist(sessioned, hist)

    for spec in active_algo:
        _finalize_job(root, state, spec["id"], stage, gate)
    if matrix_jobs:
        _finalize_models(root, state, matrix_jobs, stage, gate, panel)
    save_state(root, state)
    _dashboard(state, stage, len(panel), len(panel), t0)


def _outlier_stats(fwd: pl.DataFrame) -> dict[str, float | None]:
    g = fwd.filter((pl.col("horizon") == "15m") & pl.col("forward_return").is_not_null())
    if g.height < 20:
        return {"outlier_mean": None, "drop_best_day_mean": None}
    r = g["forward_return"].to_numpy()
    q = np.quantile(r, 0.99)
    om = float(np.mean(r[r <= q])) if np.any(r <= q) else float(np.mean(r))
    if "trading_day" in g.columns:
        by = g.group_by("trading_day").agg(pl.col("forward_return").mean().alias("m"), pl.len().alias("n"))
        if by.height:
            best = by.sort("m", descending=True)["trading_day"][0]
            dropped = g.filter(pl.col("trading_day") != best)
            dm = float(dropped["forward_return"].mean()) if dropped.height else None
        else:
            dm = om
    else:
        dm = om
    return {"outlier_mean": om, "drop_best_day_mean": dm}


def _finalize_job(root: Path, state: dict, jid: str, stage: str, gate) -> None:
    job = state["jobs"][jid]
    if job.get("state") == "ERROR":
        return
    fwd = load_job_returns(_batch_dir(root, BATCH_ID) / "returns", f"{stage}_{jid}")
    stats = summarize_forwards(fwd, primary="15m", abs_primary=False, n_boot=400, bootstrap_primary_only=True)
    job["stage_stats"] = stats
    job["interim_means"] = {h["horizon"]: h["mean"] for h in stats.get("horizons") or []}
    prim = stats.get("primary") or {}
    sample = stats.get("sample") or {}
    extras = _outlier_stats(fwd) if fwd.height else {}
    decision, reason = decide_directional(
        mean_primary=prim.get("mean"),
        median_primary=prim.get("median"),
        win_rate=prim.get("win_rate"),
        ticker_days=int(sample.get("ticker_days") or job.get("events") or 0),
        trading_days=int(sample.get("trading_days") or 0),
        top_ticker_share=sample.get("top_ticker_share"),
        frac_days_agree=frac_days_agree(fwd),
        ci=prim.get("bootstrap_ci_95"),
        gate=gate,
        stage_complete=True,
        extras=extras,
    )
    job["decision"] = decision
    job["decision_reason"] = reason
    job["outlier"] = extras
    record_run(root, {"type": "v2_stage", "job": jid, "stage": stage, "decision": decision, "reason": reason, "primary": prim, "freeze": state.get("freeze")})


def _load_matrix(root: Path, stage: str) -> pl.DataFrame:
    d = _batch_dir(root, BATCH_ID) / "matrix" / stage
    files = sorted(d.glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet([str(f) for f in files]).collect()


def _finalize_models(root: Path, state: dict, matrix_jobs: list[dict], stage: str, gate, panel: list[str]) -> None:
    mx = _load_matrix(root, stage)
    if mx.height == 0:
        for spec in matrix_jobs:
            state["jobs"][spec["id"]]["decision"] = "KILL"
            state["jobs"][spec["id"]]["decision_reason"] = "no model matrix"
        return
    days = sorted(mx["panel_day"].unique().to_list())
    n_tr = max(1, int(len(days) * 0.60))
    train_days = set(days[:n_tr])
    test_days = set(days[n_tr:]) or set(days[-1:])
    cols = [c for c in MODEL_FEATS if c in mx.columns]
    tr = mx.filter(pl.col("panel_day").is_in(list(train_days)) & pl.col("forward_return").is_not_null())
    te = mx.filter(pl.col("panel_day").is_in(list(test_days)) & pl.col("forward_return").is_not_null())
    xtr = tr.select(cols).to_numpy()
    ytr = tr["forward_return"].to_numpy()
    ridge = fit_ridge(xtr, ytr, alpha=1.0)
    logit = fit_logit(xtr, ytr, l2=1.0)
    gbm = fit_gbm(xtr, ytr, n_estimators=40, lr=0.1, seed=42)
    xte = te.select(cols).to_numpy()
    yte = te["forward_return"].to_numpy()
    scores = {
        "k1": predict_logit_proba(logit, xte) - 0.5,
        "k2": predict_ridge(ridge, xte),
        "k3": predict_gbm(gbm, xte),
        "raw": te["ret_15m"].to_numpy() if "ret_15m" in te.columns else yte,
    }
    for spec in matrix_jobs:
        jid = spec["id"]
        cov = float(spec.get("coverage") or 1.0)
        if jid.startswith("L.raw"):
            sc = scores["raw"]
            use_rank = True
        elif jid.startswith("L."):
            sc = scores["k2"]
            use_rank = True
        elif jid.startswith("K1."):
            sc = scores["k1"]
            use_rank = False
        elif jid.startswith("K2."):
            sc = scores["k2"]
            use_rank = False
        elif jid.startswith("K3."):
            sc = scores["k3"]
            use_rank = False
        else:
            continue
        if use_rank:
            signed = np.zeros(len(sc))
            # rank within time_et groups
            te_t = te.with_columns(pl.Series("score", sc))
            parts = []
            for g in te_t.partition_by("time_et"):
                s = g["score"].to_numpy()
                lm, sm = rank_tails(s, cov)
                r = np.where(lm, g["forward_return"].to_numpy(), np.where(sm, -g["forward_return"].to_numpy(), np.nan))
                gg = g.with_columns(pl.Series("signed", r)).filter(pl.col("signed").is_not_nan())
                parts.append(gg)
            used = pl.concat(parts) if parts else te_t.head(0)
            rets = used["signed"].to_numpy() if used.height else np.array([])
            tickers = used["ticker"].to_list() if used.height and "ticker" in used.columns else []
        else:
            mask = select_coverage(sc, cov)
            side = np.where(sc >= 0, 1.0, -1.0)
            rets = yte[mask] * side[mask]
            tickers = [te["ticker"][int(i)] for i in np.where(mask)[0]] if "ticker" in te.columns else []
        mean = float(np.mean(rets)) if rets.size else None
        med = float(np.median(rets)) if rets.size else None
        win = float(np.mean(rets > 0)) if rets.size else None
        n_td = len(set(tickers)) if tickers else rets.size
        # fake share
        share = None
        if tickers:
            from collections import Counter
            c = Counter(tickers)
            share = max(c.values()) / sum(c.values())
        job = state["jobs"][jid]
        job["events"] = int(rets.size)
        job["interim_means"] = {"15m": mean}
        job["stage_stats"] = {"primary": {"mean": mean, "median": med, "win_rate": win, "n": int(rets.size)}}
        decision, reason = decide_directional(
            mean_primary=mean,
            median_primary=med,
            win_rate=win,
            ticker_days=int(n_td or 0),
            trading_days=len(test_days),
            top_ticker_share=share,
            frac_days_agree=None,
            ci=None,
            gate=gate,
            stage_complete=True,
        )
        job["decision"] = decision
        job["decision_reason"] = reason + " | model eval on chronological holdout only"
        record_run(root, {"type": "v2_model", "job": jid, "stage": stage, "decision": decision, "reason": job["decision_reason"], "n": int(rets.size), "mean": mean})

    # monotonicity: K1.c05 vs K1.all
    for prefix in ("K1", "K2", "K3"):
        a = state["jobs"].get(f"{prefix}.all")
        b = state["jobs"].get(f"{prefix}.c05")
        if a and b and a.get("decision") == "PASS" and b.get("decision") == "PASS":
            ma = (a.get("interim_means") or {}).get("15m")
            mb = (b.get("interim_means") or {}).get("15m")
            if ma is not None and mb is not None and mb < ma - 0.0002:
                b["decision"] = "KILL"
                b["decision_reason"] = f"selective monotonicity fail c05={mb} < all={ma}"


def write_report(root: Path, state: dict, man: dict) -> Path:
    path = root / "reports" / "discovery" / "directional_campaign_v2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    jobs = state.get("jobs") or {}
    survivors = [k for k, j in jobs.items() if j.get("decision") == "PASS"]
    fails = [k for k, j in jobs.items() if j.get("decision") == "KILL"]
    lines = [
        "# Directional campaign v2 (SIGNAL_ONLY)",
        "",
        "Not sealed OOS. Not trading. Hypotheses were not modified after results.",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Freeze: {state.get('freeze')}",
        "",
        "## EXECUTIVE SUMMARY",
        "",
        f"Jobs: {len(jobs)}. Survivors last stage: {survivors or 'none'}.",
        "",
        "## WHAT WAS TESTED",
        "",
        "Families A–J sequential/state/relative algorithms, K1/K2/K3 selective models, L ranker, registered baselines.",
        "",
        "## FEATURE STORE",
        "",
        f"`{Paths(root).features / 'causal_v2'}` minute features with available_at = bar_ts+1m.",
        "",
        "## SURVIVORS",
        "",
    ]
    if not survivors:
        lines.append("None passed frozen dir_gates_v2. This is an acceptable result.")
    for k in survivors:
        j = jobs[k]
        prim = (j.get("stage_stats") or {}).get("primary") or {}
        lines += [
            f"### {k}",
            f"- events={j.get('events')} mean15={prim.get('mean')} median={prim.get('median')} win={prim.get('win_rate')}",
            f"- {j.get('decision_reason')}",
            "",
        ]
    lines += ["## FAILURES", ""]
    for k in fails:
        lines.append(f"- {k}: {(jobs[k] or {}).get('decision_reason')}")
    lines += [
        "",
        "## DATA INFORMATION GAPS",
        "",
        "If all failed: minute OHLCV cannot observe trade-sign imbalance, NBBO, size, news timestamps, or short availability.",
        "Highest-value next data for THIS project's failures: (1) trades/trade-sign (2) NBBO/spread (3) news timestamps.",
        "",
        "## NEXT RESEARCH STEP",
        "",
        "Do not weaken gates. Do not grid-search thresholds. New paid microstructure only with approval.",
        "",
        "No sealed OOS. No orders.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(knowledge_dir(root) / "experiments" / "directional_campaign_v2_summary.json", {"report": str(path), "freeze": state.get("freeze"), "survivors": survivors})
    return path


def run_campaign_v2(root: Path) -> dict[str, Any]:
    root = Path(root)
    Paths(root).ensure()
    freeze = freeze_campaign(root)
    man, jobs = load_manifest_jobs(root)
    rec = {"type": "campaign_v2_freeze", "freeze": freeze, "created_at": datetime.now(UTC).isoformat(), "n_jobs": len(jobs)}
    atomic_write_json(knowledge_dir(root) / "experiments" / "directional_v2_freeze.json", rec)
    append_jsonl(knowledge_dir(root) / "index.jsonl", rec)
    console.print(f"FROZEN campaign v2 manifest={freeze['manifest'][:16]}… gates={freeze['gates'][:16]}…")

    man_days = load_manifest(root)
    all_days = sorted(
        d for d, r in man_days.get("files", {}).items() if r.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
    )
    inst = pl.read_parquet(Paths(root).normalized / "massive-flat" / "instruments.parquet")
    state = load_state(root, BATCH_ID)
    if state and state.get("freeze") and state["freeze"] != freeze:
        raise RuntimeError("Campaign freeze hash changed; refusing resume.")
    if not state:
        state = {"batch_id": BATCH_ID, "freeze": freeze, "jobs": {}, "created_at": datetime.now(UTC).isoformat()}
    catalog = list(jobs)
    for stage in ("fast", "falsify", "broad", "full"):
        if stage == "fast":
            run_jobs = catalog
        else:
            run_jobs = [j for j in catalog if (state["jobs"].get(j["id"]) or {}).get("decision") == "PASS"]
            if not run_jobs:
                console.print(f"No survivors for {stage}")
                continue
        panel = _panel(stage, all_days)
        console.print(f"\n=== STAGE {stage} days={len(panel)} jobs={[j['id'] for j in run_jobs]} ===")
        run_stage(root, state, stage=stage, panel=panel, jobs=run_jobs, instruments=inst, all_days=all_days)
    report = write_report(root, state, man)
    state["report"] = str(report)
    save_state(root, state)
    console.print(f"Report: {report}")
    console.print("Campaign v2 exhausted. SIGNAL_ONLY. No sealed OOS. No orders.")
    return state
