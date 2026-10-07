"""V4R campaign runner. Separate stores. Does not overwrite V4."""

from __future__ import annotations

import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_edge_lab.discovery.campaign_v4 import CampaignStop, research_days
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json
from quant_edge_lab.discovery.v4r.checkpoint import identity_blob, load_ckpt, write_ckpt
from quant_edge_lab.discovery.v4r.overlay import apply_overlay
from quant_edge_lab.discovery.v4r.search import (
    SEARCH_FEATURES,
    assert_search_features,
    evaluate_d2_d3,
    load_panel,
    persist_d1_tree,
    ridge_train_scaler,
    run_stability,
)
from quant_edge_lab.discovery.v4r.stage1 import (
    atomic_write_parquet,
    build_v4r_day,
    v4r_ckpt_dir,
    v4r_daily_path,
    v4r_day_path,
    v4r_graph_dir,
)
from quant_edge_lab.discovery.v4r.tree import discover_tree, quantile_bank
from quant_edge_lab.discovery.conditional.partition import leaves_to_candidates
from quant_edge_lab.discovery.v4_status import fmt_elapsed
from quant_edge_lab.features.v4r.beta import beta_from_history
from quant_edge_lab.hashing import git_sha, sha256_file, sha256_json
from quant_edge_lab.peers import PeerGraphBuilder, graph_to_record, record_to_graph

console = Console()
MANIFEST_REL = Path("knowledge/campaigns/directional_v4r_manifest.yaml")
GATES_REL = Path("knowledge/campaigns/directional_v4r_gates.yaml")
SCIENCE = "v4r_beta20_fwd15_dayassoc_pathstab_v1"


def load_v4r(root: Path) -> tuple[dict, dict]:
    man = yaml.safe_load((root / MANIFEST_REL).read_text(encoding="utf-8"))
    gates = yaml.safe_load((root / GATES_REL).read_text(encoding="utf-8"))
    return man, gates


def freeze_v4r(root: Path) -> dict[str, str]:
    return {"manifest": sha256_file(root / MANIFEST_REL), "gates": sha256_file(root / GATES_REL), "git": git_sha(root), "science": SCIENCE}


def log_jsonl(root: Path, **kw: Any) -> None:
    kw.setdefault("timestamp", datetime.now(UTC).isoformat())
    kw.setdefault("run_id", "v4r")
    append_jsonl(v4r_ckpt_dir(root) / "events.jsonl", kw)


def fail_dump(root: Path, stage: str, unit: str, exc: BaseException, last: str) -> None:
    rec = {
        "FAILED_STAGE": stage,
        "FAILED_UNIT": unit,
        "EXCEPTION_TYPE": type(exc).__name__,
        "EXCEPTION_MESSAGE": str(exc),
        "FULL_TRACEBACK": traceback.format_exc(),
        "LAST_SUCCESSFUL_CHECKPOINT": last,
        "RESUME_COMMAND": "python -m quant_edge_lab discovery campaign-v4r --execute",
    }
    atomic_write_json(v4r_ckpt_dir(root) / "FAILURE.json", rec)
    console.print(json.dumps(rec, indent=2)[:4000])


def _load_daily(root: Path, days: list[str]) -> list[pl.DataFrame]:
    out = []
    for d in days:
        p = v4r_daily_path(root, d)
        if p.exists():
            out.append(pl.read_parquet(p))
    return out


def run_stage1(root: Path, man: dict, days: list[str], ident: dict) -> None:
    t0 = time.time()
    clock_sum: dict[int, float] = {}
    clock_n: dict[int, int] = {}
    ck = load_ckpt(root, "stage1", ident) or {"done": []}
    done = set(ck.get("done") or [])
    hist_days: list[str] = []
    for i, day in enumerate(days):
        dp = v4r_day_path(root, day)
        if day in done and dp.exists():
            hist_days.append(day)
            continue
        prior = [d for d in hist_days if d < day]
        beta_hist = _load_daily(root, prior[-20:])
        tickers = []
        beta_map = beta_from_history(beta_hist, [])  # filled after we know names from causal
        # first pass: empty map then we still need tickers — load names from causal via build
        clock_mean = {k: clock_sum[k] / clock_n[k] for k in clock_n if clock_n[k]}
        # rebuild beta with actual tickers after a lightweight read
        from quant_edge_lab.features.causal_store import day_path

        if not day_path(root, day).exists():
            if day < man["splits"]["D1"]["start"]:
                done.add(day)
                continue
            raise CampaignStop(f"missing causal_v2 {day}")
        names = pl.scan_parquet(day_path(root, day)).select("instrument_id").unique().collect()["instrument_id"].to_list()
        beta_map = beta_from_history(beta_hist, [str(x) for x in names])
        feat, daily = build_v4r_day(root, day, beta_map=beta_map, clock_mean=clock_mean)
        atomic_write_parquet(feat, dp)
        if daily.height:
            atomic_write_parquet(daily, v4r_daily_path(root, day))
        if "minutes_from_open" in feat.columns and "ret_5m" in feat.columns:
            g = feat.group_by("minutes_from_open").agg(pl.col("ret_5m").mean().alias("m"))
            for row in g.iter_rows(named=True):
                if row["minutes_from_open"] is None or row["m"] is None:
                    continue
                k = int(row["minutes_from_open"])
                clock_sum[k] = clock_sum.get(k, 0.0) + float(row["m"])
                clock_n[k] = clock_n.get(k, 0) + 1
        done.add(day)
        hist_days.append(day)
        write_ckpt(root, "stage1", {"done": sorted(done), "last": day}, ident)
        if i % 1 == 0:
            console.print(
                f"[V4R][STAGE1] day={i+1}/{len(days)} date={day} output_rows={feat.height} elapsed={fmt_elapsed(t0)} checkpoint=OK"
            )
            log_jsonl(root, stage="STAGE1", current_unit=day, completed_units=len(done), total_units=len(days), rows=feat.height, event="day_ok")


def run_graphs(root: Path, man: dict, days: list[str], ident: dict) -> None:
    t0 = time.time()
    gdir = v4r_graph_dir(root)
    pg = man["peer_graph"]
    builder = PeerGraphBuilder(
        history_days=int(pg["history_trading_days"]),
        structural_top=int(pg["structural_top"]),
        leaders_top=int(pg["leaders_top"]),
        lag_minutes=int(pg["lag_minutes"]),
        q_edge=float(pg["q_edge"]),
    )
    from quant_edge_lab.discovery.campaign_v4_runner import _week_monday

    d1s = man["splits"]["D1"]["start"]
    weeks = sorted({_week_monday(d) for d in days if d >= d1s})
    ck = load_ckpt(root, "graphs", ident) or {"done": []}
    done = set(ck.get("done") or [])
    for wi, as_of in enumerate(weeks):
        gp = gdir / f"as_of={as_of}.json"
        if as_of in done and gp.exists():
            g = record_to_graph(json.loads(gp.read_text(encoding="utf-8")))
        else:
            prior = [x for x in days if x < as_of][-int(pg["history_trading_days"]) :]
            frames = []
            for d in prior:
                p = v4r_day_path(root, d)
                if p.exists():
                    frames.append(pl.read_parquet(p, columns=["trading_date", "instrument_id", "decision_ts", "resid_ret_5m"]))
            hist = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
            g = builder.fit(hist, as_of)
            if g.history_end and g.history_end >= as_of:
                raise CampaignStop(f"PIT graph {as_of}")
            rec = graph_to_record(g, cfg_hash=ident["cfg_hash"])
            atomic_write_json(gp, rec)
            done.add(as_of)
            write_ckpt(root, "graphs", {"done": sorted(done), "last": as_of}, ident)
            console.print(
                f"[V4R][GRAPH] week={wi+1}/{len(weeks)} asof={as_of} valid_edges={len(g.edges)} n_tests={g.lineage.get('n_tests')} elapsed={fmt_elapsed(t0)}"
            )
        # overlay days until next week
        nxt = weeks[wi + 1] if wi + 1 < len(weeks) else "9999-99-99"
        for d in days:
            if d < as_of or d >= nxt:
                continue
            from quant_edge_lab.discovery.v4r.stage1 import v4r_peer_day_path

            if v4r_peer_day_path(root, d).exists():
                continue
            apply_overlay(root, d, g, identity=ident)


def run_campaign_v4r(root: Path, *, smoke_days: list[str] | None = None) -> dict[str, Any]:
    t0 = time.time()
    man, gates = load_v4r(root)
    fr = freeze_v4r(root)
    feats = assert_search_features(man)
    console.print(f"V4R SEARCH_FEATURES={feats}")
    ident = identity_blob(cfg_hash=sha256_json({"m": fr["manifest"], "g": fr["gates"], "s": SCIENCE}), git=fr["git"], manifest=fr["manifest"], gates=fr["gates"], science=SCIENCE)
    last = "none"
    try:
        write_ckpt(root, "preflight", {"freeze": fr}, ident)
        last = "preflight"
        days = smoke_days if smoke_days is not None else research_days(root)
        run_stage1(root, man, days, ident)
        last = "stage1"
        run_graphs(root, man, days, ident)
        last = "graphs"
        d1_days = [d for d in days if man["splits"]["D1"]["start"] <= d <= man["splits"]["D1"]["end"]]
        d2_days = [d for d in days if man["splits"]["D2"]["start"] <= d <= man["splits"]["D2"]["end"]]
        d3_days = [d for d in days if man["splits"]["D3"]["start"] <= d <= man["splits"]["D3"]["end"]]
        if smoke_days is not None:
            d1_days, d2_days, d3_days = _smoke_splits(days)
        cols = list(dict.fromkeys(feats + ["trading_date", "instrument_id", "decision_ts", "ticker", man["targets"]["primary"], "future_raw_15m", "resid_ret_5m"] + list(__import__("quant_edge_lab.peers", fromlist=["GRAPH_FEATURE_COLS"]).GRAPH_FEATURE_COLS)))
        d1 = load_panel(root, d1_days, cols)
        if d1.height == 0:
            raise CampaignStop("STOP_D no D1 rows")
        target = man["targets"]["primary"]
        qbank = quantile_bank(d1, feats, list(gates["tree"]["quantile_grid"]))
        tree = discover_tree(
            d1,
            features=feats,
            target=target,
            max_depth=int(gates["tree"]["max_depth"]),
            node_alpha=float(gates["tree"]["node_alpha"]),
            min_ticker_days=int(gates["leaf"]["min_ticker_days"] if smoke_days is None else 30),
            min_parent_frac=float(gates["leaf"]["min_parent_frac"]),
            quantile_cuts=qbank,
        )
        cands = leaves_to_candidates(tree, target=target, sample="D1")
        persist_d1_tree(root, tree, cands, ident)
        last = "d1_tree"
        console.print(f"[V4R][D1] candidates={len(cands)} rows={d1.height}")
        if not cands:
            return _final(root, ident, {"n_candidates": 0, "stop": "STOP_D_no_d1_candidates", "survivors": []}, t0, fr)
        stab_gates = dict(gates)
        if smoke_days is not None:
            stab_gates = {**gates, "stability": {**gates["stability"], "n_subsamples": 4, "subsample_days": max(len(d1_days) // 2, 1), "min_path_rediscovery": 0.0, "min_sign_agree": 0.0}}
        stab = run_stability(root, d1, cands, man=man, gates=stab_gates, feats=feats, target=target, ident=ident, t0=t0)
        last = "stability"
        stable = stab["stable"]
        if not stable:
            return _final(root, ident, {"n_candidates": len(cands), "stability": stab["stats"], "stop": "STOP_E_no_stable", "survivors": []}, t0, fr)
        d2 = load_panel(root, d2_days, cols) if d2_days else d1
        d3 = load_panel(root, d3_days, cols) if d3_days else d1
        ev = evaluate_d2_d3(d2, d3, stable, gates=gates if smoke_days is None else {**gates, "leaf": {**gates["leaf"], "min_ticker_days": 5, "min_trading_days": 2, "min_tickers": 2}, "economic_floor_abs_mean": 0.0, "d3": {**gates["d3"], "abs_mean_floor": 0.0}}, man=man, qbank=qbank, root=root, ident=ident)
        last = "d2d3"
        num = [f for f in feats if f in d1.columns and d1[f].dtype in (pl.Float32, pl.Float64)]
        lin = None
        if num and d1.height > 40:
            art = ridge_train_scaler(d1.select(num).to_numpy(), d1[target].to_numpy().astype(float), list(map(float, gates["linear"]["lambda_grid"])))
            art["features"] = num
            atomic_write_json(v4r_ckpt_dir(root) / "ridge_scaler.json", art)
            from quant_edge_lab.discovery.linear import predict_ridge

            coef, mu, sd = np.array(art["coef"]), np.array(art["mu"]), np.array(art["sd"])
            if d3.height:
                pred = predict_ridge(d3.select(num).to_numpy(), coef, mu, sd)
                lin = {"lambda": art["lambda"], "d3_decile_spread": decile_spread(pred, d3[target].to_numpy().astype(float)), "n_features": len(num), "scaler": "train_only"}
        out = {"n_candidates": len(cands), "n_stable": len(stable), "stability": stab["stats"], **ev, "linear": lin}
        return _final(root, ident, out, t0, fr)
    except Exception as exc:
        fail_dump(root, last, last, exc, last)
        raise


def _smoke_splits(days: list[str]) -> tuple[list[str], list[str], list[str]]:
    n = len(days)
    i1, i2 = max(int(n * 0.4), 1), max(int(n * 0.6), 2)
    return days[:i1], days[i1:i2], days[i2:]


def _final(root: Path, ident: dict, payload: dict, t0: float, fr: dict) -> dict[str, Any]:
    payload = {
        **payload,
        "freeze": fr,
        "identity": ident,
        "elapsed_sec": time.time() - t0,
        "status": "SIGNAL_ONLY",
        "sealed_oos": "not_opened",
        "RESEARCH_PASS": len(payload.get("survivors") or []),
    }
    write_ckpt(root, "final", payload, ident)
    path = root / "reports" / "v4r" / "final.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    surv = payload.get("survivors") or []
    lines = [
        "# V4R final report (SIGNAL_ONLY)",
        "",
        "SEALED OOS NOT OPENED.",
        "",
        f"RESEARCH PASS: **{len(surv)}**",
        f"stop: {payload.get('stop')}",
        f"D1 candidates: {payload.get('n_candidates')}",
        f"stable: {payload.get('n_stable')}",
        f"elapsed_sec: {payload.get('elapsed_sec')}",
        "",
        "## FEATURE DIAGNOSTIC",
        "",
        "See stability stats (path rediscovery, not trading).",
        "",
        "## CANDIDATE RULE / STABLE / D2 / D3 / RESEARCH PASS",
        "",
        "```json",
        json.dumps({k: payload.get(k) for k in ("d2", "d3", "survivors", "subthreshold", "stop", "linear") if k in payload}, indent=2, default=str)[:12000],
        "```",
        "",
        "V4R: ZERO VALIDATED DIRECTIONAL RULES" if not surv else "See survivors table.",
        "",
        "Did V4R discover a concrete directional relationship that could realistically deserve further execution research?",
        "",
        "NO" if not surv and not payload.get("subthreshold") else ("PARTIAL — validated phenomenon but not trading-edge candidate" if payload.get("subthreshold") and not surv else "YES"),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    payload["report"] = str(path)
    console.print(f"report={path} RESEARCH_PASS={len(surv)} SIGNAL_ONLY sealed OOS not opened")
    return payload
