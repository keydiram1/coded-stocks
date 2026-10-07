"""V4R D1/D2/D3 search: day-aware CIT, path stability, frozen D2/D3, train-only ridge scaler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.campaign_v4 import CampaignStop
from quant_edge_lab.discovery.conditional import AlgorithmCandidate, apply_conditions
from quant_edge_lab.discovery.conditional.partition import leaves_to_candidates
from quant_edge_lab.discovery.conditional.simplify import estimate_candidate, neighbor_ok, simplify_candidate
from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.linear import decile_spread, predict_ridge
from quant_edge_lab.discovery.v4r.overlay import join_overlay
from quant_edge_lab.discovery.v4r.stability import (
    direction_impossible,
    path_key_from_conditions,
    path_key_no_dir,
    rule_text,
    stability_impossible,
    thresholds_similar,
)
from quant_edge_lab.discovery.v4r.stage1 import v4r_ckpt_dir, v4r_day_path, v4r_peer_day_path
from quant_edge_lab.discovery.v4r.tree import discover_tree, quantile_bank, tree_to_dict
from quant_edge_lab.discovery.v4_status import fmt_elapsed, maybe_status
from quant_edge_lab.validation.multiple_testing import white_reality_check
from quant_edge_lab.peers import GRAPH_FEATURE_COLS

SEARCH_FEATURES = [
    "leader_response_gap",
    "resid_rank_change_5m",
    "resid_ret_5m",
    "xs_dispersion_5m",
    "xs_breadth_5m",
    "rvol_5m_tod",
    "activity_acceleration",
    "volume_price_disagreement",
    "same_clock_surprise",
    "H004",
    "impulse",
    "full_state",
    "drop_H004",
    "drop_impulse",
]


def assert_search_features(man: dict) -> list[str]:
    feats = list(man["search_features"])
    if feats != SEARCH_FEATURES:
        raise CampaignStop(f"unexpected search_features {feats}")
    return feats


def load_panel(root: Path, days: list[str], cols: list[str]) -> pl.DataFrame:
    frames = []
    for d in days:
        p = v4r_day_path(root, d)
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        ovp = v4r_peer_day_path(root, d)
        if ovp.exists():
            ov = pl.read_parquet(ovp)
            df = join_overlay(df, ov)
        keep = [c for c in cols if c in df.columns]
        frames.append(df.select(keep))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def persist_d1_tree(root: Path, tree, cands: list[AlgorithmCandidate], ident: dict) -> Path:
    rec = {
        "identity": ident,
        "tree": tree_to_dict(tree),
        "candidates": [
            {
                "id": c.candidate_id,
                "rule": rule_text(c),
                "direction": c.direction,
                "conditions": [x.to_list() for x in c.conditions],
                "target": c.target,
            }
            for c in cands
        ],
    }
    path = v4r_ckpt_dir(root) / "d1_tree.json"
    atomic_write_json(path, rec)
    return path


def _fold_match(cand: AlgorithmCandidate, fold_cands: list[AlgorithmCandidate], fold_df: pl.DataFrame, q_tol: float) -> tuple[bool, bool]:
    """Returns (path_hit, direction_hit)."""
    want = path_key_no_dir(cand.conditions)
    path_hit = False
    dir_hit = False
    for fc in fold_cands:
        if path_key_no_dir(fc.conditions) != want:
            continue
        feats = [c.feature for c in cand.conditions]
        if not thresholds_similar([c.threshold for c in cand.conditions], [c.threshold for c in fc.conditions], fold_df, feats, q_tol):
            continue
        path_hit = True
        if fc.direction == cand.direction:
            dir_hit = True
            break
    return path_hit, dir_hit


def run_stability(
    root: Path,
    d1: pl.DataFrame,
    cands: list[AlgorithmCandidate],
    *,
    man: dict,
    gates: dict,
    feats: list[str],
    target: str,
    ident: dict,
    t0: float,
) -> dict[str, Any]:
    days = sorted(d1["trading_date"].unique().to_list())
    n_sub = int(gates["stability"]["n_subsamples"])
    take_n = int(gates["stability"].get("subsample_days") or max(len(days) // 2, 1))
    take_n = min(take_n, len(days))
    q_tol = float(gates["stability"]["threshold_similarity_q"])
    need_path = float(gates["stability"]["min_path_rediscovery"])
    need_sign = float(gates["stability"]["min_sign_agree"])
    rng = np.random.default_rng(int(man["search"]["seed"]))
    leaf = gates["leaf"]
    ckpt = v4r_ckpt_dir(root) / "stability.json"
    start = 0
    stats = {
        c.candidate_id: {"path_hits": 0, "dir_hits": 0, "impossible": False, "matches": []}
        for c in cands
    }
    if ckpt.exists():
        st = json.loads(ckpt.read_text(encoding="utf-8"))
        if st.get("identity") == ident:
            start = int(st.get("i") or 0)
            stats = st["stats"]
            if st.get("rng"):
                rng.bit_generator.state = st["rng"]
    alive = [c for c in cands if not stats[c.candidate_id]["impossible"]]
    for i in range(start, n_sub):
        if not alive:
            break
        take = rng.choice(days, size=take_n, replace=False).tolist()
        fold = d1.filter(pl.col("trading_date").is_in(take))
        qbank = quantile_bank(fold, feats, list(gates["tree"]["quantile_grid"]))
        t2 = discover_tree(
            fold,
            features=feats,
            target=target,
            max_depth=int(gates["tree"]["max_depth"]),
            node_alpha=float(gates["tree"]["node_alpha"]),
            min_ticker_days=max(20, int(leaf["min_ticker_days"] // 10)),
            min_parent_frac=float(leaf["min_parent_frac"]),
            quantile_cuts=qbank,
        )
        fcands = leaves_to_candidates(t2, target=target, sample="D1_fold")
        done = i + 1
        remaining = n_sub - done
        for c in list(alive):
            s = stats[c.candidate_id]
            ph, dh = _fold_match(c, fcands, fold, q_tol)
            if ph:
                s["path_hits"] += 1
            if dh:
                s["dir_hits"] += 1
            if len(s["matches"]) < 8:
                s["matches"].append({"fold": done, "path": ph, "dir": dh})
            if stability_impossible(s["path_hits"], done, n_sub, need_path) or (
                s["path_hits"] > 0 and direction_impossible(s["dir_hits"], s["path_hits"], remaining, need_sign)
            ):
                s["impossible"] = True
                s["stop_arithmetic"] = {
                    "path_hits": s["path_hits"],
                    "dir_hits": s["dir_hits"],
                    "done": done,
                    "remaining": remaining,
                    "max_path": s["path_hits"] + remaining,
                    "need_path": int(np.ceil(need_path * n_sub)),
                }
        alive = [c for c in cands if not stats[c.candidate_id]["impossible"]]
        if (done % 5 == 0) or done == n_sub or not alive:
            atomic_write_json(
                ckpt,
                {"identity": ident, "i": done, "stats": stats, "rng": rng.bit_generator.state, "n_sub": n_sub},
            )
        maybe_status(
            "v4r_stab",
            f"[V4R][STABILITY] fold={done}/{n_sub} candidate_alive={len(alive)} rss_gb=n/a elapsed={fmt_elapsed(t0)}",
            force=True,
            interval=60,
        )
        if not alive:
            break
    stable = []
    for c in cands:
        s = stats[c.candidate_id]
        freq = s["path_hits"] / n_sub
        agree = (s["dir_hits"] / s["path_hits"]) if s["path_hits"] else 0.0
        s["path_rediscovery"] = freq
        s["direction_agreement"] = agree
        if freq + 1e-12 >= need_path and agree + 1e-12 >= need_sign and not s["impossible"]:
            stable.append(c)
    return {"stats": stats, "stable": stable, "n_sub": n_sub}


def ridge_train_scaler(x: np.ndarray, y: np.ndarray, grid: list[float]) -> dict[str, Any]:
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    n = x.shape[0]
    cut = int(n * 0.7) if n > 50 else n
    xtr, ytr = x[:cut], y[:cut]
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    best_l, best_mse, best_c = grid[0], 1e18, None
    for lam in grid:
        z = np.hstack([np.ones((xtr.shape[0], 1)), (xtr - mu) / sd])
        p = z.shape[1]
        a = z.T @ z + lam * np.eye(p)
        a[0, 0] -= lam
        coef = np.linalg.solve(a, z.T @ ytr)
        pred = predict_ridge(x[cut:] if cut < n else x, coef, mu, sd)
        yy = y[cut:] if cut < n else y
        mse = float(np.mean((pred - yy) ** 2))
        if mse < best_mse:
            best_l, best_mse, best_c = lam, mse, coef
    return {
        "lambda": best_l,
        "coef": best_c.tolist() if best_c is not None else [],
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "n_train": int(cut),
    }


def evaluate_d2_d3(
    d2: pl.DataFrame,
    d3: pl.DataFrame,
    stable: list[AlgorithmCandidate],
    *,
    gates: dict,
    man: dict,
    qbank: dict,
    root: Path,
    ident: dict,
) -> dict[str, Any]:
    leaf = gates["leaf"]
    kw = dict(min_td=int(leaf["min_ticker_days"]), min_days=int(leaf["min_trading_days"]), min_tickers=int(leaf["min_tickers"]), max_conc=float(gates["max_top_ticker_share"]))
    floor = float(gates["d3"]["abs_mean_floor"])
    d2_ok = []
    d2_rows = []
    for c in stable:
        # frozen rule: no refit. simplify is parsimony on D2 (allowed in V4); V4R D2 says no adding/removing.
        est = estimate_candidate(d2, c, **kw)
        rec = {"id": c.candidate_id, "rule": rule_text(c), "d2": est, "gates": {}}
        rec["gates"]["sample"] = not est.get("kill")
        rec["gates"]["floor_quarter"] = est.get("mean") is not None and abs(est["mean"]) >= float(gates["economic_floor_abs_mean"]) * 0.25
        ok = rec["gates"]["sample"] and rec["gates"]["floor_quarter"] and est.get("mean") is not None
        rec["status"] = "D2_PASS" if ok else "KILL"
        d2_rows.append(rec)
        atomic_write_json(v4r_ckpt_dir(root) / f"d2_{c.candidate_id[:12]}.json", {**rec, "identity": ident})
        if ok:
            d2_ok.append((c, est))
    if not d2_ok:
        return {"d2": d2_rows, "d3": [], "survivors": [], "subthreshold": [], "spa": None, "stop": "STOP_F_zero_d2"}
    d3_rows = []
    survivors = []
    sub = []
    for c, e2 in d2_ok:
        e3 = estimate_candidate(d3, c, **kw)
        sign_ok = e2["mean"] is not None and e3["mean"] is not None and (e2["mean"] > 0) == (e3["mean"] > 0)
        trading = bool(sign_ok and not e3.get("kill") and abs(e3["mean"]) >= floor)
        ci_ok = e3.get("se") is not None and e3.get("mean") is not None and abs(e3["mean"]) > 1.96 * float(e3["se"])
        status = "RESEARCH_PASS" if trading else (
            "VALIDATED_SUBTHRESHOLD_PHENOMENON" if sign_ok and not e3.get("kill") and ci_ok else "KILL"
        )
        rec = {"id": c.candidate_id, "rule": rule_text(c), "d2": e2, "d3": e3, "decision": status, "note": "SIGNAL_ONLY"}
        d3_rows.append(rec)
        atomic_write_json(v4r_ckpt_dir(root) / f"d3_{c.candidate_id[:12]}.json", {**rec, "identity": ident})
        if trading:
            survivors.append(rec)
        elif status == "VALIDATED_SUBTHRESHOLD_PHENOMENON":
            sub.append(rec)
    spa = None
    if d3.height and d2_ok:
        all_days = sorted(d3["trading_date"].unique().to_list())
        cols_m = []
        for c, _ in d2_ok:
            from quant_edge_lab.discovery.conditional import signed_target

            subdf = apply_conditions(d3, c.conditions)
            by = {}
            if subdf.height:
                tmp = subdf.with_columns(signed_target(subdf, c.target, c.direction).alias("_y"))
                for row in tmp.group_by("trading_date").agg(pl.col("_y").mean()).iter_rows(named=True):
                    by[row["trading_date"]] = float(row["_y"]) if row["_y"] is not None else float("nan")
            cols_m.append([by.get(d, float("nan")) for d in all_days])
        if cols_m:
            spa = white_reality_check(np.column_stack(cols_m), n_boot=200, seed=int(man["search"]["seed"]))
    return {"d2": d2_rows, "d3": d3_rows, "survivors": survivors, "subthreshold": sub, "spa": spa, "stop": None}
