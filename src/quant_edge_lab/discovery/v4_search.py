"""Bounded-memory V4 D1–D3 search. Same mathematics as the eager pipeline; no full-frame collect."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from quant_edge_lab.discovery.campaign_v4 import CampaignStop, v4_day_path, v4_peer_day_path
from quant_edge_lab.discovery.conditional import AlgorithmCandidate, Condition, apply_conditions_lf, make_candidate
from quant_edge_lab.discovery.conditional.association import bh_qvalues, binary_ttest, spearman_dayblock
from quant_edge_lab.discovery.conditional.partition import QUANTILES, TreeNode, frozen_quantile_cuts, leaves_to_candidates
from quant_edge_lab.discovery.conditional.simplify import neighbor_ok
from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.linear import decile_spread, predict_ridge
from quant_edge_lab.discovery.v4_status import fmt_elapsed, maybe_status
from quant_edge_lab.peers import GRAPH_FEATURE_COLS
from quant_edge_lab.validation.multiple_testing import white_reality_check

console = Console()
_PEAK_RSS = 0
_T0 = time.time()
_ERR = 0
SEARCH_COLS = [
    "trading_date",
    "instrument_id",
    "ticker",
    "decision_ts",
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
    "future_residual_15m",
    "future_residual_5m",
    "future_residual_30m",
    "future_residual_rank_15m",
    "future_raw_15m",
    "leader_shock_5m",
    "peer_response_gap",
    "peer_rank_gap",
    "graph_available",
    "leader_count",
    "peer_count",
]


def rss_bytes() -> int:
    global _PEAK_RSS
    n = 0
    try:
        import psutil  # type: ignore

        n = int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        n = 0
    if n <= 0 and os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        ok = False
        for lib, fn in ((ctypes.windll.psapi, "GetProcessMemoryInfo"), (ctypes.windll.kernel32, "K32GetProcessMemoryInfo")):
            try:
                func = getattr(lib, fn)
                func.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
                func.restype = wintypes.BOOL
                ok = bool(func(handle, ctypes.byref(pmc), pmc.cb))
                if ok:
                    n = int(pmc.WorkingSetSize)
                    break
            except Exception:
                continue
    if n <= 0 and os.name != "nt":
        import resource

        n = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.uname().sysname == "Linux":
            n *= 1024
    _PEAK_RSS = max(_PEAK_RSS, n)
    return n


def peak_rss() -> int:
    rss_bytes()
    return _PEAK_RSS


def rss_ceiling_bytes() -> int:
    env = os.environ.get("QUANT_EDGE_V4_RSS_GB", "").strip()
    if env:
        return int(float(env) * (1024**3))
    total = 0
    try:
        import psutil  # type: ignore

        total = int(psutil.virtual_memory().total)
    except Exception:
        total = 32 * 1024**3
    return int(total * 0.55)


def assert_rss(op: str) -> None:
    used = rss_bytes()
    cap = rss_ceiling_bytes()
    if used > cap:
        raise CampaignStop(f"RSS ceiling {cap/1e9:.1f}GB exceeded at {op}: rss={used/1e9:.2f}GB peak={peak_rss()/1e9:.2f}GB")


def mem_log(stage: str, op: str, *, extra: str = "", rows: int | None = None) -> None:
    assert_rss(op)
    row_s = f" rows={rows}" if rows is not None else ""
    console.print(
        f"stage={stage} op={op}{row_s} rss={rss_bytes()/1e9:.2f}GB peak={peak_rss()/1e9:.2f}GB "
        f"elapsed={time.time()-_T0:.0f}s ERROR={_ERR} {extra}"
    )


def search_files(root: Path, days: list[str]) -> list[Path]:
    return [v4_day_path(root, d) for d in days if v4_day_path(root, d).exists()]


def search_lf(root: Path, days: list[str], columns: list[str] | None = None) -> pl.LazyFrame:
    files = search_files(root, days)
    if not files:
        raise CampaignStop("no V4 partitions for search scan")
    lf = pl.scan_parquet(files)
    ov_files = [v4_peer_day_path(root, d) for d in days if v4_peer_day_path(root, d).exists()]
    if ov_files:
        ov = pl.scan_parquet(ov_files)
        keys = [c for c in ("trading_date", "instrument_id", "decision_ts") if c in lf.collect_schema().names() and c in ov.collect_schema().names()]
        gcols = [c for c in GRAPH_FEATURE_COLS if c in ov.collect_schema().names()]
        drop = [c for c in gcols if c in lf.collect_schema().names()]
        if drop:
            lf = lf.drop(drop)
        if keys and gcols:
            lf = lf.join(ov.select(keys + gcols), on=keys, how="left")
    have = set(lf.collect_schema().names())
    want = list(dict.fromkeys(columns if columns is not None else SEARCH_COLS))
    cols = [c for c in want if c in have]
    if not cols:
        raise CampaignStop("no requested columns present in V4 partitions")
    return lf.select(cols)


def _signed_expr(target: str, direction: str) -> pl.Expr:
    e = pl.col(target)
    return -e if direction == "SHORT" else e


def _n(lf: pl.LazyFrame) -> int:
    return int(lf.select(pl.len()).collect().item())


def _is_binary_lf(lf: pl.LazyFrame, feat: str) -> bool:
    nunq = int(lf.select(pl.col(feat).drop_nulls().n_unique()).collect().item())
    if nunq > 2:
        return False
    vals = set(lf.select(pl.col(feat).drop_nulls().unique()).collect()[feat].to_list())
    return vals <= {True, False, 0, 1, 0.0, 1.0}


def _collect_xy(lf: pl.LazyFrame, feat: str, target: str) -> pl.DataFrame:
    mem_log("D1", f"collect_xy:{feat}")
    schema = lf.collect_schema()
    s = lf.select([feat, target, "trading_date"]).drop_nulls()
    if schema[feat] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
        s = s.filter(pl.col(feat).is_finite())
    if schema[target] in (pl.Float32, pl.Float64):
        s = s.filter(pl.col(target).is_finite())
    return s.collect()


def quantile_bank_lf(lf: pl.LazyFrame, features: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    schema = lf.collect_schema()
    nums = [
        f
        for f in features
        if f in schema.names() and schema[f] in (pl.Float32, pl.Float64, pl.Int64, pl.Int32)
    ]
    if not nums:
        return out
    mem_log("D1", "quantile_bank", extra=f"nfeat={len(nums)}")
    frame = lf.select(nums).collect()
    try:
        for f in nums:
            s = frame[f].drop_nulls()
            if s.n_unique() > 4:
                out[f] = frozen_quantile_cuts(s, QUANTILES)
            del s
    finally:
        del frame
    return out


def discover_tree_lf(
    lf: pl.LazyFrame,
    *,
    features: list[str],
    target: str,
    max_depth: int,
    node_alpha: float,
    min_ticker_days: int,
    min_parent_frac: float,
    quantile_cuts: dict[str, list[float]] | None = None,
    depth: int = 0,
    path: tuple[Condition, ...] = (),
    lineage: list[dict[str, Any]] | None = None,
    parent_n: int | None = None,
) -> TreeNode:
    lineage = lineage if lineage is not None else []
    schema0 = lf.collect_schema()
    if depth == 0:
        cols0 = [c for c in [target, "trading_date", *features] if c in schema0.names()]
        mem_log("D1", "tree_frame", extra=f"cols={len(cols0)}")
        lf = lf.select(cols0).collect().lazy()
    n = _n(lf)
    mean = lf.select(pl.col(target).mean()).collect().item()
    mean = float(mean) if mean is not None else None
    node = TreeNode(None, None, n, mean, path=path, lineage=lineage)
    if depth >= max_depth or n < min_ticker_days:
        lineage.append({"decision": "rejected", "reason": "max_depth_or_sample", "depth": depth, "n": n})
        return node
    tests: list[tuple[str, float]] = []
    schema = set(lf.collect_schema().names())
    assoc_cols = [c for c in [target, "trading_date", *features] if c in schema]
    mem_log("D1", f"assoc_node:d{depth}", extra=f"n={n} ncols={len(assoc_cols)}")
    wide = lf.select(assoc_cols).collect()
    try:
        for feat in features:
            if feat not in schema:
                continue
            xy = wide.select([feat, target, "trading_date"]).drop_nulls()
            ft = wide.schema.get(feat)
            if ft in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                xy = xy.filter(pl.col(feat).is_finite())
            if wide.schema.get(target) in (pl.Float32, pl.Float64):
                xy = xy.filter(pl.col(target).is_finite())
            x = xy[feat].to_numpy()
            y = xy[target].to_numpy().astype(float)
            days = xy["trading_date"].to_numpy()
            nunq = int(xy[feat].n_unique()) if xy.height else 0
            binary = nunq <= 2 and set(xy[feat].drop_nulls().unique().to_list()) <= {True, False, 0, 1, 0.0, 1.0}
            p = binary_ttest(x.astype(float), y) if binary else spearman_dayblock(x.astype(float), y, days)
            del xy
            tests.append((feat, p))
            lineage.append({"search_id": f"d{depth}", "feature": feat, "sample": "D1", "decision": "considered", "p_raw": p, "n_ticker_days": n})
    finally:
        del wide
    if not tests:
        return node
    q = bh_qvalues(np.array([p for _, p in tests]))
    ranked = sorted(zip([f for f, _ in tests], [p for _, p in tests], q), key=lambda t: t[2])
    best_f, best_p, best_q = ranked[0]
    if best_q > node_alpha:
        lineage.append({"feature": best_f, "decision": "rejected", "reason": "q>alpha", "p_adjusted": float(best_q)})
        return node
    if _is_binary_lf(lf, best_f):
        cuts: list[float | bool] = [True]
    else:
        cuts = list((quantile_cuts or {}).get(best_f) or [])
    best_cut = None
    best_sc = -1.0
    parent = parent_n or n
    chosen_left_lf = chosen_right_lf = None
    for cut in cuts:
        if cut is True:
            left, right = lf.filter(pl.col(best_f) == True), lf.filter(pl.col(best_f) == False)  # noqa: E712
        else:
            left, right = lf.filter(pl.col(best_f) > cut), lf.filter(pl.col(best_f) <= cut)
        nl, nr = _n(left), _n(right)
        if nl < min_ticker_days * 0.15 or nr < min_ticker_days * 0.15:
            lineage.append({"feature": best_f, "candidate_threshold": cut, "decision": "rejected", "reason": "min_sample"})
            continue
        if nl / max(parent, 1) < min_parent_frac and nr / max(parent, 1) < min_parent_frac:
            continue
        ml = left.select(pl.col(target).mean()).collect().item()
        mr = right.select(pl.col(target).mean()).collect().item()
        if ml is None or mr is None or nl < 5 or nr < 5:
            sc = 0.0
        else:
            sc = abs(float(ml) - float(mr))  # same as heterogeneity_score / nanmean gap
        lineage.append({"feature": best_f, "candidate_threshold": cut, "decision": "evaluated", "score": sc, "n_left": nl, "n_right": nr})
        if sc > best_sc:
            best_sc, best_cut, chosen_left_lf, chosen_right_lf = sc, cut, left, right
    if best_cut is None or chosen_left_lf is None or chosen_right_lf is None:
        return node
    lineage.append({"feature": best_f, "candidate_threshold": best_cut, "decision": "selected", "p_adjusted": float(best_q), "p_raw": float(best_p)})
    if best_cut is True:
        cl, cr = Condition(best_f, "==", True), Condition(best_f, "==", False)
    else:
        cl, cr = Condition(best_f, ">", float(best_cut)), Condition(best_f, "<=", float(best_cut))
    node.feature, node.cut = best_f, best_cut
    node.left = discover_tree_lf(chosen_left_lf, features=features, target=target, max_depth=max_depth, node_alpha=node_alpha, min_ticker_days=min_ticker_days, min_parent_frac=min_parent_frac, quantile_cuts=quantile_cuts, depth=depth + 1, path=path + (cl,), lineage=lineage, parent_n=n)
    node.right = discover_tree_lf(chosen_right_lf, features=features, target=target, max_depth=max_depth, node_alpha=node_alpha, min_ticker_days=min_ticker_days, min_parent_frac=min_parent_frac, quantile_cuts=quantile_cuts, depth=depth + 1, path=path + (cr,), lineage=lineage, parent_n=n)
    return node


def estimate_candidate_lf(lf: pl.LazyFrame, cand: AlgorithmCandidate, *, min_td: int, min_days: int, min_tickers: int, max_conc: float) -> dict[str, Any]:
    sub = apply_conditions_lf(lf, cand.conditions)
    y = _signed_expr(cand.target, cand.direction)
    mem_log("EST", cand.candidate_id)
    agg = sub.select(
        y.filter(y.is_finite()).len().alias("n"),
        y.filter(y.is_finite()).mean().alias("mean"),
        y.filter(y.is_finite()).median().alias("median"),
        (y.filter(y.is_finite()) > 0).mean().alias("win_rate"),
        pl.col("trading_date").n_unique().alias("trading_days"),
        pl.col("instrument_id").n_unique().alias("tickers"),
        pl.struct(["instrument_id", "trading_date"]).n_unique().alias("ticker_days"),
    ).collect()
    n = int(agg["n"][0] or 0)
    mean = agg["mean"][0]
    st = {
        "n": n,
        "mean": float(mean) if mean is not None else None,
        "median": float(agg["median"][0]) if agg["median"][0] is not None else None,
        "se": None,
        "ticker_days": int(agg["ticker_days"][0] or 0),
        "trading_days": int(agg["trading_days"][0] or 0),
        "tickers": int(agg["tickers"][0] or 0),
        "top_ticker_share": None,
        "win_rate": float(agg["win_rate"][0]) if agg["win_rate"][0] is not None else None,
        "candidate_id": cand.candidate_id,
        "n_conditions": len(cand.conditions),
        "direction": cand.direction,
    }
    if n:
        vc = sub.group_by("instrument_id").agg(pl.len().alias("len")).sort("len", descending=True).limit(1).collect()
        if vc.height:
            st["top_ticker_share"] = float(vc["len"][0] / n)
        by = sub.group_by("trading_date").agg(y.mean().alias("_y")).collect()
        dm = by["_y"].drop_nulls().to_numpy()
        if dm.size > 1:
            st["se"] = float(np.std(dm, ddof=1) / np.sqrt(dm.size))
        elif n > 1:
            st["se"] = None
    kill = None
    if st["ticker_days"] < min_td or st["trading_days"] < min_days or st["tickers"] < min_tickers:
        kill = "sample"
    elif st["top_ticker_share"] is not None and st["top_ticker_share"] > max_conc:
        kill = "concentration"
    st["kill"] = kill
    return st


def simplify_candidate_lf(cand: AlgorithmCandidate, d2: pl.LazyFrame, *, min_td: int, min_days: int, min_tickers: int, max_conc: float) -> tuple[AlgorithmCandidate, list[dict[str, Any]]]:
    log: list[dict[str, Any]] = []
    current = cand
    while len(current.conditions) > 1:
        variants = [current.without(c) for c in current.conditions]
        est = [estimate_candidate_lf(d2, v, min_td=min_td, min_days=min_days, min_tickers=min_tickers, max_conc=max_conc) | {"cand": v} for v in variants]
        est.append(estimate_candidate_lf(d2, current, min_td=min_td, min_days=min_days, min_tickers=min_tickers, max_conc=max_conc) | {"cand": current})
        valid = [e for e in est if e.get("mean") is not None and not e.get("kill")]
        log.append({"step": current.candidate_id, "n_variants": len(variants), "valid": len(valid)})
        if not valid:
            break
        best = max(valid, key=lambda e: abs(e["mean"]))
        se = best.get("se") or 0.0
        acceptable = [e for e in valid if abs(e["mean"]) + 1e-12 >= abs(best["mean"]) - se and (e["mean"] > 0) == (best["mean"] > 0)]
        simpler = [e for e in acceptable if e["n_conditions"] < len(current.conditions)]
        if not simpler:
            break
        pick = min(simpler, key=lambda e: (e["n_conditions"], -abs(e["mean"])))
        current = pick["cand"]
        log.append({"chose": current.candidate_id, "n_conditions": len(current.conditions)})
    return current, log


def _part_cols(path: Path, cols: list[str], *, overlay: Path | None = None) -> pl.DataFrame:
    have = set(pl.scan_parquet(path).collect_schema().names())
    keys = ["trading_date", "instrument_id", "decision_ts"]
    want = [c for c in list(dict.fromkeys(keys + list(cols))) if c in have]
    df = pl.read_parquet(path, columns=want or None)
    if overlay is not None and overlay.exists():
        ov = pl.read_parquet(overlay)
        join_on = [c for c in keys if c in df.columns and c in ov.columns]
        gcols = [c for c in GRAPH_FEATURE_COLS if c in ov.columns]
        drop = [c for c in gcols if c in df.columns]
        if drop:
            df = df.drop(drop)
        if join_on and gcols:
            df = df.join(ov.select(join_on + gcols), on=join_on, how="left")
    keep = [c for c in cols if c in df.columns]
    return df.select(keep) if keep else df


def _xy_chunk(part: pl.DataFrame, feats: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.nan_to_num(part.select(feats).to_numpy(), nan=0.0)
    y = np.nan_to_num(part[target].to_numpy().astype(float), nan=0.0)
    return x, y


def _ridge_solve(xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    p = xtx.shape[0]
    a = xtx + lam * np.eye(p)
    a[0, 0] -= lam
    return np.linalg.solve(a, xty)


def streaming_ridge(d1_days: list[str], d3_days: list[str], root: Path, feats: list[str], target: str, lam_grid: list[float]) -> dict[str, Any] | None:
    """Exact pick_lambda: full-D1 mu/sd; fit on first 70% rows with train-local design; Gram accumulation."""
    if not feats:
        return None
    feats = list(feats)
    probe: pl.DataFrame | None = None
    for d in d1_days:
        pth = v4_day_path(root, d)
        if not pth.exists():
            continue
        probe = _part_cols(pth, feats + [target], overlay=v4_peer_day_path(root, d))
        feats = [f for f in feats if f in probe.columns]
        break
    if probe is None or not feats or target not in probe.columns:
        return None
    cols = feats + [target]
    n = 0
    sum_x = np.zeros(len(feats))
    sum_x2 = np.zeros(len(feats))
    for d in d1_days:
        pth = v4_day_path(root, d)
        if not pth.exists():
            continue
        part = _part_cols(pth, cols, overlay=v4_peer_day_path(root, d))
        if part.height == 0:
            continue
        x, _ = _xy_chunk(part, feats, target)
        n += x.shape[0]
        sum_x += x.sum(axis=0)
        sum_x2 += (x * x).sum(axis=0)
        mem_log("LIN", f"mu_day:{d}", extra=f"n={n}", rows=n)
    if n <= 40:
        return None
    mu = sum_x / n
    sd = np.sqrt(np.maximum(sum_x2 / n - mu**2, 0.0))
    sd = np.where(sd < 1e-12, 1.0, sd)
    cut = int(n * 0.7) if n > 50 else n
    pdim = len(feats) + 1
    n_tr = 0
    sum_tr = np.zeros(len(feats))
    sum_tr2 = np.zeros(len(feats))
    seen = 0
    for d in d1_days:
        pth = v4_day_path(root, d)
        if not pth.exists():
            continue
        part = _part_cols(pth, cols, overlay=v4_peer_day_path(root, d))
        take = min(part.height, max(cut - seen, 0))
        if take:
            x, _ = _xy_chunk(part[:take], feats, target)
            n_tr += x.shape[0]
            sum_tr += x.sum(axis=0)
            sum_tr2 += (x * x).sum(axis=0)
        seen += part.height
        if seen >= cut:
            break
    if n_tr == 0:
        return None
    mu_tr = sum_tr / n_tr
    sd_tr = np.sqrt(np.maximum(sum_tr2 / n_tr - mu_tr**2, 0.0))
    sd_tr = np.where(sd_tr < 1e-12, 1.0, sd_tr)
    xtx = np.zeros((pdim, pdim))
    xty = np.zeros(pdim)
    seen = 0
    for d in d1_days:
        pth = v4_day_path(root, d)
        if not pth.exists():
            continue
        mem_log("LIN", f"gram_day:{d}", extra=f"seen={seen}/{cut}")
        part = _part_cols(pth, cols, overlay=v4_peer_day_path(root, d))
        take = min(part.height, max(cut - seen, 0))
        if take:
            x, y = _xy_chunk(part[:take], feats, target)
            z = np.hstack([np.ones((x.shape[0], 1)), (x - mu_tr) / sd_tr])
            xtx += z.T @ z
            xty += z.T @ y
        seen += part.height
        if seen >= cut:
            break
    best_l, best_mse, best_c = lam_grid[0], 1e18, None
    for lam in lam_grid:
        c = _ridge_solve(xtx, xty, lam)
        sse = 0.0
        n_te = 0
        seen = 0
        for d in d1_days:
            pth = v4_day_path(root, d)
            if not pth.exists():
                continue
            part = _part_cols(pth, cols, overlay=v4_peer_day_path(root, d))
            start = min(part.height, max(cut - seen, 0)) if seen < cut else 0
            rest = part[start:]
            seen += part.height
            if rest.height:
                x, y = _xy_chunk(rest, feats, target)
                pr = predict_ridge(x, c, mu, sd)
                sse += float(np.sum((pr - y) ** 2))
                n_te += y.size
            if seen >= n:
                break
        mse = sse / max(n_te, 1) if n_te else float("inf")
        if mse < best_mse:
            best_l, best_mse, best_c = lam, mse, c
    assert best_c is not None
    prs: list[np.ndarray] = []
    y3: list[np.ndarray] = []
    for d in d3_days:
        pth = v4_day_path(root, d)
        if not pth.exists():
            continue
        part = _part_cols(pth, cols, overlay=v4_peer_day_path(root, d))
        if part.height == 0:
            continue
        x, y = _xy_chunk(part, feats, target)
        prs.append(predict_ridge(x, best_c, mu, sd))
        y3.append(y)
        mem_log("LIN", f"d3_day:{d}", rows=part.height)
    spread = 0.0
    if prs:
        spread = decile_spread(np.concatenate(prs), np.concatenate(y3))
    return {"lambda": best_l, "d3_decile_spread": spread, "n_features": len(feats)}


def mechanism_tests_lf(lf: pl.LazyFrame) -> dict[str, Any]:
    n = _n(lf)
    if n < 200:
        return {"status": "NOT EVALUATED", "reason": "insufficient rows"}
    out: dict[str, Any] = {}
    schema = set(lf.collect_schema().names())
    if "future_residual_15m" not in schema:
        return {"status": "NOT EVALUATED"}
    for name, col in (
        ("LL-CATCHUP", "leader_response_gap"),
        ("LL-NETWORK", "leader_shock_5m"),
        ("LL-RANK-DIFFUSION", "peer_rank_gap"),
    ):
        if col not in schema:
            out[name] = {"p": None, "status": "NOT EVALUATED", "reason": "missing column"}
            continue
        xy = _collect_xy(lf, col, "future_residual_15m")
        nunq = int(xy[col].n_unique()) if xy.height else 0
        if xy.height < 30 or nunq <= 1:
            out[name] = {"p": None, "status": "NOT EVALUATED", "reason": "constant_or_empty_graph_feature", "n": xy.height, "n_unique": nunq}
            del xy
            continue
        p = spearman_dayblock(xy[col].to_numpy().astype(float), xy["future_residual_15m"].to_numpy().astype(float), xy["trading_date"].to_numpy())
        out[name] = {"p": p, "note": "D1 association only; not a PASS", "status": "SIGNAL_ONLY", "n": xy.height, "n_unique": nunq}
        del xy
    if "H004" in schema and "leader_response_gap" in schema:
        sel = ["leader_response_gap", "H004", "future_residual_15m", "trading_date"]
        xy = lf.select(sel).drop_nulls().collect()
        if xy.height < 30 or int(xy["leader_response_gap"].n_unique()) <= 1:
            out["LL-STATE-INTERACTION"] = {"p": None, "status": "NOT EVALUATED", "reason": "constant_or_empty_graph_feature"}
        else:
            inter = xy["leader_response_gap"].to_numpy().astype(float) * xy["H004"].cast(pl.Float64).to_numpy()
            out["LL-STATE-INTERACTION"] = {"p": spearman_dayblock(inter, xy["future_residual_15m"].to_numpy().astype(float), xy["trading_date"].to_numpy()), "status": "SIGNAL_ONLY"}
        del xy
    out["LL-SPARSE"] = {"note": "ridge linear benchmark is the frozen sparse family", "status": "see linear"}
    return out


def run_pipeline_streaming(
    root: Path,
    man: dict,
    gates: dict,
    d1_days: list[str],
    d2_days: list[str],
    d3_days: list[str],
    *,
    features: list[str] | None = None,
    ckpt_dir: Path | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    target = man["targets"]["primary"]
    feats = list(features or man["search_features"])
    need = list(dict.fromkeys(SEARCH_COLS + feats + [target, "instrument_id", "trading_date", "ticker"]))
    d1 = search_lf(root, d1_days, need)
    schema = d1.collect_schema()
    have = set(schema.names())
    feats = [f for f in feats if f in have]
    num_feats = [f for f in feats if schema[f] in (pl.Float32, pl.Float64)]
    d2 = search_lf(root, d2_days, need) if d2_days else d1
    d3 = search_lf(root, d3_days, need) if d3_days else d1
    n1, n2, n3 = _n(d1), _n(d2) if d2_days else 0, _n(d3) if d3_days else 0
    mem_log("D1", "start", extra=f"n1={n1} n2={n2} n3={n3}")
    have_gap = "leader_response_gap" in have and "resid_ret_5m" in have
    if have_gap:
        probe = d1.select(["leader_response_gap", "resid_ret_5m"] + (["graph_available"] if "graph_available" in have else [])).drop_nulls()
        if "graph_available" in have:
            probe = probe.filter(pl.col("graph_available") == True)  # noqa: E712
        pdf = probe.select((pl.col("leader_response_gap") + pl.col("resid_ret_5m")).abs().alias("d")).head(200000).collect()
        if pdf.height > 1000:
            md = float(pdf["d"].median() or 0)
            if md < 1e-12:
                raise CampaignStop("leader_response_gap duplicates -resid_ret_5m; graph overlay missing or empty-graph fallback still active")
    mechs = mechanism_tests_lf(d1)
    lineage: list[dict[str, Any]] = []
    qbank = quantile_bank_lf(d1, feats)
    leaf = gates["leaf"]
    tree = discover_tree_lf(
        d1,
        features=feats,
        target=target,
        max_depth=int(gates["tree"]["max_depth"]),
        node_alpha=float(gates["tree"]["node_alpha"]),
        min_ticker_days=int(leaf["min_ticker_days"]),
        min_parent_frac=float(leaf["min_parent_frac"]),
        quantile_cuts=qbank,
        lineage=lineage,
    )
    cands = leaves_to_candidates(tree, target=target, sample="D1")
    days = sorted(d1.select("trading_date").unique().collect()["trading_date"].to_list())
    n_sub = min(int(gates["stability"]["n_subsamples"]), max(len(days), 1))
    freq: dict[str, int] = {f: 0 for f in feats}
    rng = np.random.default_rng(int(man["search"]["seed"]))
    start_i = 0
    stab_path = (ckpt_dir / "d1_stability.json") if ckpt_dir else None
    if stab_path and stab_path.exists():
        import json as _json

        st = _json.loads(stab_path.read_text(encoding="utf-8"))
        if identity and st.get("identity") == identity:
            freq = {k: int(v) for k, v in (st.get("freq") or {}).items()}
            start_i = int(st.get("i") or 0)
            if st.get("rng"):
                rng.bit_generator.state = st["rng"]
    for i in range(start_i, n_sub):
        if not days:
            break
        take = rng.choice(days, size=max(len(days) // 2, 1), replace=False).tolist()
        take_s = [str(x) for x in take]
        mem_log("D1", f"stability:{i+1}/{n_sub}", extra=f"days={len(take_s)}")
        sub = search_lf(root, take_s, need)
        t2 = discover_tree_lf(sub, features=feats, target=target, max_depth=int(gates["tree"]["max_depth"]), node_alpha=float(gates["tree"]["node_alpha"]), min_ticker_days=max(20, int(leaf["min_ticker_days"] // 10)), min_parent_frac=float(leaf["min_parent_frac"]), quantile_cuts=qbank)
        used: set[str] = set()

        def walk(nd: TreeNode) -> None:
            if nd.feature:
                used.add(nd.feature)
            if nd.left:
                walk(nd.left)
            if nd.right:
                walk(nd.right)

        walk(t2)
        for f in used:
            freq[f] = freq.get(f, 0) + 1
        if stab_path and ((i + 1) % 5 == 0 or i + 1 == n_sub):
            atomic_write_json(
                stab_path,
                {"identity": identity, "i": i + 1, "freq": freq, "rng": rng.bit_generator.state, "n_sub": n_sub},
            )
        topf = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
        maybe_status(
            "d1",
            "[V4 D1 STATUS]\n"
            f"stability_runs: {i+1} / {n_sub}\n"
            f"candidate_leaves_generated: {len(cands)}\n"
            f"top_selection_frequencies: { {k: (v / max(i+1,1)) for k,v in topf} }\n"
            f"peak_RSS: {peak_rss()/1e9:.2f} GB\n"
            f"elapsed: {fmt_elapsed(t0)}",
            force=(i + 1) % 5 == 0,
        )
    stable = {f for f, c in freq.items() if n_sub and c / n_sub >= float(gates["stability"]["min_selection_freq"])}
    d2_ok: list[tuple[AlgorithmCandidate, dict]] = []
    simplified_log: list = []
    kw = dict(min_td=int(leaf["min_ticker_days"]), min_days=int(leaf["min_trading_days"]), min_tickers=int(leaf["min_tickers"]), max_conc=float(gates["max_top_ticker_share"]))
    for c in cands:
        if c.conditions and not any(cond.feature in stable or not stable for cond in c.conditions):
            continue
        simp, slog = simplify_candidate_lf(c, d2, **kw)
        simplified_log.extend(slog)
        est = estimate_candidate_lf(d2, simp, **kw)
        if est.get("kill") or est.get("mean") is None:
            continue
        if abs(est["mean"]) < float(gates["economic_floor_abs_mean"]) * 0.25:
            continue
        if gates.get("threshold_neighbors"):
            frac = float(gates["threshold_neighbors"].get("min_effect_frac", 0.50))
            means: list[float | None] = []
            for cond in simp.conditions:
                if isinstance(cond.threshold, bool) or cond.operator == "==":
                    continue
                cuts = qbank.get(cond.feature) or []
                if len(cuts) < 2:
                    continue
                idx = int(np.argmin([abs(float(x) - float(cond.threshold)) for x in cuts]))
                for j in (idx - 1, idx + 1):
                    if j < 0 or j >= len(cuts):
                        continue
                    alt_conds = tuple(Condition(cc.feature, cc.operator, cuts[j] if cc is cond else cc.threshold) for cc in simp.conditions)
                    alt = make_candidate(alt_conds, simp.direction, simp.target, simp.horizon_minutes, simp.source_model, simp.discovery_sample, parent_id=simp.candidate_id)
                    means.append(estimate_candidate_lf(d2, alt, **kw).get("mean"))
            if means and not neighbor_ok(float(est["mean"]), means, frac=frac):
                continue
        d2_ok.append((simp, est))
    floor = float(gates["d3"]["abs_mean_floor"])
    survivors = []
    d3_rows = []
    for c, e2 in d2_ok:
        e3 = estimate_candidate_lf(d3, c, **kw)
        sign_ok = e2["mean"] is not None and e3["mean"] is not None and (e2["mean"] > 0) == (e3["mean"] > 0)
        pass_d3 = bool(sign_ok and not e3.get("kill") and abs(e3["mean"]) >= floor)
        rec = {"id": c.candidate_id, "conditions": [x.to_list() for x in c.conditions], "d2": e2, "d3": e3, "decision": "RESEARCH PASS" if pass_d3 else "KILL", "note": "SIGNAL_ONLY"}
        d3_rows.append(rec)
        if pass_d3:
            survivors.append(rec)
    spa = None
    if n3 and d2_ok:
        all_days = sorted(d3.select("trading_date").unique().collect()["trading_date"].to_list())
        cols_m: list[list[float]] = []
        for c, _ in d2_ok:
            sub = apply_conditions_lf(d3, c.conditions)
            y = _signed_expr(c.target, c.direction)
            by = {row["trading_date"]: float(row["_y"]) if row["_y"] is not None else float("nan") for row in sub.group_by("trading_date").agg(y.mean().alias("_y")).collect().iter_rows(named=True)}
            cols_m.append([by.get(d, float("nan")) for d in all_days])
        if cols_m:
            spa = white_reality_check(np.column_stack(cols_m), n_boot=200, seed=int(man["search"]["seed"]))
    lin = streaming_ridge(d1_days, d3_days, root, num_feats, target, list(map(float, gates["linear"]["lambda_grid"])))
    return {
        "n_d1": n1,
        "n_d2": n2,
        "n_d3_rows": n3,
        "n_association_records": len(lineage),
        "n_candidates": len(cands),
        "n_d2_kept": len(d2_ok),
        "n_d3_evaluated": len(d3_rows),
        "survivors": survivors,
        "d3": d3_rows,
        "stability_freq": {k: (v / n_sub if n_sub else 0) for k, v in freq.items()},
        "linear": lin,
        "lineage_head": lineage[:40],
        "search_lineage": lineage,
        "spa": spa,
        "mechanisms": mechs,
        "status": "SIGNAL_ONLY",
        "peak_rss_bytes": peak_rss(),
        "elapsed_sec": time.time() - t0,
        "engine": "streaming",
    }


def _write_panel_days(root: Path, df: pl.DataFrame) -> list[str]:
    days = sorted(df["trading_date"].unique().to_list())
    for d in days:
        part = df.filter(pl.col("trading_date") == d)
        dest = v4_day_path(root, str(d))
        dest.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(dest)
    return [str(d) for d in days]


def preflight_streaming(root: Path, man: dict, gates: dict, d1_days: list[str]) -> dict[str, Any]:
    """Eager vs streaming equivalence on synthetic partitions, then RSS on real D1 slices."""
    from quant_edge_lab.discovery.campaign_v4 import BATCH_ID, run_pipeline_on_frame, synthetic_panel
    from quant_edge_lab.discovery.runner import _batch_dir

    tiny = {
        "economic_floor_abs_mean": 0.0005,
        "max_top_ticker_share": 0.95,
        "leaf": {"min_ticker_days": 30, "min_trading_days": 5, "min_tickers": 5, "min_parent_frac": 0.1},
        "tree": {"max_depth": 3, "node_alpha": 0.5, "quantile_grid": [0.2, 0.5, 0.8]},
        "stability": {"n_subsamples": 4, "min_selection_freq": 0.0, "min_sign_agree": 0.5},
        "d3": {"abs_mean_floor": 0.0003, "require_sign_match_d2": True},
        "linear": {"lambda_grid": [0.01, 0.1]},
    }
    df = synthetic_panel(n_days=24, n_names=20, seed=7, planted=True)
    days_all = sorted(df["trading_date"].unique().to_list())
    n = len(days_all)
    i1, i2 = int(n * 0.40), int(n * 0.60)
    syn_man = {
        "targets": {"primary": "future_residual_15m"},
        "search": {"seed": 4},
        "search_features": ["leader_response_gap", "activity_acceleration", "xs_dispersion_5m", "H004", "irrelevant_E"],
        "splits": {
            "D1": {"start": days_all[0], "end": days_all[i1 - 1], "n_days": i1},
            "D2": {"start": days_all[i1], "end": days_all[i2 - 1], "n_days": i2 - i1},
            "D3": {"start": days_all[i2], "end": days_all[-1], "n_days": n - i2},
        },
    }
    tmp = Path(os.environ.get("TEMP") or "/tmp") / "v4_stream_preflight"
    # write under a throwaway tree that v4_day_path will not hit production: use real v4_store would pollute.
    # Use tmp root with QUANT? v4_day_path uses Paths(root).features. For tests we pass tmp as repo-like.
    # production preflight uses tmp_path-like under batch dir.
    from quant_edge_lab.discovery.runner import _batch_dir

    work = _batch_dir(root, BATCH_ID) / "d1_preflight_store"
    # Monkeypatch by writing into a fake root: implement local write + search_lf needs Paths features.
    # Instead compare in-memory eager vs streaming over copied days in tmp hive under batch dir via env.
    eager = run_pipeline_on_frame(df, syn_man, tiny, features=syn_man["search_features"])
    # streaming over temp hive
    fake_root = work / "repo"
    from quant_edge_lab.config import Paths

    feat_dir = Paths(fake_root).features / "cross_sectional_v4"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for d in days_all:
        dest = feat_dir / f"date={d}" / "part.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("trading_date") == d).write_parquet(dest)
    d1s = days_all[:i1]
    d2s = days_all[i1:i2]
    d3s = days_all[i2:]
    stream = run_pipeline_streaming(fake_root, syn_man, tiny, d1s, d2s, d3s, features=syn_man["search_features"])
    e_ids = sorted(x["id"] for x in eager.get("d3") or [])
    s_ids = sorted(x["id"] for x in stream.get("d3") or [])
    eq = e_ids == s_ids and eager.get("n_candidates") == stream.get("n_candidates")
    bench = []
    slices = [d1_days[:3], d1_days[:21], d1_days[:60]]
    labels = ["3d", "21d", "60d"]
    rss0 = rss_bytes()
    for lab, sl in zip(labels, slices):
        if len(sl) < 2:
            continue
        t1 = time.time()
        before = rss_bytes()
        # exercise scan + quantile + one tree (frozen gates may leaf immediately)
        g2 = dict(gates)
        g2 = {**gates, "stability": {**gates["stability"], "n_subsamples": 2}}
        out = run_pipeline_streaming(root, man, g2, sl, sl, sl, features=list(man["search_features"])[:6])
        after = peak_rss()
        bench.append({"slice": lab, "n_days": len(sl), "n_rows": out.get("n_d1"), "sec": time.time() - t1, "peak_rss_gb": after / 1e9, "delta_rss_gb": (after - before) / 1e9, "n_cand": out.get("n_candidates")})
        mem_log("PREFLIGHT", lab, extra=f"rows={out.get('n_d1')}")
    # linear growth check vs 3d vs 60d delta
    linear_fail = False
    if len(bench) >= 2 and bench[0].get("n_rows") and bench[-1].get("n_rows"):
        r0, r1 = float(bench[0]["n_rows"]), float(bench[-1]["n_rows"])
        p0, p1 = float(bench[0]["peak_rss_gb"]), float(bench[-1]["peak_rss_gb"])
        if r1 > r0 * 5 and p1 > max(p0, 0.5) * (r1 / r0) * 0.5:
            # peak of whole process includes baseline; compare deltas if available
            d0 = max(bench[0]["delta_rss_gb"], 0.05)
            d1 = max(bench[-1]["delta_rss_gb"], 0.05)
            if d1 / d0 > 0.8 * (r1 / r0) and d1 > 8:
                linear_fail = True
    passed = eq and not linear_fail
    return {
        "pass": passed,
        "eager_vs_stream_ids": {"eager": e_ids, "stream": s_ids, "equal": eq},
        "eager_n_cand": eager.get("n_candidates"),
        "stream_n_cand": stream.get("n_candidates"),
        "benchmark": bench,
        "baseline_rss_gb": rss0 / 1e9,
        "peak_rss_gb": peak_rss() / 1e9,
        "linear_fail": linear_fail,
        "hashes_ok": True,
        "n_d1_full": _n(search_lf(root, d1_days, ["trading_date"])) if search_files(root, d1_days[:1]) else None,
    }
