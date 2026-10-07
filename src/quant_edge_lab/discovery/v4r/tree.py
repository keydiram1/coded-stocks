"""V4R CIT using day-aware tests. Fold trees must pass fold-local quantile cuts."""

from __future__ import annotations

from typing import Any

import polars as pl

from quant_edge_lab.discovery.conditional import Condition
from quant_edge_lab.discovery.conditional.association import heterogeneity_score
from quant_edge_lab.discovery.conditional.partition import QUANTILES, TreeNode, frozen_quantile_cuts, leaves_to_candidates
from quant_edge_lab.discovery.v4r.association import is_binary_series, node_association


def quantile_bank(df: pl.DataFrame, features: list[str], grid: list[float] | None = None) -> dict[str, list[float]]:
    g = grid or QUANTILES
    out: dict[str, list[float]] = {}
    for f in features:
        if f not in df.columns:
            continue
        if df[f].dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            continue
        s = df[f].drop_nulls()
        if s.n_unique() > 4:
            out[f] = frozen_quantile_cuts(s, g)
    return out


def discover_tree(
    df: pl.DataFrame,
    *,
    features: list[str],
    target: str,
    max_depth: int,
    node_alpha: float,
    min_ticker_days: int,
    min_parent_frac: float,
    quantile_cuts: dict[str, list[float]] | None,
    depth: int = 0,
    path: tuple[Condition, ...] = (),
    lineage: list[dict[str, Any]] | None = None,
    parent_n: int | None = None,
) -> TreeNode:
    lineage = lineage if lineage is not None else []
    n = int(df.height)
    mean = float(df[target].mean()) if target in df.columns and df.height else None
    node = TreeNode(None, None, n, mean, path=path, lineage=lineage)
    td = int(df.select(pl.struct(["instrument_id", "trading_date"]).n_unique()).item()) if {"instrument_id", "trading_date"} <= set(df.columns) else n
    n_days = int(df["trading_date"].n_unique()) if "trading_date" in df.columns else 0
    n_tickers = int(df["instrument_id"].n_unique()) if "instrument_id" in df.columns else 0
    if depth >= max_depth or td < min_ticker_days:
        lineage.append({"decision": "rejected", "reason": "max_depth_or_sample", "depth": depth, "n": n, "n_ticker_days": td, "n_days": n_days, "n_tickers": n_tickers})
        return node
    tests, recs, q = node_association(df, features, target)
    lineage.extend(recs)
    if not tests:
        return node
    ranked = sorted(zip([f for f, _ in tests], [p for _, p in tests], q), key=lambda t: t[2])
    best_f, best_p, best_q = ranked[0]
    if float(best_q) > node_alpha:
        lineage.append({"feature": best_f, "decision": "rejected", "reason": "q>alpha", "p_adjusted": float(best_q)})
        return node
    if is_binary_series(df[best_f]):
        cuts: list[float | bool] = [True]
    else:
        cuts = list((quantile_cuts or {}).get(best_f) or [])
    best_cut = None
    best_sc = -1.0
    best_left = best_right = None
    parent = parent_n or n
    for cut in cuts:
        if cut is True:
            left, right = df.filter(pl.col(best_f) == True), df.filter(pl.col(best_f) == False)  # noqa: E712
        else:
            left, right = df.filter(pl.col(best_f) > cut), df.filter(pl.col(best_f) <= cut)
        if left.height < min_ticker_days * 0.15 or right.height < min_ticker_days * 0.15:
            lineage.append({"feature": best_f, "candidate_threshold": cut, "decision": "rejected", "reason": "min_sample"})
            continue
        if left.height / max(parent, 1) < min_parent_frac and right.height / max(parent, 1) < min_parent_frac:
            continue
        sc = heterogeneity_score(left[target].to_numpy().astype(float), right[target].to_numpy().astype(float))
        lineage.append({"feature": best_f, "candidate_threshold": cut, "decision": "evaluated", "score": sc, "n_left": left.height, "n_right": right.height})
        if sc > best_sc:
            best_sc, best_cut, best_left, best_right = sc, cut, left, right
    if best_cut is None or best_left is None:
        return node
    lineage.append({"feature": best_f, "candidate_threshold": best_cut, "decision": "selected", "p_adjusted": float(best_q), "p_raw": float(best_p)})
    if best_cut is True:
        cl, cr = Condition(best_f, "==", True), Condition(best_f, "==", False)
    else:
        cl, cr = Condition(best_f, ">", float(best_cut)), Condition(best_f, "<=", float(best_cut))
    node.feature, node.cut = best_f, best_cut
    node.left = discover_tree(
        best_left, features=features, target=target, max_depth=max_depth, node_alpha=node_alpha,
        min_ticker_days=min_ticker_days, min_parent_frac=min_parent_frac, quantile_cuts=quantile_cuts,
        depth=depth + 1, path=path + (cl,), lineage=lineage, parent_n=n,
    )
    node.right = discover_tree(
        best_right, features=features, target=target, max_depth=max_depth, node_alpha=node_alpha,
        min_ticker_days=min_ticker_days, min_parent_frac=min_parent_frac, quantile_cuts=quantile_cuts,
        depth=depth + 1, path=path + (cr,), lineage=lineage, parent_n=n,
    )
    return node


def tree_to_dict(node: TreeNode) -> dict[str, Any]:
    return {
        "feature": node.feature,
        "cut": node.cut,
        "n": node.n,
        "mean": node.mean,
        "path": [c.to_list() for c in node.path],
        "left": tree_to_dict(node.left) if node.left else None,
        "right": tree_to_dict(node.right) if node.right else None,
    }
