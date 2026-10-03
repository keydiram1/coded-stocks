"""Conditional-inference tree on D1. Lineage records every association/split decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.conditional import AlgorithmCandidate, Condition, make_candidate
from quant_edge_lab.discovery.conditional.association import bh_qvalues, binary_ttest, heterogeneity_score, spearman_dayblock

QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


@dataclass
class TreeNode:
    feature: str | None
    cut: float | bool | None
    n: int
    mean: float | None
    left: TreeNode | None = None
    right: TreeNode | None = None
    path: tuple[Condition, ...] = ()
    lineage: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _is_binary(s: pl.Series) -> bool:
    u = s.drop_nulls().unique()
    if u.len() <= 2:
        vals = set(u.to_list())
        return vals <= {True, False, 0, 1, 0.0, 1.0}
    return False


def frozen_quantile_cuts(s: pl.Series, grid: list[float]) -> list[float]:
    vals = s.drop_nulls().to_numpy()
    if vals.size < 50:
        return []
    return [float(np.nanquantile(vals, q)) for q in grid]


def discover_tree(
    df: pl.DataFrame,
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
    y = df[target].to_numpy().astype(float) if target in df.columns else np.array([])
    days = df["trading_date"].to_numpy() if "trading_date" in df.columns else np.arange(df.height)
    n = int(df.height)
    mean = float(np.nanmean(y)) if y.size else None
    node = TreeNode(None, None, n, mean, path=path, lineage=lineage)
    if depth >= max_depth or n < min_ticker_days:
        lineage.append({"decision": "rejected", "reason": "max_depth_or_sample", "depth": depth, "n": n})
        return node
    tests: list[tuple[str, float]] = []
    for feat in features:
        if feat not in df.columns:
            continue
        x = df[feat].to_numpy()
        if _is_binary(df[feat]):
            p = binary_ttest(x.astype(float), y)
        else:
            p = spearman_dayblock(x.astype(float), y, days)
        tests.append((feat, p))
        lineage.append({"search_id": f"d{depth}", "feature": feat, "sample": "D1", "decision": "considered", "p_raw": p, "n_ticker_days": n})
    if not tests:
        return node
    q = bh_qvalues(np.array([p for _, p in tests]))
    ranked = sorted(zip([f for f, _ in tests], [p for _, p in tests], q), key=lambda t: t[2])
    best_f, best_p, best_q = ranked[0]
    if best_q > node_alpha:
        lineage.append({"feature": best_f, "decision": "rejected", "reason": "q>alpha", "p_adjusted": float(best_q)})
        return node
    cuts: list[float | bool]
    if _is_binary(df[best_f]):
        cuts = [True]
    else:
        cuts = list((quantile_cuts or {}).get(best_f) or frozen_quantile_cuts(df[best_f], QUANTILES))
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
    node.left = discover_tree(best_left, features=features, target=target, max_depth=max_depth, node_alpha=node_alpha, min_ticker_days=min_ticker_days, min_parent_frac=min_parent_frac, quantile_cuts=quantile_cuts, depth=depth + 1, path=path + (cl,), lineage=lineage, parent_n=n)
    node.right = discover_tree(best_right, features=features, target=target, max_depth=max_depth, node_alpha=node_alpha, min_ticker_days=min_ticker_days, min_parent_frac=min_parent_frac, quantile_cuts=quantile_cuts, depth=depth + 1, path=path + (cr,), lineage=lineage, parent_n=n)
    return node


def leaves_to_candidates(node: TreeNode, *, target: str, sample: str) -> list[AlgorithmCandidate]:
    out: list[AlgorithmCandidate] = []

    def walk(n: TreeNode) -> None:
        if n.is_leaf and n.path:
            direction = "LONG" if (n.mean or 0) >= 0 else "SHORT"
            out.append(make_candidate(n.path, direction, target, 15, "cit_v4", sample, lineage=("cit",)))
        if n.left:
            walk(n.left)
        if n.right:
            walk(n.right)

    walk(node)
    return out
