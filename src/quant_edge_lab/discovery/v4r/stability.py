"""Canonical rule paths and stability matching. Threshold similarity is fold-local percentile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.conditional import AlgorithmCandidate, Condition


def op_key(op: str) -> str:
    if op in {">", ">="}:
        return "gt"
    if op in {"<", "<="}:
        return "le"
    if op == "==":
        return "eq"
    return op


@dataclass(frozen=True)
class PathKey:
    steps: tuple[tuple[str, str], ...]  # (feature, op_key)
    direction: str


def path_key_from_conditions(conds: tuple[Condition, ...], direction: str) -> PathKey:
    return PathKey(tuple((c.feature, op_key(c.operator)) for c in conds), direction)


def path_key_no_dir(conds: tuple[Condition, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((c.feature, op_key(c.operator)) for c in conds)


def percentile_of(values: np.ndarray, t: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")
    return float((v <= t).mean())


def thresholds_similar(
    cand_thresh: list[float | bool],
    fold_thresh: list[float | bool],
    fold_df: pl.DataFrame,
    features: list[str],
    q_tol: float,
) -> bool:
    if len(cand_thresh) != len(fold_thresh):
        return False
    for t0, t1, feat in zip(cand_thresh, fold_thresh, features):
        if isinstance(t0, bool) or isinstance(t1, bool):
            if t0 != t1:
                return False
            continue
        if feat not in fold_df.columns:
            return False
        arr = fold_df[feat].drop_nulls().to_numpy().astype(float)
        q0 = percentile_of(arr, float(t0))
        q1 = percentile_of(arr, float(t1))
        if not np.isfinite(q0) or not np.isfinite(q1):
            return False
        if abs(q0 - q1) > q_tol:
            return False
    return True


def rule_text(c: AlgorithmCandidate) -> str:
    parts = [f"{x.feature} {x.operator} {x.threshold}" for x in c.conditions]
    return " AND ".join(parts) + f" → {c.direction}"


def max_possible(hits: int, done: int, total: int) -> int:
    return hits + (total - done)


def stability_impossible(hits: int, done: int, total: int, need_frac: float) -> bool:
    need = int(np.ceil(need_frac * total - 1e-12))
    return max_possible(hits, done, total) < need


def direction_impossible(agrees: int, path_hits: int, remaining: int, need_frac: float) -> bool:
    """Worst-case: all remaining folds are path hits that agree."""
    if remaining < 0:
        remaining = 0
    max_a = agrees + remaining
    max_p = path_hits + remaining
    if max_p == 0:
        return True
    return (max_a / max_p) < need_frac - 1e-12
