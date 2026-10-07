"""Immutable V4 algorithm candidates. Identity is a hash of the canonical spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from quant_edge_lab.hashing import sha256_json

Op = Literal["<", "<=", ">", ">=", "=="]


@dataclass(frozen=True)
class Condition:
    feature: str
    operator: Op
    threshold: float | bool

    def to_list(self) -> list:
        return [self.feature, self.operator, self.threshold]


@dataclass(frozen=True)
class AlgorithmCandidate:
    candidate_id: str
    conditions: tuple[Condition, ...]
    direction: Literal["LONG", "SHORT"]
    target: str
    horizon_minutes: int
    source_model: str
    discovery_sample: str
    parent_id: str | None
    lineage: tuple[str, ...] = ()

    def without(self, cond: Condition) -> "AlgorithmCandidate":
        rest = tuple(c for c in self.conditions if c != cond)
        return make_candidate(rest, self.direction, self.target, self.horizon_minutes, self.source_model, self.discovery_sample, parent_id=self.candidate_id, lineage=self.lineage + ("drop:" + cond.feature,))


def canonical_spec(conditions: tuple[Condition, ...] | list[Condition], direction: str, target: str, horizon_minutes: int = 15) -> dict[str, Any]:
    conds = [c.to_list() if isinstance(c, Condition) else list(c) for c in conditions]
    conds = sorted(conds, key=lambda x: (str(x[0]), str(x[1]), str(x[2])))
    return {"conditions": conds, "direction": direction, "target": target, "horizon_minutes": horizon_minutes}


def make_candidate(
    conditions: tuple[Condition, ...] | list[Condition],
    direction: str,
    target: str,
    horizon_minutes: int,
    source_model: str,
    discovery_sample: str,
    parent_id: str | None = None,
    lineage: tuple[str, ...] = (),
) -> AlgorithmCandidate:
    cond_t = tuple(conditions)
    spec = canonical_spec(cond_t, direction, target, horizon_minutes)
    cid = "v4_" + sha256_json(spec)[:16]
    return AlgorithmCandidate(cid, cond_t, direction, target, horizon_minutes, source_model, discovery_sample, parent_id, lineage)  # type: ignore[arg-type]


def apply_conditions(df: pl.DataFrame, conditions: tuple[Condition, ...]) -> pl.DataFrame:
    out = df
    for c in conditions:
        out = out.filter(_cond_expr(c))
    return out


def apply_conditions_lf(lf: pl.LazyFrame, conditions: tuple[Condition, ...] | list[Condition]) -> pl.LazyFrame:
    for c in conditions:
        lf = lf.filter(_cond_expr(c))
    return lf


def _cond_expr(c: Condition) -> pl.Expr:
    col = pl.col(c.feature)
    if c.operator == ">":
        return col > c.threshold
    if c.operator == ">=":
        return col >= c.threshold
    if c.operator == "<":
        return col < c.threshold
    if c.operator == "<=":
        return col <= c.threshold
    return col == c.threshold


def signed_target(df: pl.DataFrame, target: str, direction: str) -> pl.Series:
    s = df[target]
    if direction == "SHORT":
        return -s
    return s
