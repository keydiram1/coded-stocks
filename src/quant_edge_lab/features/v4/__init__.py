"""V4 timestamp and decision-grid contract. Naive times are UTC instants."""

from __future__ import annotations

from datetime import time

import polars as pl

from quant_edge_lab.data.massive.news import CANONICAL_UTC, with_utc_us

DECISION_START = time(9, 40)
DECISION_END = time(15, 30)
GRID_STEP_MIN = 5


def ensure_utc(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return with_utc_us(df, col)


def assert_join_keys_utc(left: pl.DataFrame, right: pl.DataFrame, a: str, b: str) -> None:
    if left.schema[a] != CANONICAL_UTC or right.schema[b] != CANONICAL_UTC:
        raise TypeError(f"join keys must be {CANONICAL_UTC}: {left.schema[a]} vs {right.schema[b]}")
    if left.schema[a] != right.schema[b]:
        raise TypeError("join key dtypes differ")


def on_decision_grid(df: pl.DataFrame) -> pl.DataFrame:
    if "time_et" not in df.columns:
        raise ValueError("time_et required")
    mins = pl.col("time_et").dt.hour().cast(pl.Int32) * 60 + pl.col("time_et").dt.minute().cast(pl.Int32)
    start = 9 * 60 + 40
    end = 15 * 60 + 30
    return df.filter((mins >= start) & (mins <= end) & ((mins - start) % GRID_STEP_MIN == 0))


def assert_available_le_decision(df: pl.DataFrame, avail: str, decision: str) -> None:
    if df.height == 0:
        return
    late = df.filter(pl.col(avail) > pl.col(decision))
    if late.height:
        raise AssertionError(f"PIT violation: {late.height} rows {avail} > {decision}")
