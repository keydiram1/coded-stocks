"""PIT / causality assertions for V5 events."""

from __future__ import annotations

from datetime import datetime

import polars as pl


def assert_available_le_decision(
    available_at: datetime | None,
    decision_ts: datetime,
    *,
    label: str = "available_at",
) -> None:
    if available_at is None:
        raise AssertionError(f"{label} missing")
    if available_at > decision_ts:
        raise AssertionError(f"PIT violation: {label}={available_at} > decision_ts={decision_ts}")


def assert_frame_available_le_decision(
    df: pl.DataFrame,
    avail_col: str = "first_available_at",
    decision_col: str = "decision_ts",
) -> None:
    if df.height == 0:
        return
    if avail_col not in df.columns or decision_col not in df.columns:
        raise AssertionError(f"missing {avail_col} or {decision_col}")
    late = df.filter(pl.col(avail_col) > pl.col(decision_col))
    if late.height:
        raise AssertionError(f"PIT violation: {late.height} rows {avail_col} > {decision_col}")


def assert_no_future_columns_in_events(df: pl.DataFrame) -> None:
    banned = [c for c in df.columns if c.startswith("future_") or c.startswith("fwd_")]
    if banned:
        raise AssertionError(f"future outcome columns must not enter event construction: {banned}")
