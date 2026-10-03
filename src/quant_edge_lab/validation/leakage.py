"""Causality / leakage helpers used by tests and the pipeline."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from quant_edge_lab.features.engine import compute_features


def truncate_at_event(bars: pl.DataFrame, decision_ts: datetime) -> pl.DataFrame:
    """Keep only bars whose close (ts+1m) is <= decision_ts."""
    return bars.filter(pl.col("ts_utc") + pl.duration(minutes=1) <= decision_ts)


def future_poison(bars: pl.DataFrame, after_ts: datetime, factor: float = 10.0) -> pl.DataFrame:
    """Aggressively mutate prices after a timestamp. Past bars must be unchanged."""
    return bars.with_columns(
        pl.when(pl.col("ts_utc") > after_ts)
        .then(pl.col("close") * factor)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("ts_utc") > after_ts)
        .then(pl.col("open") * factor)
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.when(pl.col("ts_utc") > after_ts)
        .then(pl.col("high") * factor)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.col("ts_utc") > after_ts)
        .then(pl.col("low") * factor)
        .otherwise(pl.col("low"))
        .alias("low"),
    )


def assert_available(source_available_at: datetime, decision_ts: datetime) -> None:
    if source_available_at > decision_ts:
        raise AssertionError(
            f"Look-ahead: source_available_at={source_available_at} > decision_ts={decision_ts}"
        )


def pit_float_join(
    events: pl.DataFrame,
    floats: pl.DataFrame,
) -> pl.DataFrame:
    """Join float observations only when effective_at <= decision_ts.

    Current (observed_at) snapshots without an effective timestamp must not be used.
    """
    f = floats.filter(pl.col("effective_at").is_not_null())
    out = events.join_asof(
        f.sort("effective_at"),
        left_on="decision_ts",
        right_on="effective_at",
        by="instrument_id",
        strategy="backward",
        check_sortedness=False,
    )
    return out


def features_at_decision(bars: pl.DataFrame, instrument_id: str, decision_ts: datetime) -> dict:
    feat = compute_features(bars)
    row = feat.filter(
        (pl.col("instrument_id") == instrument_id) & (pl.col("decision_ts") == decision_ts)
    )
    if row.height != 1:
        raise AssertionError(f"expected 1 feature row, got {row.height}")
    return row.to_dicts()[0]
