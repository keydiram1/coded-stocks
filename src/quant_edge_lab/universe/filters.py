from __future__ import annotations

from datetime import time

import polars as pl

from quant_edge_lab.models.hypothesis import UniverseSpec


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def add_session_columns(bars: pl.DataFrame, tz: str = "America/New_York") -> pl.DataFrame:
    return bars.sort(["instrument_id", "ts_utc"]).with_columns(
        pl.col("ts_utc")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(tz)
        .alias("ts_et")
    ).with_columns(
        pl.col("ts_et").dt.date().alias("session_date"),
        pl.col("ts_et").dt.time().alias("time_et"),
    ).with_columns(
        (pl.col("time_et") >= time(9, 30)).and_(pl.col("time_et") < time(16, 0)).alias("is_rth"),
        (pl.col("time_et") >= time(4, 0)).and_(pl.col("time_et") < time(9, 30)).alias("is_premarket"),
    )


def apply_universe(
    featured: pl.DataFrame,
    instruments: pl.DataFrame,
    spec: UniverseSpec,
) -> pl.DataFrame:
    inst = instruments.select(["instrument_id", "exchange", "security_type"])
    df = featured.join(inst, on="instrument_id", how="left")
    df = df.filter(pl.col("security_type") == spec.security_type)
    df = df.filter(pl.col("exchange").is_in(spec.exchanges))
    # Point-in-time eligibility uses previous close, never current snapshot metadata.
    df = df.filter(
        pl.col("prev_close").is_not_null()
        & (pl.col("prev_close") >= spec.price_prev_close.min)
        & (pl.col("prev_close") <= spec.price_prev_close.max)
    )
    return df
