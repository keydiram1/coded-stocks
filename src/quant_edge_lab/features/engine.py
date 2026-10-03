"""Point-in-time feature engine.

Features on bar timestamp `ts` include that bar's OHLCV and are available at ts + 1 minute
(the bar close). They must not be used at the bar's open.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from quant_edge_lab.universe.filters import add_session_columns

FEATURE_VERSION = "features_v1"
BAR_MINUTES = 1


def available_at_expr(ts_col: str = "ts_utc") -> pl.Expr:
    return pl.col(ts_col) + pl.duration(minutes=BAR_MINUTES)


def compute_features(bars: pl.DataFrame) -> pl.DataFrame:
    df = add_session_columns(bars)
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1).alias("ret_1m"),
        (pl.col("close") / pl.col("close").shift(5).over("instrument_id") - 1).alias("ret_5m"),
        (pl.col("close") / pl.col("close").shift(15).over("instrument_id") - 1).alias("ret_15m"),
        pl.col("volume").rolling_sum(window_size=5, min_samples=1).over("instrument_id").alias("roll_vol_5m"),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        (pl.col("close") * pl.col("volume"))
        .rolling_sum(window_size=5, min_samples=1)
        .over("instrument_id")
        .alias("roll_dvol_5m"),
    )

    rth = df.filter(pl.col("is_rth"))
    daily = rth.group_by(["instrument_id", "session_date"]).agg(
        pl.col("close").last().alias("rth_close"),
        pl.col("open").first().alias("rth_open"),
        pl.col("volume").median().alias("rth_med_vol"),
        (pl.col("close") * pl.col("volume")).sum().alias("day_dollar_volume"),
    ).sort(["instrument_id", "session_date"])
    daily = daily.with_columns(
        pl.col("rth_close").shift(1).over("instrument_id").alias("prev_close"),
        pl.col("rth_med_vol")
        .shift(1)
        .rolling_median(window_size=20, min_samples=1)
        .over("instrument_id")
        .alias("typ_1m_vol"),
        pl.col("day_dollar_volume")
        .shift(1)
        .rolling_median(window_size=20, min_samples=1)
        .over("instrument_id")
        .alias("median_20d_dollar_volume"),
    )

    pm = df.filter(pl.col("is_premarket")).group_by(["instrument_id", "session_date"]).agg(
        pl.col("high").max().alias("premarket_high"),
        pl.col("low").min().alias("premarket_low"),
        pl.col("volume").sum().alias("premarket_volume"),
    )

    df = df.join(
        daily.select(
            [
                "instrument_id",
                "session_date",
                "prev_close",
                "rth_open",
                "typ_1m_vol",
                "median_20d_dollar_volume",
            ]
        ),
        on=["instrument_id", "session_date"],
        how="left",
    ).join(pm, on=["instrument_id", "session_date"], how="left")

    df = df.with_columns(
        (pl.col("rth_open") / pl.col("prev_close") - 1).alias("gap_pct"),
        (pl.col("roll_vol_5m") / (pl.col("typ_1m_vol") * 5)).alias("rvol_5m"),
        (pl.col("ret_5m") - pl.col("ret_15m") / 3.0).alias("acceleration"),
    )

    # Session VWAP / high / low from RTH start, running within the day.
    df = df.with_columns(
        pl.when(pl.col("is_rth")).then(pl.col("close") * pl.col("volume")).otherwise(0.0).alias("_pv"),
        pl.when(pl.col("is_rth")).then(pl.col("volume")).otherwise(0.0).alias("_v"),
        pl.when(pl.col("is_rth")).then(pl.col("high")).otherwise(None).alias("_h"),
        pl.when(pl.col("is_rth")).then(pl.col("low")).otherwise(None).alias("_l"),
        pl.when(pl.col("is_rth")).then(pl.col("volume")).otherwise(0.0).alias("_cv"),
    ).with_columns(
        pl.col("_pv").cum_sum().over(["instrument_id", "session_date"]).alias("_cum_pv"),
        pl.col("_v").cum_sum().over(["instrument_id", "session_date"]).alias("_cum_v"),
        pl.col("_cv").cum_sum().over(["instrument_id", "session_date"]).alias("cum_intraday_volume"),
        pl.col("_h").cum_max().over(["instrument_id", "session_date"]).alias("session_high"),
        pl.col("_l").cum_min().over(["instrument_id", "session_date"]).alias("session_low"),
    ).with_columns(
        (pl.col("_cum_pv") / pl.col("_cum_v")).alias("session_vwap"),
    ).with_columns(
        (pl.col("close") / pl.col("session_vwap") - 1).alias("dist_from_vwap"),
        (pl.col("close") / pl.col("session_high") - 1).alias("dist_from_session_high"),
        (pl.col("close") / pl.col("session_low") - 1).alias("dist_from_session_low"),
        (pl.col("close") / pl.col("premarket_high") - 1).alias("dist_from_premarket_high"),
    ).with_columns(
        available_at_expr().alias("available_at"),
        (pl.col("ts_utc") + pl.duration(minutes=1)).alias("decision_ts"),
    )

    drop = [c for c in df.columns if c.startswith("_")]
    return df.drop(drop)


def feature_columns() -> list[str]:
    return [
        "prev_close",
        "gap_pct",
        "ret_1m",
        "ret_5m",
        "ret_15m",
        "roll_dvol_5m",
        "dollar_volume",
        "median_20d_dollar_volume",
        "roll_vol_5m",
        "rvol_5m",
        "cum_intraday_volume",
        "session_vwap",
        "dist_from_vwap",
        "dist_from_session_high",
        "dist_from_session_low",
        "premarket_high",
        "premarket_low",
        "premarket_volume",
        "dist_from_premarket_high",
        "acceleration",
    ]
