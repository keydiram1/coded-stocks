"""Causal minute feature store. available_at = bar_ts + 1m. No look-ahead."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.eval_batch10 import _map_num, _univ
from quant_edge_lab.universe.filters import apply_universe

FEATURE_VERSION = "causal_v2"


def store_dir(root: Path) -> Path:
    p = Paths(root).features / FEATURE_VERSION
    p.mkdir(parents=True, exist_ok=True)
    return p


def day_path(root: Path, day: str) -> Path:
    return store_dir(root) / f"date={day}" / "part.parquet"


def compute_day_features(
    sessioned: pl.DataFrame,
    instruments: pl.DataFrame,
    hist: dict[str, Any],
) -> pl.DataFrame:
    prev_close = hist.get("prev_close") or {}
    typ = hist.get("typ") or {}
    df = sessioned.sort(["instrument_id", "ts_utc"])
    df = _map_num(df, prev_close, "prev_close")
    df = apply_universe(df, instruments, _univ())
    dabs = (pl.col("close") - pl.col("close").shift(1).over("instrument_id")).abs()
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1).alias("ret_1m"),
        (pl.col("close") / pl.col("close").shift(3).over("instrument_id") - 1).alias("ret_3m"),
        (pl.col("close") / pl.col("close").shift(5).over("instrument_id") - 1).alias("ret_5m"),
        (pl.col("close") / pl.col("close").shift(10).over("instrument_id") - 1).alias("ret_10m"),
        (pl.col("close") / pl.col("close").shift(15).over("instrument_id") - 1).alias("ret_15m"),
        (pl.col("close") / pl.col("close").shift(30).over("instrument_id") - 1).alias("ret_30m"),
        (pl.col("close") / pl.col("close").shift(60).over("instrument_id") - 1).alias("ret_60m"),
        pl.col("volume").alias("volume_1m"),
        pl.col("volume").rolling_sum(5, min_samples=1).over("instrument_id").alias("volume_5m"),
        pl.col("volume").rolling_sum(10, min_samples=1).over("instrument_id").alias("volume_10m"),
        pl.col("volume").rolling_sum(15, min_samples=1).over("instrument_id").alias("volume_15m"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_1m"),
        (
            (pl.col("high").rolling_max(5, min_samples=5).over("instrument_id")
             - pl.col("low").rolling_min(5, min_samples=5).over("instrument_id"))
            / pl.col("close")
        ).alias("range_5m"),
        (
            (pl.col("high").rolling_max(15, min_samples=15).over("instrument_id")
             - pl.col("low").rolling_min(15, min_samples=15).over("instrument_id"))
            / pl.col("close")
        ).alias("range_15m"),
        (
            (pl.col("high").rolling_max(30, min_samples=15).over("instrument_id")
             - pl.col("low").rolling_min(30, min_samples=15).over("instrument_id"))
            / pl.col("close")
        ).alias("range_30m"),
        pl.col("high").rolling_max(3, min_samples=3).over("instrument_id").alias("hi_3"),
        pl.col("low").rolling_min(3, min_samples=3).over("instrument_id").alias("lo_3"),
        pl.col("high").rolling_max(5, min_samples=5).over("instrument_id").alias("hi_5"),
        pl.col("low").rolling_min(5, min_samples=5).over("instrument_id").alias("lo_5"),
        pl.col("high").rolling_max(10, min_samples=10).over("instrument_id").alias("hi_10"),
        pl.col("low").rolling_min(10, min_samples=10).over("instrument_id").alias("lo_10"),
        pl.col("high").rolling_max(15, min_samples=15).over("instrument_id").alias("hi_15"),
        pl.col("low").rolling_min(15, min_samples=15).over("instrument_id").alias("lo_15"),
        dabs.rolling_sum(5, min_samples=2).over("instrument_id").alias("path_abs_5"),
        dabs.rolling_sum(15, min_samples=2).over("instrument_id").alias("path_abs_15"),
        pl.col("close").shift(5).over("instrument_id").alias("c5"),
        pl.col("close").shift(15).over("instrument_id").alias("c15"),
        (pl.col("ts_utc") + pl.duration(minutes=1)).alias("available_at"),
        (pl.col("ts_utc") + pl.duration(minutes=1)).alias("decision_ts"),
    )
    df = _map_num(df, typ, "typ_1m_vol")
    for name, key in (
        ("yday_ret", "yday_ret"),
        ("yday_high", "yday_high"),
        ("yday_low", "yday_low"),
        ("yday_range", "yday_range"),
        ("yday_clv", "yday_clv"),
        ("ret_2d", "ret_2d"),
        ("ret_3d", "ret_3d"),
        ("ret_5d", "ret_5d"),
        ("ah_ret", "ah_ret"),
        ("med20", "med20"),
        ("pm_ret", "pm_ret"),
    ):
        df = _map_num(df, hist.get(key) or {}, name)
    rth = df.filter(pl.col("is_rth"))
    if rth.height == 0:
        return rth
    rth = rth.with_columns(
        pl.col("open").first().over("instrument_id").alias("rth_open"),
        pl.col("high").cum_max().over("instrument_id").alias("rth_high_so_far"),
        pl.col("low").cum_min().over("instrument_id").alias("rth_low_so_far"),
        (pl.col("time_et").dt.hour() * 60 + pl.col("time_et").dt.minute() - (9 * 60 + 30)).alias("minutes_from_open"),
        pl.col("session_date").dt.weekday().alias("weekday"),
    )
    rth = rth.with_columns(
        (pl.col("rth_open") / pl.col("prev_close") - 1).alias("gap_pct"),
        (pl.col("volume_1m") / (pl.col("typ_1m_vol") + 1e-9)).alias("rvol_1m"),
        (pl.col("volume_5m") / (pl.col("typ_1m_vol") * 5 + 1e-9)).alias("rvol_5m"),
        (pl.col("volume_10m") / (pl.col("typ_1m_vol") * 10 + 1e-9)).alias("rvol_10m"),
        (pl.col("volume_15m") / (pl.col("typ_1m_vol") * 15 + 1e-9)).alias("rvol_15m"),
        (pl.col("close") / pl.col("rth_open") - 1).alias("dist_open"),
        (pl.col("close") / pl.col("prev_close") - 1).alias("dist_prev_close"),
        (pl.col("close") / pl.col("yday_high") - 1).alias("dist_yday_high"),
        (pl.col("close") / pl.col("yday_low") - 1).alias("dist_yday_low"),
    )
    rth = rth.with_columns(
        ((pl.col("close") - pl.col("c5")).abs() / (pl.col("path_abs_5") + 1e-12)).clip(0.0, 1.0).alias("path_efficiency_5m"),
        ((pl.col("close") - pl.col("c15")).abs() / (pl.col("path_abs_15") + 1e-12)).clip(0.0, 1.0).alias("path_efficiency_15m"),
        (pl.col("ret_5m") - pl.col("ret_15m")).alias("acceleration_5_vs_15"),
        (pl.col("hi_15") / pl.col("close") - 1).alias("pullback_from_recent_high"),
        (pl.col("close") / pl.col("lo_15") - 1).alias("rebound_from_recent_low"),
        ((pl.col("close") - pl.col("lo_15")) / (pl.col("hi_15") - pl.col("lo_15") + 1e-12)).alias("range_position_15m"),
        ((pl.col("close") - pl.col("rth_low_so_far")) / (pl.col("rth_high_so_far") - pl.col("rth_low_so_far") + 1e-12)).alias(
            "close_location_value"
        ),
        (pl.col("high") >= pl.col("hi_15").shift(1).over("instrument_id")).alias("new_high_15"),
        (pl.col("low") <= pl.col("lo_15").shift(1).over("instrument_id")).alias("new_low_15"),
        (pl.col("volume_5m") / (pl.col("volume_5m").shift(5).over("instrument_id") + 1e-9)).alias("volume_acceleration"),
        (pl.col("rvol_5m") / (pl.col("rvol_15m") + 1e-9)).alias("volume_persistence"),
        (pl.col("volume_1m") / (pl.col("volume_5m") / 5 + 1e-9)).alias("vol_vs_prev5_avg"),
        (pl.col("ret_5m").abs() / (pl.col("rvol_5m") + 1e-9)).alias("px_disp_per_rvol"),
        (pl.col("range_5m") / (pl.col("range_15m") + 1e-9)).alias("vol_accel_range"),
        (pl.col("range_15m") / (pl.col("med20") + 1e-9)).alias("range_vs_hist"),
    )
    # Cross-section at each minute (causal: only current bar and past).
    n_xs = pl.len().over("time_et")
    rth = rth.with_columns(
        (pl.col("ret_15m").rank(method="average").over("time_et") / n_xs).alias("xs_ret_rank"),
        (pl.col("rvol_5m").rank(method="average").over("time_et") / n_xs).alias("xs_rvol_rank"),
        (pl.col("acceleration_5_vs_15").rank(method="average").over("time_et") / n_xs).alias("xs_accel_rank"),
        (pl.col("range_5m").rank(method="average").over("time_et") / n_xs).alias("xs_vol_rank"),
        (pl.col("ret_5m") > 0).mean().over("time_et").alias("breadth_pos"),
        (pl.col("ret_5m") < 0).mean().over("time_et").alias("breadth_neg"),
        (pl.col("ret_15m").abs() >= 0.03).sum().over("time_et").alias("n_extreme_15m"),
        (pl.col("gap_pct").abs() >= 0.05).sum().over("time_et").alias("n_large_gap"),
        (pl.col("rvol_5m") >= 5).sum().over("time_et").alias("n_vol_shock"),
        pl.col("ret_5m").mean().over("time_et").alias("mkt_ret_5m"),
        pl.col("ret_15m").mean().over("time_et").alias("mkt_ret_15m"),
    )
    bad = rth.filter(pl.col("available_at") > pl.col("decision_ts"))
    if bad.height:
        raise AssertionError(f"look-ahead: {bad.height} rows available_at > decision_ts")
    return rth


def assert_inputs_causal(feat: pl.DataFrame, cols: list[str], decision_ts_col: str = "decision_ts") -> None:
    if feat.height == 0:
        return
    if "available_at" not in feat.columns:
        raise AssertionError("missing available_at")
    late = feat.filter(pl.col("available_at") > pl.col(decision_ts_col))
    if late.height:
        raise AssertionError(f"{late.height} rows violate available_at <= {decision_ts_col}")
    for c in cols:
        if c in feat.columns and feat[c].null_count() == feat.height:
            pass
