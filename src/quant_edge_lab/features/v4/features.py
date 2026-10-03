"""V4 continuous + legacy flag layer on a 5-minute decision grid. Does not modify V2 evaluators."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_edge_lab.features.v4.residuals import attach_loo_residuals

# Frozen batch10/V2 thresholds reused as FLAGS only.
H004_RVOL = 5.0
H004_ABS_RET = 0.012
IMPULSE_RET = 0.04
IMPULSE_RVOL = 3.0


def add_own_and_activity(feat: pl.DataFrame) -> pl.DataFrame:
    f = feat
    extra = []
    if "ret_5m" in f.columns:
        extra.append((pl.col("ret_5m") - pl.col("ret_5m").shift(5).over("instrument_id")).alias("ret_accel_5m"))
    if "ret_5m" in f.columns and "path_abs_5" in f.columns:
        extra.append((pl.col("ret_5m") / (pl.col("path_abs_5") / (pl.col("close") + 1e-12) + 1e-12)).alias("signed_path_eff_5m"))
    elif "path_efficiency_5m" in f.columns and "ret_5m" in f.columns:
        extra.append((pl.col("path_efficiency_5m") * pl.col("ret_5m").sign()).alias("signed_path_eff_5m"))
    if "close_location_value" in f.columns:
        extra.append(pl.col("close_location_value").alias("close_location_5m"))
    if "rth_high_so_far" in f.columns and "ret_5m" in f.columns:
        extra.append(((pl.col("rth_high_so_far") - pl.col("close")) / (pl.col("close") * (pl.col("ret_5m").abs().rolling_std(15).over("instrument_id") + 1e-6))).alias("dist_sess_high_volnorm"))
        extra.append(((pl.col("close") - pl.col("rth_low_so_far")) / (pl.col("close") * (pl.col("ret_5m").abs().rolling_std(15).over("instrument_id") + 1e-6))).alias("dist_sess_low_volnorm"))
    if "rvol_5m" in f.columns:
        extra.append(pl.col("rvol_5m").alias("rvol_5m_tod"))
    if "volume_5m" in f.columns and "volume_1m" in f.columns:
        extra.append((pl.col("volume_5m") / (pl.col("volume_5m").shift(5).over("instrument_id") + 1e-9)).alias("activity_acceleration"))
    if extra:
        f = f.with_columns(extra)
    return f


def add_legacy_flags(feat: pl.DataFrame) -> pl.DataFrame:
    f = feat
    rvol = pl.col("rvol_5m") if "rvol_5m" in f.columns else pl.lit(0.0)
    ret = pl.col("ret_5m") if "ret_5m" in f.columns else pl.lit(0.0)
    rvol10 = pl.col("rvol_10m") if "rvol_10m" in f.columns else rvol
    ret10 = pl.col("ret_10m") if "ret_10m" in f.columns else ret
    mkt = pl.col("mkt_ret_5m") if "mkt_ret_5m" in f.columns else pl.col("market_ret_5m") if "market_ret_5m" in f.columns else pl.lit(0.0)
    f = f.with_columns(
        ((rvol >= H004_RVOL) & (ret.abs() <= H004_ABS_RET)).alias("H004"),
        ((ret10.abs() >= IMPULSE_RET) & (rvol10 >= IMPULSE_RVOL)).alias("impulse"),
        (rvol >= 3.0).alias("high_rvol"),
        (ret.abs() <= H004_ABS_RET).alias("quiet_px"),
        (ret.sign() == mkt.sign()).alias("mkt_agree"),
    )
    f = f.with_columns(
        (pl.col("H004") & pl.col("impulse") & pl.col("high_rvol") & pl.col("quiet_px") & pl.col("mkt_agree")).alias("full_state"),
        (pl.col("impulse") & pl.col("high_rvol") & pl.col("quiet_px") & pl.col("mkt_agree")).alias("drop_H004"),
        (pl.col("H004") & pl.col("high_rvol") & pl.col("quiet_px") & pl.col("mkt_agree")).alias("drop_impulse"),
        (pl.col("H004") & pl.col("impulse") & pl.col("quiet_px") & pl.col("mkt_agree")).alias("drop_high_rvol"),
        (pl.col("H004") & pl.col("impulse") & pl.col("high_rvol") & pl.col("mkt_agree")).alias("drop_quiet_px"),
        (pl.col("H004") & pl.col("impulse") & pl.col("high_rvol") & pl.col("quiet_px")).alias("drop_mkt_agree"),
    )
    return f


def add_cross_section(feat: pl.DataFrame) -> pl.DataFrame:
    if feat.height == 0 or "resid_ret_5m" not in feat.columns:
        return feat
    n = pl.len().over("decision_ts")
    z = (pl.col("resid_ret_5m") - pl.col("resid_ret_5m").mean().over("decision_ts")) / (
        pl.col("resid_ret_5m").std().over("decision_ts") + 1e-12
    )
    feat = feat.with_columns(
        (pl.col("resid_ret_5m").rank(method="average").over("decision_ts") / n).alias("resid_rank_5m"),
        (pl.col("resid_ret_5m").abs().median().over("decision_ts")).alias("xs_dispersion_5m"),
        (pl.col("resid_ret_5m") > 0).mean().over("decision_ts").alias("xs_breadth_5m"),
        ((z > 2).mean().over("decision_ts") - (z < -2).mean().over("decision_ts")).alias("xs_tail_imbalance"),
    )
    feat = feat.with_columns(
        (pl.col("resid_rank_5m") - pl.col("resid_rank_5m").shift(1).over("instrument_id")).alias("resid_rank_change_5m")
    )
    return feat


def add_time_of_day(feat: pl.DataFrame) -> pl.DataFrame:
    if "minutes_from_open" not in feat.columns and "time_et" in feat.columns:
        feat = feat.with_columns((pl.col("time_et").dt.hour() * 60 + pl.col("time_et").dt.minute() - (9 * 60 + 30)).alias("minutes_from_open"))
    if "same_clock_mean_return" not in feat.columns and "ret_5m" in feat.columns:
        feat = feat.with_columns(pl.lit(None).cast(pl.Float64).alias("same_clock_mean_return"))
        feat = feat.with_columns((pl.col("ret_5m") - pl.col("same_clock_mean_return").fill_null(0)).alias("same_clock_surprise"))
    return feat


def add_activity_proxies(feat: pl.DataFrame) -> pl.DataFrame:
    f = feat
    cols = []
    if "n_trades" not in f.columns and "transactions" in f.columns:
        f = f.with_columns(pl.col("transactions").alias("n_trades"))
    if "volume" in f.columns and "n_trades" in f.columns:
        cols.append((pl.col("volume") / (pl.col("n_trades") + 1e-9)).alias("relative_trade_size_proxy"))
    elif "volume_1m" in f.columns:
        cols.append((pl.col("volume_1m") / (pl.col("volume_5m") / 5 + 1e-9)).alias("relative_trade_size_proxy"))
    if "resid_ret_5m" in f.columns and "dollar_volume_5m" in f.columns:
        cols.append((pl.col("resid_ret_5m").abs() / (pl.col("dollar_volume_5m") + 1e-9)).alias("price_impact_proxy"))
    elif "resid_ret_5m" in f.columns and "volume_5m" in f.columns and "close" in f.columns:
        cols.append((pl.col("resid_ret_5m").abs() / (pl.col("volume_5m") * pl.col("close") + 1e-9)).alias("price_impact_proxy"))
    if "rvol_5m" in f.columns and "ret_5m" in f.columns:
        cols.append((pl.col("rvol_5m") - pl.col("ret_5m").abs() * 100).alias("volume_price_disagreement"))
    if "range_15m" in f.columns and "med20" in f.columns:
        cols.append((pl.col("range_15m") / (pl.col("med20") + 1e-9)).alias("range_ratio_15m"))
    if "ret_15m" in f.columns:
        cols.append((pl.col("ret_15m").abs() / (pl.col("ret_5m").abs() + 1e-9)).alias("realized_vol_ratio_15m"))
    if cols:
        f = f.with_columns(cols)
    if "rtxn_5m_tod" not in f.columns:
        f = f.with_columns(pl.lit(None).cast(pl.Float64).alias("rtxn_5m_tod"))
    if "premarket_residual_return" not in f.columns:
        pm = pl.col("pm_ret") if "pm_ret" in f.columns else pl.lit(None).cast(pl.Float64)
        f = f.with_columns(pm.alias("premarket_residual_return"))
    if "missing_minute_rate_15m" not in f.columns:
        f = f.with_columns(pl.lit(0.0).alias("missing_minute_rate_15m"))
    if "gap_market_residual" not in f.columns:
        g = pl.col("gap_pct") if "gap_pct" in f.columns else pl.lit(None).cast(pl.Float64)
        f = f.with_columns(g.alias("gap_market_residual"))
    return f


def enrich_grid(feat: pl.DataFrame) -> pl.DataFrame:
    f = feat
    if "ticker" not in f.columns and "instrument_id" in f.columns:
        f = f.with_columns(pl.col("instrument_id").str.replace("^ticker:", "").alias("ticker"))
    if "weight_mkt" not in f.columns:
        if "med20_dvol" in f.columns:
            f = f.with_columns(pl.col("med20_dvol").sqrt().alias("weight_mkt"))
        elif "typ_1m_vol" in f.columns and "prev_close" in f.columns:
            f = f.with_columns((pl.col("typ_1m_vol") * pl.col("prev_close") * 390).sqrt().alias("weight_mkt"))
        else:
            f = f.with_columns(pl.lit(1.0).alias("weight_mkt"))
    if "beta_20d" not in f.columns:
        f = f.with_columns(pl.lit(1.0).alias("beta_20d"))
    if "decision_ts" not in f.columns and "ts_utc" in f.columns:
        f = f.with_columns((pl.col("ts_utc") + pl.duration(minutes=1)).alias("decision_ts"))
    ret_col = "ret_5m" if "ret_5m" in f.columns else None
    if ret_col:
        f = attach_loo_residuals(f, ret_col=ret_col, weight_col="weight_mkt", out_mkt="market_ret_5m", out_resid="resid_ret_5m", beta_col="beta_20d")
    if "ret_15m" in f.columns:
        f = attach_loo_residuals(f, ret_col="ret_15m", weight_col="weight_mkt", out_mkt="market_ret_15m", out_resid="resid_ret_15m", beta_col="beta_20d")
    f = add_own_and_activity(f)
    f = add_legacy_flags(f)
    f = add_cross_section(f)
    f = add_time_of_day(f)
    f = add_activity_proxies(f)
    for c in ("leader_shock_5m", "leader_agreement", "leader_response_gap", "peer_return_5m", "peer_response_gap", "peer_rank_gap", "peer_breadth", "peer_dispersion"):
        if c not in f.columns:
            f = f.with_columns(pl.lit(0.0).alias(c))
    return f


def attach_forward_residuals(grid: pl.DataFrame) -> pl.DataFrame:
    """Labels: shift residual returns forward. Future market may enter the LABEL only."""
    if grid.height == 0:
        return grid
    g = grid.sort(["instrument_id", "decision_ts"])
    if "resid_ret_5m" in g.columns:
        g = g.with_columns(pl.col("resid_ret_5m").shift(-1).over("instrument_id").alias("future_residual_5m"))
        g = g.with_columns(pl.col("resid_ret_5m").shift(-3).over("instrument_id").alias("future_residual_15m"))
        g = g.with_columns(pl.col("resid_ret_5m").shift(-6).over("instrument_id").alias("future_residual_30m"))
        g = g.with_columns(pl.col("resid_ret_5m").shift(-12).over("instrument_id").alias("future_residual_60m"))
    if "resid_rank_5m" in g.columns:
        g = g.with_columns(pl.col("resid_rank_5m").shift(-3).over("instrument_id").alias("future_residual_rank_15m"))
    if "ret_5m" in g.columns:
        g = g.with_columns(pl.col("ret_5m").shift(-3).over("instrument_id").alias("future_raw_15m"))
    return g
