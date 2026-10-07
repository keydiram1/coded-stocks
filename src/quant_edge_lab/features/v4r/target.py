"""True forward 15-minute residual. No day wrap. Beta known at t. Future market in LABEL only."""

from __future__ import annotations

import polars as pl


def _compound(exprs: list[pl.Expr]) -> pl.Expr:
    out = 1.0 + exprs[0]
    for e in exprs[1:]:
        out = out * (1.0 + e)
    return out - 1.0


def attach_forward_holding_residuals(grid: pl.DataFrame) -> pl.DataFrame:
    """Within (instrument_id, trading_date) only. Incomplete forward windows are NULL."""
    if grid.height == 0:
        return grid
    g = grid.sort(["instrument_id", "trading_date", "decision_ts"])
    over = ["instrument_id", "trading_date"]
    r = pl.col("ret_5m")
    m = pl.col("market_ret_5m") if "market_ret_5m" in g.columns else pl.lit(None).cast(pl.Float64)
    rlegs = [r.shift(-i).over(over) for i in range(1, 4)]
    mlegs = [m.shift(-i).over(over) for i in range(1, 4)]
    stock15 = _compound(rlegs)
    mkt15 = _compound(mlegs)
    beta = pl.col("beta_20d") if "beta_20d" in g.columns else pl.lit(None).cast(pl.Float64)
    g = g.with_columns(
        rlegs[0].alias("fwd_ret_leg1_5m"),
        rlegs[1].alias("fwd_ret_leg2_5m"),
        rlegs[2].alias("fwd_ret_leg3_5m"),
        stock15.alias("future_raw_15m"),
        mkt15.alias("future_market_15m"),
        pl.when(beta.is_not_null() & stock15.is_not_null() & mkt15.is_not_null())
        .then(stock15 - beta * mkt15)
        .otherwise(None)
        .alias("future_residual_15m"),
        r.shift(-1).over(over).alias("future_raw_5m"),
    )
    if "resid_ret_5m" in g.columns:
        g = g.with_columns(pl.col("resid_ret_5m").shift(-1).over(over).alias("future_residual_5m"))
    r6 = [r.shift(-i).over(over) for i in range(1, 7)]
    m6 = [m.shift(-i).over(over) for i in range(1, 7)]
    stock30 = _compound(r6)
    mkt30 = _compound(m6)
    g = g.with_columns(
        stock30.alias("future_raw_30m"),
        pl.when(beta.is_not_null() & stock30.is_not_null() & mkt30.is_not_null())
        .then(stock30 - beta * mkt30)
        .otherwise(None)
        .alias("future_residual_30m"),
    )
    return g
