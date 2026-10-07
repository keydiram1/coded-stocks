"""Causal 20-day OLS beta. Insufficient history is NULL. Never fill with 1.0."""

from __future__ import annotations

import numpy as np
import polars as pl

BETA_WINDOW = 20
MIN_PAIRS = 20


def ols_beta(stock: np.ndarray, market: np.ndarray) -> float:
    """Covariance-form slope (OLS with intercept). Requires exactly MIN_PAIRS finite pairs."""
    s = np.asarray(stock, dtype=float)
    m = np.asarray(market, dtype=float)
    mask = np.isfinite(s) & np.isfinite(m)
    if int(mask.sum()) < MIN_PAIRS:
        return float("nan")
    y = s[mask]
    x = m[mask]
    xm = float(x.mean())
    ym = float(y.mean())
    varx = float(np.dot(x - xm, x - xm))
    if varx < 1e-18:
        return float("nan")
    return float(np.dot(x - xm, y - ym) / varx)


def daily_return_panel(grid: pl.DataFrame) -> pl.DataFrame:
    """One row per ticker: compounded 5m returns for the day. Weight from first row."""
    if grid.height == 0 or "ret_5m" not in grid.columns:
        return pl.DataFrame()
    wcol = "weight_mkt" if "weight_mkt" in grid.columns else None
    agg = [
        ((1.0 + pl.col("ret_5m").fill_null(0.0)).product() - 1.0).alias("daily_ret"),
        pl.col("ret_5m").len().alias("n_bars"),
    ]
    if wcol:
        agg.append(pl.col(wcol).first().alias("weight_mkt"))
    out = grid.group_by("instrument_id").agg(agg)
    if "trading_date" in grid.columns:
        out = out.with_columns(pl.lit(grid["trading_date"][0]).alias("trading_date"))
    return out


def loo_daily_market(daily: pl.DataFrame) -> pl.DataFrame:
    from quant_edge_lab.features.v4.residuals import loo_market_returns, winsor_weights

    if daily.height == 0 or "daily_ret" not in daily.columns:
        return daily.with_columns(pl.lit(None).cast(pl.Float64).alias("daily_mkt_loo"))
    r = daily["daily_ret"].to_numpy().astype(float)
    w = daily["weight_mkt"].to_numpy().astype(float) if "weight_mkt" in daily.columns else np.ones(len(r))
    w = winsor_weights(w)
    m = loo_market_returns(r, w)
    return daily.with_columns(pl.Series("daily_mkt_loo", m))


def beta_from_history(history: list[pl.DataFrame], tickers: list[str]) -> dict[str, float]:
    """history: prior completed days, oldest→newest. Current day must not be included."""
    if len(history) < BETA_WINDOW:
        return {}
    use = history[-BETA_WINDOW:]
    if not tickers:
        ids: set[str] = set()
        for day in use:
            if day.height and "instrument_id" in day.columns:
                ids.update(str(x) for x in day["instrument_id"].to_list())
        tickers = sorted(ids)
    by: dict[str, list[tuple[float, float]]] = {t: [] for t in tickers}
    for day in use:
        if day.height == 0:
            continue
        for row in day.select(["instrument_id", "daily_ret", "daily_mkt_loo"]).iter_rows(named=True):
            tid = str(row["instrument_id"])
            if tid not in by:
                continue
            sr, mr = row["daily_ret"], row["daily_mkt_loo"]
            if sr is None or mr is None:
                continue
            by[tid].append((float(sr), float(mr)))
    out: dict[str, float] = {}
    for tid, pairs in by.items():
        if len(pairs) < MIN_PAIRS:
            continue
        s = np.array([p[0] for p in pairs[-MIN_PAIRS:]], dtype=float)
        m = np.array([p[1] for p in pairs[-MIN_PAIRS:]], dtype=float)
        b = ols_beta(s, m)
        if np.isfinite(b):
            out[tid] = b
    return out


def attach_beta(grid: pl.DataFrame, beta_map: dict[str, float]) -> pl.DataFrame:
    if "beta_20d" in grid.columns:
        grid = grid.drop("beta_20d")
    if grid.height == 0:
        return grid.with_columns(pl.lit(None).cast(pl.Float64).alias("beta_20d"))
    if not beta_map:
        return grid.with_columns(pl.lit(None).cast(pl.Float64).alias("beta_20d"))
    bdf = pl.DataFrame({"instrument_id": list(beta_map.keys()), "beta_20d": [float(v) for v in beta_map.values()]}).with_columns(
        pl.col("instrument_id").cast(grid.schema["instrument_id"])
    )
    return grid.join(bdf, on="instrument_id", how="left")
