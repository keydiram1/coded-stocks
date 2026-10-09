"""Continuation-only inference. Does not change frozen reversal day_block_stats."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.evaluation import concentration, finite_primary, onesided_greater_p

ESTIMAND = "equal_weight_trading_day_mean"


def _bootstrap_daily_means(daily: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = daily.size
    dist = np.empty(n_boot)
    for i in range(n_boot):
        dist[i] = float(np.mean(rng.choice(daily, size=n, replace=True)))
    return dist


def equal_weight_day_stats(
    events: pl.DataFrame, col: str = "primary_signed", *, n_boot: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Primary mean/SE/CI/p/floor target = equal-weight mean of trading-day means."""
    ydf = finite_primary(events, col)
    conc = concentration(ydf)
    empty = {
        **conc,
        "mean": None,
        "mean_bp": None,
        "se": None,
        "ci": None,
        "p_one_sided": 1.0,
        "daily_means": [],
        "median": None,
        "win_rate": None,
        "event_median": None,
        "event_win_rate": None,
        "primary_estimand": ESTIMAND,
        "inference_unit": "trading_day",
    }
    if ydf.height == 0:
        return empty
    y = ydf[col].to_numpy().astype(float)
    by = ydf.group_by("trading_date").agg(pl.col(col).mean().alias("m")).sort("trading_date")
    dm = by["m"].drop_nulls().to_numpy().astype(float)
    if dm.size == 0:
        return empty
    mean = float(np.mean(dm))
    se = float(np.std(dm, ddof=1) / np.sqrt(dm.size)) if dm.size > 1 else None
    ci = None
    if dm.size > 1:
        dist = _bootstrap_daily_means(dm, n_boot, seed)
        ci = [float(np.nanquantile(dist, 0.025)), float(np.nanquantile(dist, 0.975))]
    event_median = float(np.median(y))
    event_wr = float(np.mean(y > 0))
    return {
        **conc,
        "mean": mean,
        "mean_bp": mean * 10_000.0,
        "se": se,
        "ci": ci,
        "p_one_sided": onesided_greater_p(dm),
        "daily_means": [float(x) for x in dm],
        "median": event_median,
        "win_rate": event_wr,
        "event_median": event_median,
        "event_win_rate": event_wr,
        "primary_estimand": ESTIMAND,
        "inference_unit": "trading_day",
        "direction_agreement": float(np.mean(dm > 0)),
    }
