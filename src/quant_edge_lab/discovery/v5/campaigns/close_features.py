"""Pure CLOSE_DISLOCATION arithmetic. No I/O. No threshold constants."""

from __future__ import annotations

from datetime import time

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.models import Direction
from quant_edge_lab.features.v4.residuals import loo_market_returns, winsor_weights
from quant_edge_lab.features.v4r.beta import BETA_WINDOW, MIN_PAIRS, beta_from_history, daily_return_panel, loo_daily_market, ols_beta

CLOSE_ANCHOR = time(15, 54)
CLOSE_LAST = time(15, 59)


def discrepancy(observed: float | None, expected: float | None) -> float | None:
    if observed is None or expected is None:
        return None
    if not np.isfinite(observed) or not np.isfinite(expected):
        return None
    return float(observed) - float(expected)


def expected_final_5m(beta: float | None, market_loo: float | None) -> float | None:
    if beta is None or market_loo is None:
        return None
    if not np.isfinite(beta) or not np.isfinite(market_loo):
        return None
    return float(beta) * float(market_loo)


def resolution_direction(disc: float | None) -> Direction | None:
    if disc is None or not np.isfinite(disc) or disc == 0.0:
        return None
    return "SHORT" if disc > 0 else "LONG"


def signed_resolution(forward: float | None, direction: Direction | None) -> float | None:
    if forward is None or direction is None or not np.isfinite(forward):
        return None
    sign = 1.0 if direction == "LONG" else -1.0
    return sign * float(forward)


def close_volume_rvol(current_vol: float | None, prior_vols: list[float]) -> float | None:
    if current_vol is None or not np.isfinite(current_vol):
        return None
    base = [v for v in prior_vols if v is not None and np.isfinite(v) and v > 0]
    if not base:
        return None
    den = float(np.mean(base))
    if den <= 0:
        return None
    return float(current_vol) / den


def bar_return(close_now: float, close_then: float) -> float | None:
    if close_then is None or close_now is None:
        return None
    if not np.isfinite(close_then) or not np.isfinite(close_now) or close_then == 0:
        return None
    return float(close_now) / float(close_then) - 1.0


def loo_cross_section(returns: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=float)
    w = winsor_weights(w) if w.size else w
    return loo_market_returns(r, w)


def final_5m_from_named_bars(anchor_close: float, last_close: float) -> float | None:
    """15:59 close / 15:54 close − 1. Same session only; caller must guarantee that."""
    return bar_return(last_close, anchor_close)


def pick_bar_close(g: pl.DataFrame, t: time) -> float | None:
    sub = g.filter(pl.col("time_et") == t)
    if sub.height == 0:
        return None
    return float(sub["close"][-1])


def pick_window_volume(g: pl.DataFrame) -> float | None:
    sub = g.filter((pl.col("time_et") >= time(15, 55)) & (pl.col("time_et") <= CLOSE_LAST))
    if sub.height == 0:
        return None
    return float(sub["volume"].sum())


def pick_open(g: pl.DataFrame) -> tuple[float | None, object | None]:
    rth = g.sort("time_et")
    if rth.height == 0:
        return None, None
    row = rth.row(0, named=True)
    return float(row["open"]), row.get("ts_utc")


def pick_close_px(g: pl.DataFrame) -> float | None:
    last = g.filter(pl.col("time_et") == CLOSE_LAST)
    if last.height == 0:
        return None
    return float(last["close"][-1])


def horizon_close_after_open(g: pl.DataFrame, minutes: int) -> float | None:
    """Close of the bar minutes after RTH open (09:30 + minutes - 1 minute label)."""
    start_m = 9 * 60 + 30 + minutes - 1
    h, m = divmod(start_m, 60)
    target = time(h, m)
    sub = g.filter(pl.col("time_et") == target)
    if sub.height == 0:
        return None
    return float(sub["close"][-1])


__all__ = [
    "BETA_WINDOW",
    "MIN_PAIRS",
    "bar_return",
    "beta_from_history",
    "close_volume_rvol",
    "daily_return_panel",
    "discrepancy",
    "expected_final_5m",
    "final_5m_from_named_bars",
    "horizon_close_after_open",
    "loo_cross_section",
    "loo_daily_market",
    "ols_beta",
    "pick_bar_close",
    "pick_close_px",
    "pick_open",
    "pick_window_volume",
    "resolution_direction",
    "signed_resolution",
]
