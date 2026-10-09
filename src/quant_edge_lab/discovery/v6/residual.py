"""Market-LOO residual, same-clock shock_z, unclipped retention. No I/O."""

from __future__ import annotations

import numpy as np

from quant_edge_lab.discovery.v5.campaigns.residual_scale import residual_std_scale
from quant_edge_lab.features.v4.residuals import loo_market_returns

SCALE_LOOKBACK = 20


def simple_return(p_end: float | None, p_start: float | None) -> float | None:
    if p_end is None or p_start is None:
        return None
    a, b = float(p_end), float(p_start)
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return a / b - 1.0


def market_residual(
    r_stock: float | None, beta: float | None, r_market: float | None
) -> float | None:
    if r_stock is None or beta is None or r_market is None:
        return None
    s, b, m = float(r_stock), float(beta), float(r_market)
    if not np.isfinite(s) or not np.isfinite(b) or not np.isfinite(m):
        return None
    return s - b * m


def equal_weight_loo(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    w = np.ones_like(r)
    return loo_market_returns(r, w)


def retention_ratio(remaining: float | None, impulse: float | None) -> float | None:
    if remaining is None or impulse is None:
        return None
    rem, imp = float(remaining), float(impulse)
    if not np.isfinite(rem) or not np.isfinite(imp) or imp == 0.0:
        return None
    return rem / imp


def shock_z(impulse_residual: float | None, prior_20: list[float | None] | None) -> float | None:
    if impulse_residual is None or not np.isfinite(float(impulse_residual)):
        return None
    if prior_20 is None:
        return None
    scale = residual_std_scale(prior_20, require_n=SCALE_LOOKBACK)
    if scale is None:
        return None
    return float(impulse_residual) / scale


def impulse_sign(impulse_residual: float | None) -> str | None:
    if impulse_residual is None or not np.isfinite(float(impulse_residual)):
        return None
    x = float(impulse_residual)
    if x > 0:
        return "POSITIVE"
    if x < 0:
        return "NEGATIVE"
    return None
