"""Own-instrument residual scale. Not used by frozen reversal or continuation H1–H3.

Denominator is computed from prior completed same-clock residuals only.
Current-day and future residuals must not enter the scale.
"""

from __future__ import annotations

import numpy as np

MIN_PRIOR = 20
MAD_TO_SIGMA = 1.4826


def _finite_prior(values: list[float | None]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        x = float(v)
        if np.isfinite(x):
            out.append(x)
    return out


def residual_std_scale(
    prior_residuals: list[float | None], *, min_prior: int = MIN_PRIOR
) -> float | None:
    """Sample std of prior same-clock 5m residuals. Insufficient history -> None."""
    use = _finite_prior(prior_residuals)
    if len(use) < min_prior:
        return None
    window = use[-min_prior:]
    if np.allclose(window, window[0]):
        return None
    s = float(np.std(window, ddof=1))
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def residual_mad_scale(
    prior_residuals: list[float | None], *, min_prior: int = MIN_PRIOR
) -> float | None:
    """Robust MAD scale. Recorded for a future campaign; not selected here."""
    use = _finite_prior(prior_residuals)
    if len(use) < min_prior:
        return None
    window = np.asarray(use[-min_prior:], dtype=float)
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med)))
    s = MAD_TO_SIGMA * mad
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def normalized_dislocation(residual: float | None, scale: float | None) -> float | None:
    if residual is None or scale is None:
        return None
    r = float(residual)
    s = float(scale)
    if not np.isfinite(r) or not np.isfinite(s) or s <= 0:
        return None
    return r / s


def prior_residuals_only(
    history: list[tuple[str, float]],
    *,
    day: str,
) -> list[float | None]:
    """history is (session_date, residual). Keep strictly earlier sessions."""
    return [v for d, v in history if d < day]


def update_residual_history(
    history: dict[str, list[tuple[str, float]]],
    *,
    iid: str,
    day: str,
    residual: float | None,
    lookback: int = MIN_PRIOR,
) -> None:
    """Append after the signal day is complete. Do not use this update for today's scale."""
    if residual is None or not np.isfinite(residual):
        return
    rec = history.setdefault(iid, [])
    rec.append((day, float(residual)))
    history[iid] = rec[-lookback:]
