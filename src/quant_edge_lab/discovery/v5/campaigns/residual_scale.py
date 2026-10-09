"""Own-instrument residual scale for continuation H4/H5. H1–H3 stay on |discrepancy|.

Scale is the sample std of stored same-clock residuals on the EXACT previous
N research-calendar sessions, each residual already
observed_final_5m − beta_t × eligible_LOO_final_5m at that session.
Missing any of those sessions makes z unavailable. No reaching back.
"""

from __future__ import annotations

import numpy as np
import polars as pl

MIN_PRIOR = 20
MAD_TO_SIGMA = 1.4826


def prior_calendar_sessions(
    day: str, calendar: list[str], *, lookback: int = MIN_PRIOR
) -> list[str] | None:
    """Exact `lookback` completed sessions immediately before `day` on the calendar."""
    if day not in calendar:
        return None
    i = calendar.index(day)
    if i < lookback:
        return None
    return list(calendar[i - lookback : i])


def residual_std_scale(
    window: list[float | None], *, require_n: int = MIN_PRIOR
) -> float | None:
    """Sample std of an exact-length session window. Any missing/nonfinite -> None."""
    if len(window) != require_n:
        return None
    vals: list[float] = []
    for v in window:
        if v is None or not np.isfinite(v):
            return None
        vals.append(float(v))
    if np.allclose(vals, vals[0]):
        return None
    s = float(np.std(vals, ddof=1))
    if not np.isfinite(s) or s <= 0:
        return None
    return s


def residual_mad_scale(
    window: list[float | None], *, require_n: int = MIN_PRIOR
) -> float | None:
    """Robust MAD scale. Recorded; H4/H5 use sample std."""
    if len(window) != require_n:
        return None
    vals: list[float] = []
    for v in window:
        if v is None or not np.isfinite(v):
            return None
        vals.append(float(v))
    arr = np.asarray(vals, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
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


def scale_for_event(
    residuals_by_day: dict[str, float],
    day: str,
    calendar: list[str],
    *,
    lookback: int = MIN_PRIOR,
) -> float | None:
    sessions = prior_calendar_sessions(day, calendar, lookback=lookback)
    if sessions is None:
        return None
    window = [residuals_by_day.get(s) for s in sessions]
    return residual_std_scale(window, require_n=lookback)


def attach_normalized_dislocation(
    events: pl.DataFrame,
    *,
    calendar: list[str],
    min_prior: int = MIN_PRIOR,
) -> pl.DataFrame:
    """Attach residual_scale and normalized_dislocation using the research calendar."""
    if events.height == 0:
        return events.with_columns(
            pl.lit(None).cast(pl.Float64).alias("residual_scale"),
            pl.lit(None).cast(pl.Float64).alias("normalized_dislocation"),
        )
    by_iid: dict[str, dict[str, float]] = {}
    for row in events.iter_rows(named=True):
        disc = row.get("discrepancy")
        if disc is None or not np.isfinite(disc):
            continue
        iid = str(row["instrument_id"])
        day = str(row["trading_date"])
        by_iid.setdefault(iid, {})[day] = float(disc)
    ordered = events.with_row_index("_rid")
    scales: list[float | None] = []
    zs: list[float | None] = []
    for row in ordered.iter_rows(named=True):
        iid = str(row["instrument_id"])
        day = str(row["trading_date"])
        scale = scale_for_event(by_iid.get(iid, {}), day, calendar, lookback=min_prior)
        zs.append(normalized_dislocation(row.get("discrepancy"), scale))
        scales.append(scale)
    return ordered.with_columns(
        pl.Series("residual_scale", scales, dtype=pl.Float64),
        pl.Series("normalized_dislocation", zs, dtype=pl.Float64),
    ).drop("_rid")
