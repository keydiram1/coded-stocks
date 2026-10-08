"""Regular-session / half-day classification from the minute panel."""

from __future__ import annotations

from datetime import time

import polars as pl

from quant_edge_lab.universe.filters import add_session_columns

FULL_DAY_LAST_BAR = time(15, 59)
CLOSE_FIRST = time(15, 55)
CLOSE_LAST = time(15, 59)
CLOSE_ANCHOR = time(15, 54)
MIN_RTH_MINUTES = 380
RTH_OPEN = time(9, 30)
OPEN_PLUS_5 = time(9, 34)
OPEN_PLUS_15 = time(9, 44)
OPEN_PLUS_30 = time(9, 59)
CLOSE_WINDOW_TIMES = (
    time(15, 54),
    time(15, 55),
    time(15, 56),
    time(15, 57),
    time(15, 58),
    time(15, 59),
)


def _hhmm(t: time) -> int:
    return t.hour * 60 + t.minute


def with_session(bars: pl.DataFrame) -> pl.DataFrame:
    if "time_et" in bars.columns and "is_rth" in bars.columns:
        return bars
    return add_session_columns(bars)


def rth_only(bars: pl.DataFrame) -> pl.DataFrame:
    df = with_session(bars)
    return df.filter(pl.col("is_rth"))


def classify_trading_day(sessioned: pl.DataFrame, *, min_rth_minutes: int = MIN_RTH_MINUTES) -> str:
    """FULL_RTH from MARKET-LEVEL unique RTH minutes and last clock, not per-name density."""
    rth = sessioned.filter(pl.col("is_rth")) if "is_rth" in sessioned.columns else sessioned
    if rth.height == 0 or "time_et" not in rth.columns:
        return "HALF_OR_INCOMPLETE"
    n_market = int(rth["time_et"].n_unique())
    last_m = int(
        rth.select((pl.col("time_et").dt.hour() * 60 + pl.col("time_et").dt.minute()).max()).item()
    )
    want = _hhmm(FULL_DAY_LAST_BAR)
    if n_market < min_rth_minutes or last_m < want:
        return "HALF_OR_INCOMPLETE"
    return "FULL_RTH"


def has_close_window(g: pl.DataFrame) -> bool:
    times = set(g["time_et"].to_list()) if "time_et" in g.columns else set()
    return set(CLOSE_WINDOW_TIMES) <= times


def no_overnight_in_close_window(g: pl.DataFrame) -> bool:
    if g.height == 0 or "session_date" not in g.columns:
        return True
    return g["session_date"].n_unique() == 1
