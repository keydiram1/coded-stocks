from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

ET = ZoneInfo("America/New_York")


def bars_from_et(
    rows: list[tuple[datetime, float, float, float, float, float]],
    instrument_id: str = "inst-t",
    ticker: str = "TEST",
) -> pl.DataFrame:
    payload = []
    for ts_et, o, h, l, c, v in rows:
        utc = ts_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        payload.append(
            {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "ts_utc": utc,
                "open": o,
                "high": max(h, o, c),
                "low": min(l, o, c),
                "close": c,
                "volume": v,
                "vwap": (max(h, o, c) + min(l, o, c) + c) / 3,
                "transactions": 10,
                "source": "fixture",
            }
        )
    return pl.DataFrame(payload).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))


def session_minutes(
    day,
    start_hm: tuple[int, int],
    n: int,
    *,
    open_px: float,
    close_step: float = 0.0,
    volume: float = 1000,
    instrument_id: str = "inst-t",
    ticker: str = "TEST",
) -> pl.DataFrame:
    h, m = start_hm
    rows = []
    px = open_px
    for i in range(n):
        ts = datetime(day.year, day.month, day.day, h, m, tzinfo=ET) + timedelta(minutes=i)
        o = px
        c = px * (1 + close_step)
        rows.append((ts, o, max(o, c) * 1.001, min(o, c) * 0.999, c, volume))
        px = c
    return bars_from_et(rows, instrument_id=instrument_id, ticker=ticker)
