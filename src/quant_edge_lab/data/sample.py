"""Deterministic synthetic 1-minute bars for engine tests — not research evidence."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.store import (
    sample_actions_path,
    sample_bar_path,
    sample_instrument_path,
    write_parquet,
)

ET = ZoneInfo("America/New_York")
FEATURE_VERSION = "features_v1"


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _minute_index(session: date, start: time, end: time) -> list[datetime]:
    cur = datetime.combine(session, start, tzinfo=ET)
    stop = datetime.combine(session, end, tzinfo=ET)
    stamps: list[datetime] = []
    while cur < stop:
        stamps.append(cur)
        cur += timedelta(minutes=1)
    return stamps


def _ohlc(open_px: float, close_px: float, wick: float = 0.002) -> tuple[float, float, float, float]:
    high = max(open_px, close_px) * (1 + abs(wick))
    low = min(open_px, close_px) * (1 - abs(wick))
    high = max(high, open_px, close_px)
    low = min(low, open_px, close_px)
    return open_px, high, low, close_px


def generate_sample_market(seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    days = _weekdays(date(2024, 1, 2), 8)
    instruments = [
        {
            "instrument_id": "inst-gapa",
            "ticker": "GAPA",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "base": 8.0,
        },
        {
            "instrument_id": "inst-gapb",
            "ticker": "GAPB",
            "exchange": "NYSE",
            "security_type": "COMMON_STOCK",
            "base": 12.0,
        },
        {
            "instrument_id": "inst-fade",
            "ticker": "FADE",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "base": 6.5,
        },
        {
            "instrument_id": "inst-pmhi",
            "ticker": "PMHI",
            "exchange": "NYSE_AMERICAN",
            "security_type": "COMMON_STOCK",
            "base": 4.5,
        },
        {
            "instrument_id": "inst-ctrl",
            "ticker": "CTRL",
            "exchange": "NASDAQ",
            "security_type": "COMMON_STOCK",
            "base": 9.0,
        },
    ]

    rows: list[dict] = []
    event_day = days[5]  # enough history for rolling features
    fade_day = days[6]
    pm_day = days[5]
    amb_day = days[4]

    for inst in instruments:
        px = float(inst["base"])
        for d in days:
            stamps = _minute_index(d, time(8, 0), time(11, 30))
            rth_open_i = next(i for i, t in enumerate(stamps) if t.timetz().replace(tzinfo=None) >= time(9, 30))
            # default quiet session
            gap = 0.0
            vol_mult = 1.0
            path = "quiet"
            if inst["ticker"] == "GAPA" and d == event_day:
                gap, vol_mult, path = 0.22, 12.0, "gap_cont"
            elif inst["ticker"] == "GAPB" and d == event_day:
                gap, vol_mult, path = 0.18, 9.0, "gap_cont"
            elif inst["ticker"] == "FADE" and d == fade_day:
                gap, vol_mult, path = 0.04, 4.0, "extension_fade"
            elif inst["ticker"] == "PMHI" and d == pm_day:
                gap, vol_mult, path = 0.08, 3.0, "pm_high"
            elif inst["ticker"] == "GAPA" and d == amb_day:
                gap, vol_mult, path = 0.16, 8.0, "ambiguous"

            session_open = px * (1 + gap)
            last = session_open if stamps[0].time() >= time(9, 30) else px * (1 + gap * 0.4)

            for i, ts in enumerate(stamps):
                t = ts.time()
                is_rth = t >= time(9, 30)
                is_pm = t < time(9, 30)
                o = last
                noise = float(rng.normal(0, 0.0008))
                ret = noise
                volume = float(rng.integers(800, 1800))

                if path == "quiet":
                    ret += 0.00005
                elif path == "gap_cont":
                    if is_pm:
                        last = px * (1 + gap * 0.5)  # keep pm below the gap open
                        o = last
                        ret = float(rng.normal(0, 0.0005))
                        volume = float(rng.integers(400, 900))
                    elif t < time(10, 15):
                        ret = 0.0012 + float(rng.normal(0, 0.0004))
                        volume = float(rng.integers(6000, 14000)) * (vol_mult / 10)
                    else:
                        ret = float(rng.normal(0, 0.0006))
                elif path == "extension_fade":
                    if is_rth and t < time(9, 50):
                        ret = 0.004 + float(rng.normal(0, 0.0005))
                        volume = float(rng.integers(8000, 16000))
                    elif is_rth and t < time(10, 30):
                        ret = -0.0025 + float(rng.normal(0, 0.0005))
                        volume = float(rng.integers(5000, 10000))
                elif path == "pm_high":
                    if is_pm and time(8, 40) <= t <= time(9, 10):
                        ret = 0.0015 + float(rng.normal(0, 0.0004))
                        volume = float(rng.integers(2000, 4000))
                    elif is_rth and t < time(10, 0):
                        ret = 0.0008 + float(rng.normal(0, 0.0004))
                        volume = float(rng.integers(2500, 5000))
                elif path == "ambiguous" and is_rth and t < time(10, 0):
                    volume = float(rng.integers(7000, 12000))
                    ret = 0.0004

                if i == rth_open_i:
                    o = session_open
                    last = session_open

                c = max(0.5, o * (1 + ret))
                wick = 0.0015
                o, h, l, c = _ohlc(o, c, wick)

                if path == "ambiguous" and is_rth and t == time(9, 36):
                    # both +3% and -2% barriers reachable in one minute from this open
                    h = o * 1.04
                    l = o * 0.97
                    c = o * 1.001

                vwap = (h + l + c) / 3.0
                rows.append(
                    {
                        "instrument_id": inst["instrument_id"],
                        "ticker": inst["ticker"],
                        "ts_utc": ts.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": volume,
                        "vwap": vwap,
                        "transactions": int(max(1, volume / 50)),
                        "source": "sample",
                    }
                )
                last = c
            px = last

    bars = pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
    inst_df = pl.DataFrame(
        [
            {
                "instrument_id": x["instrument_id"],
                "ticker": x["ticker"],
                "effective_from": datetime(2020, 1, 1),
                "effective_to": None,
                "exchange": x["exchange"],
                "security_type": x["security_type"],
                "cik": None,
                "active": True,
            }
            for x in instruments
        ]
    )
    # Explicit corporate-action fixture (split) — not applied silently to raw bars.
    actions = pl.DataFrame(
        [
            {
                "instrument_id": "inst-ctrl",
                "type": "SPLIT",
                "effective_date": datetime(2024, 6, 1),
                "ratio": 2.0,
                "source": "sample_fixture",
            }
        ]
    )
    return bars, inst_df, actions


def write_sample_data(root: Path | None = None, seed: int = 42) -> dict[str, str]:
    paths = Paths(root)
    paths.ensure()
    bars, inst, actions = generate_sample_market(seed=seed)
    b = write_parquet(bars, sample_bar_path(paths.root))
    i = write_parquet(inst, sample_instrument_path(paths.root))
    a = write_parquet(actions, sample_actions_path(paths.root))
    return {"bars": str(b), "instruments": str(i), "corporate_actions": str(a)}
