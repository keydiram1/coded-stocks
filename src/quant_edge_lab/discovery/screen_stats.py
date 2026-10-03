"""Cheap SIGNAL_ONLY forward means for screens. Same entry/horizon convention as compute_outcomes."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from quant_edge_lab.outcomes.engine import parse_horizon_minutes


def screen_forward_means(events: pl.DataFrame, bars: pl.DataFrame, horizons: list[str], side: str) -> dict[str, float | None]:
    if events.height == 0 or bars.height == 0:
        return {h: None for h in horizons}
    need = bars.select(["instrument_id", "ts_utc", "open", "close"])
    out: dict[str, float | None] = {}
    for h in horizons:
        mins = parse_horizon_minutes(h)
        rows = []
        by = {iid: g for iid, g in need.partition_by("instrument_id", as_dict=True).items()}
        norm = {}
        for k, v in by.items():
            key = k[0] if isinstance(k, tuple) else k
            norm[key] = v.sort("ts_utc")
        for ev in events.iter_rows(named=True):
            bdf = norm.get(ev["instrument_id"])
            if bdf is None:
                continue
            entry_ts = ev["entry_ts"]
            future = bdf.filter(pl.col("ts_utc") >= entry_ts)
            if future.height == 0:
                continue
            entry = float(future["open"][0])
            tgt = entry_ts + timedelta(minutes=mins - 1)
            match = future.filter(pl.col("ts_utc") == tgt)
            if match.height == 0:
                continue
            close_px = float(match["close"][0])
            fwd = close_px / entry - 1 if side == "long" else 1 - close_px / entry
            rows.append(fwd)
        out[h] = float(sum(rows) / len(rows)) if rows else None
    return out
