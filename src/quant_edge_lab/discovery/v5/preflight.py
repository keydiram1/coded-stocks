"""Strict research-calendar preflight. Does not load bar payloads for empirical results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quant_edge_lab.discovery.v5.partitions import SPLIT_ORDER, WARMUP_LAST_DAY, filter_days
from quant_edge_lab.hashing import sha256_json


class CalendarError(RuntimeError):
    pass


def expected_counts(man: dict[str, Any]) -> dict[str, int]:
    s = man["splits"]
    return {k: int(s[k]["n_days"]) for k in SPLIT_ORDER}


def validate_research_calendar(
    days: list[str],
    man: dict[str, Any],
    *,
    parquet_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    ordered = sorted(set(days))
    if ordered != sorted(days):
        raise CalendarError("research calendar must be unique sorted trading days")
    counts = expected_counts(man)
    for split in SPLIT_ORDER:
        got = filter_days(ordered, split, man)
        n = counts[split]
        if len(got) != n:
            raise CalendarError(f"{split} expected {n} trading days, got {len(got)}")
        a, b = man["splits"][split]["start"], man["splits"][split]["end"]
        if not got or got[0] != a or got[-1] != b:
            raise CalendarError(
                f"{split} bounds drifted: got {got[:1]}..{got[-1:] if got else []} vs {a}..{b}"
            )
    d1_start = man["splits"]["D1"]["start"]
    warmup = [d for d in ordered if d <= WARMUP_LAST_DAY]
    if not warmup or warmup[-1] != WARMUP_LAST_DAY:
        # warmup_last_day must be present if it is a trading day in the panel design
        before = [d for d in ordered if d < d1_start]
        if len(before) < int(man.get("eligibility", {}).get("warmup_trading_days") or 20):
            raise CalendarError("insufficient warmup trading days before D1")
    if parquet_exists is not None:
        missing = [d for d in ordered if not parquet_exists(d)]
        if missing:
            raise CalendarError(f"missing normalized parquet for trading days e.g. {missing[:5]}")
    return {
        "n_days": len(ordered),
        "counts": {k: len(filter_days(ordered, k, man)) for k in SPLIT_ORDER},
        "calendar_hash": sha256_json(ordered),
        "first": ordered[0],
        "last": ordered[-1],
    }


def next_session(day: str, calendar: list[str]) -> str | None:
    ordered = sorted(calendar)
    if day not in ordered:
        raise CalendarError(f"{day} is not on the validated research calendar")
    i = ordered.index(day)
    if i + 1 >= len(ordered):
        return None
    return ordered[i + 1]


def require_next_session_file(
    day: str, calendar: list[str], parquet_exists: Callable[[str], bool]
) -> str | None:
    nxt = next_session(day, calendar)
    if nxt is None:
        return None
    if not parquet_exists(nxt):
        raise CalendarError(f"next session {nxt} after {day} is missing; refusing silent skip")
    return nxt
