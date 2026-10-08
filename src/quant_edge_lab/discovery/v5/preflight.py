"""Strict research-calendar preflight. Does not load bar payloads for empirical results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quant_edge_lab.discovery.v5.partitions import (
    PANEL_END,
    PANEL_START,
    SPLIT_ORDER,
    WARMUP_LAST_DAY,
    filter_days,
)
from quant_edge_lab.hashing import sha256_json


class CalendarError(RuntimeError):
    pass


def panel_bounds(man: dict[str, Any]) -> tuple[str, str]:
    data = man.get("data") or {}
    return str(data.get("panel_start") or PANEL_START), str(data.get("panel_end") or PANEL_END)


def warmup_end_day(man: dict[str, Any]) -> str:
    return str((man.get("splits") or {}).get("warmup_last_day") or WARMUP_LAST_DAY)


def warmup_n(man: dict[str, Any]) -> int:
    return int((man.get("eligibility") or {}).get("warmup_trading_days") or 20)


def clip_to_panel(days: list[str], man: dict[str, Any]) -> list[str]:
    start, end = panel_bounds(man)
    return sorted({d for d in days if start <= d <= end})


def assert_in_frozen_panel(day: str, man: dict[str, Any]) -> None:
    start, end = panel_bounds(man)
    if day < start or day > end:
        raise CalendarError(f"refusing to read {day} outside frozen panel {start}..{end}")


def expected_counts(man: dict[str, Any]) -> dict[str, int]:
    s = man["splits"]
    return {k: int(s[k]["n_days"]) for k in SPLIT_ORDER}


def validate_research_calendar(
    days: list[str],
    man: dict[str, Any],
    *,
    parquet_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    start, end = panel_bounds(man)
    outside = [d for d in days if d < start or d > end]
    in_panel = [d for d in days if start <= d <= end]
    if len(in_panel) != len(set(in_panel)):
        raise CalendarError("research calendar must be unique sorted trading days")
    ordered = clip_to_panel(days, man)
    if not ordered:
        raise CalendarError("frozen panel calendar is empty")
    if ordered[0] != start or ordered[-1] != end:
        raise CalendarError(
            f"processed panel must equal {start}..{end}, got {ordered[0]}..{ordered[-1]}"
        )
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
    warm_end = warmup_end_day(man)
    need = warmup_n(man)
    warmup = [d for d in ordered if d <= warm_end]
    if len(warmup) < need:
        raise CalendarError(f"need {need} warmup trading sessions before D1, got {len(warmup)}")
    if not warmup or warmup[0] != start or warmup[-1] != warm_end:
        raise CalendarError(
            f"warmup must run {start}..{warm_end} ({need} sessions); "
            f"got {(warmup[0] if warmup else None)}..{(warmup[-1] if warmup else None)}"
        )
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
        "panel_start": start,
        "panel_end": end,
        "ignored_outside_panel": outside,
    }


def next_session(day: str, calendar: list[str], *, panel_end: str | None = None) -> str | None:
    ordered = sorted(calendar)
    if day not in ordered:
        raise CalendarError(f"{day} is not on the validated research calendar")
    i = ordered.index(day)
    if i + 1 >= len(ordered):
        return None
    nxt = ordered[i + 1]
    if panel_end is not None and nxt > panel_end:
        return None
    return nxt


def require_next_session_file(
    day: str,
    calendar: list[str],
    parquet_exists: Callable[[str], bool],
    *,
    panel_end: str | None = None,
) -> str | None:
    nxt = next_session(day, calendar, panel_end=panel_end)
    if nxt is None:
        return None
    if not parquet_exists(nxt):
        raise CalendarError(f"next session {nxt} after {day} is missing; refusing silent skip")
    return nxt
