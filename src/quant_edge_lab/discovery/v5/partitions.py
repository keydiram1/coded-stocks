"""Canonical V5 chronological partitions. Sealed OOS stays closed."""

from __future__ import annotations

from datetime import date
from typing import Any

SEALED_OOS_STATE = "inaccessible"
PANEL_START = "2021-10-01"
PANEL_END = "2026-10-01"
WARMUP_LAST_DAY = "2021-10-28"
D1 = ("2021-10-29", "2023-10-17")
D2 = ("2023-10-18", "2024-10-10")
D3 = ("2024-10-11", "2026-10-01")
SPLIT_ORDER = ("D1", "D2", "D3")


def split_bounds(manifest: dict[str, Any] | None = None) -> dict[str, tuple[str, str]]:
    if not manifest:
        return {"D1": D1, "D2": D2, "D3": D3}
    s = manifest["splits"]
    return {k: (s[k]["start"], s[k]["end"]) for k in SPLIT_ORDER}


def in_split(day: str, split: str, manifest: dict[str, Any] | None = None) -> bool:
    a, b = split_bounds(manifest)[split]
    return a <= day <= b


def split_for_day(day: str, manifest: dict[str, Any] | None = None) -> str | None:
    if day <= WARMUP_LAST_DAY:
        return "WARMUP"
    for name in SPLIT_ORDER:
        if in_split(day, name, manifest):
            return name
    return None


def filter_days(days: list[str], split: str, manifest: dict[str, Any] | None = None) -> list[str]:
    return [d for d in days if in_split(d, split, manifest)]


def assert_no_future_split_in_estimation(
    estimation_split: str, used_days: list[str], manifest: dict[str, Any] | None = None
) -> None:
    rank = {n: i for i, n in enumerate(SPLIT_ORDER)}
    cap = rank[estimation_split]
    for d in used_days:
        sp = split_for_day(d, manifest)
        if sp in rank and rank[sp] > cap:
            raise AssertionError(f"D1-or-earlier estimation used later split day {d} ({sp})")


def assert_sealed_oos_closed(manifest: dict[str, Any] | None = None) -> None:
    state = (manifest or {}).get("sealed_oos", SEALED_OOS_STATE)
    if state not in {SEALED_OOS_STATE, "closed", "closed_do_not_open"}:
        raise AssertionError(f"sealed OOS must remain closed, got {state}")
    splits = (manifest or {}).get("splits") or {}
    if (
        splits.get("OOS")
        or splits.get("D4")
        or splits.get("sealed_oos")
        not in {
            None,
            "closed_do_not_open",
            SEALED_OOS_STATE,
            "closed",
        }
    ):
        if "OOS" in splits or "D4" in splits:
            raise AssertionError("do not invent a historical fourth/sealed OOS split")


def open_sealed_oos(*_a: Any, **_k: Any) -> None:
    raise RuntimeError("sealed OOS remains unopened")


def parse_day(day: str) -> date:
    y, m, d = (int(x) for x in day.split("-"))
    return date(y, m, d)
