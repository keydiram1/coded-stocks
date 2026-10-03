"""Deterministic stage day panels. Never chosen from strategy performance."""

from __future__ import annotations

import hashlib
from typing import Literal

STAGE_PANEL_VERSION = "stage_panels_v1"

StageName = Literal["fast", "broad", "full"]


def _stable_index(day: str, salt: str) -> int:
    h = hashlib.sha256(f"{STAGE_PANEL_VERSION}:{salt}:{day}".encode()).hexdigest()
    return int(h[:8], 16)


def select_stage_days(all_days: list[str], stage: StageName, *, n_fast: int = 50, n_broad: int = 200) -> list[str]:
    """Spread days across the calendar via a versioned hash, not performance."""
    days = sorted(all_days)
    if stage == "full":
        return days
    n = n_fast if stage == "fast" else n_broad
    n = min(n, len(days))
    ranked = sorted(days, key=lambda d: (_stable_index(d, stage) % 10_007, d))
    picked = sorted(ranked[:n])
    return picked


def select_falsify_days(all_days: list[str], *, n: int = 100) -> list[str]:
    """Hashed days disjoint from fast+broad panels. Salt frozen as 'falsify'."""
    fast = set(select_stage_days(all_days, "fast"))
    broad = set(select_stage_days(all_days, "broad"))
    rest = [d for d in sorted(all_days) if d not in fast and d not in broad]
    n = min(n, len(rest))
    ranked = sorted(rest, key=lambda d: (_stable_index(d, "falsify") % 10_007, d))
    return sorted(ranked[:n])
