"""Minute discovery execution: completed bar → next bar open. SIGNAL_ONLY."""

from __future__ import annotations

EXECUTION_MODEL_VERSION = "next_bar_open_v1"


def next_bar_open_entry(signal_bar_open_ts, bar_minutes: int = 1):
    """Entry timestamp is the open of the bar that begins at decision_ts (bar close)."""
    from datetime import timedelta

    return signal_bar_open_ts + timedelta(minutes=bar_minutes)
