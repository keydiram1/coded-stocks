"""Replay observer stub: record what the system knew historically. Not a paper broker."""

from __future__ import annotations

from datetime import datetime

from quant_edge_lab.models.schemas import LiveSetup


def record_replay_setup(
    event_id: str,
    market_ts: datetime,
    last: float | None,
) -> LiveSetup:
    now = market_ts
    return LiveSetup(
        event_id=event_id,
        detected_at=now,
        market_ts=market_ts,
        last=last,
        feed_lag_ms=0.0,
        scanner_latency_ms=0.0,
    )
