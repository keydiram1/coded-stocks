from __future__ import annotations

import polars as pl

from quant_edge_lab.discovery.models import EventPolicy


def apply_event_policy(events: pl.DataFrame, policy: EventPolicy, cooldown_minutes: int | None = None) -> pl.DataFrame:
    if events.height == 0:
        return events
    if policy == "all_clustered":
        return events
    if policy == "first_ticker_day":
        return (
            events.sort(["session_date", "ticker", "decision_ts"])
            .unique(subset=["ticker", "session_date"], keep="first")
        )
    if policy == "cooldown":
        if not cooldown_minutes:
            raise ValueError("cooldown policy requires cooldown_minutes")
        # Explicit: after an event, ignore the same ticker until decision_ts + cooldown.
        rows = []
        last: dict[str, object] = {}
        for row in events.sort(["ticker", "decision_ts"]).iter_rows(named=True):
            t = row["ticker"]
            ts = row["decision_ts"]
            prev = last.get(t)
            if prev is None or (ts - prev).total_seconds() >= cooldown_minutes * 60:
                rows.append(row)
                last[t] = ts
        return pl.DataFrame(rows) if rows else events.head(0)
    raise ValueError(policy)
