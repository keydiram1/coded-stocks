from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from quant_edge_lab.features.engine import compute_features
from quant_edge_lab.validation.leakage import (
    assert_available,
    features_at_decision,
    future_poison,
    pit_float_join,
    truncate_at_event,
)
from tests.fixtures.bars import session_minutes

ET = ZoneInfo("America/New_York")


def _two_days():
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    return pl.concat(
        [
            session_minutes(d0, (9, 30), 40, open_px=10.0, volume=1000),
            session_minutes(d1, (9, 30), 40, open_px=10.5, volume=1000),
        ]
    )


def test_truncate_at_event_features_stable():
    bars = _two_days()
    feat = compute_features(bars)
    target = feat.filter(pl.col("session_date") == date(2024, 1, 3)).head(1)
    decision = target["decision_ts"][0]
    iid = target["instrument_id"][0]
    gap = float(target["gap_pct"][0])
    truncated = truncate_at_event(bars, decision)
    feat2 = compute_features(truncated)
    row = feat2.filter((pl.col("instrument_id") == iid) & (pl.col("decision_ts") == decision))
    assert row.height == 1
    assert abs(float(row["gap_pct"][0]) - gap) < 1e-12


def test_future_poison_does_not_change_past_features():
    bars = _two_days()
    feat = compute_features(bars)
    target = feat.filter(pl.col("session_date") == date(2024, 1, 3)).head(1)
    decision = target["decision_ts"][0]
    iid = target["instrument_id"][0]
    before = features_at_decision(bars, iid, decision)
    poisoned = future_poison(bars, after_ts=decision, factor=50.0)
    after = features_at_decision(poisoned, iid, decision)
    for key in ("gap_pct", "prev_close", "ret_1m"):
        if before[key] is None:
            continue
        assert after[key] == pytest.approx(before[key], rel=1e-9, abs=1e-12)


def test_availability_assertion():
    ts = datetime(2024, 1, 3, 14, 36, tzinfo=UTC)
    assert_available(ts, ts)
    with pytest.raises(AssertionError):
        assert_available(ts + timedelta(seconds=1), ts)


def test_current_metadata_ban_float_asof():
    events = pl.DataFrame(
        {
            "instrument_id": ["inst-t"],
            "decision_ts": [datetime(2024, 1, 3, 14, 36)],
        }
    ).with_columns(pl.col("decision_ts").cast(pl.Datetime("us")))
    floats = pl.DataFrame(
        {
            "instrument_id": ["inst-t", "inst-t"],
            "effective_at": [datetime(2023, 6, 1), datetime(2026, 1, 1)],
            "observed_at": [datetime(2023, 6, 2), datetime(2026, 1, 2)],
            "float_shares": [1_000_000.0, 9_999_999.0],
        }
    ).with_columns(
        pl.col("effective_at").cast(pl.Datetime("us")),
        pl.col("observed_at").cast(pl.Datetime("us")),
    )
    joined = pit_float_join(events, floats)
    assert joined["float_shares"][0] == 1_000_000.0
    assert joined["float_shares"][0] != 9_999_999.0
