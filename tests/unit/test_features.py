from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from quant_edge_lab.features.engine import compute_features
from tests.fixtures.bars import bars_from_et, session_minutes

ET = ZoneInfo("America/New_York")


def test_prev_close_and_gap_are_point_in_time():
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    day0 = session_minutes(d0, (9, 30), 30, open_px=10.0, close_step=0.0, volume=1000)
    # last RTH close remains ~10
    day1 = session_minutes(d1, (9, 30), 20, open_px=12.0, close_step=0.0, volume=1000)
    df = pl.concat([day0, day1])
    feat = compute_features(df)
    row = feat.filter(
        (pl.col("session_date") == d1) & (pl.col("time_et") == datetime(2024, 1, 3, 9, 30).time())
    )
    assert row.height == 1
    assert abs(float(row["prev_close"][0]) - 10.0) < 0.05
    assert abs(float(row["gap_pct"][0]) - 0.2) < 0.02


def test_one_bar_shift_available_at_is_bar_close():
    d0 = date(2024, 1, 2)
    d1 = date(2024, 1, 3)
    df = pl.concat(
        [
            session_minutes(d0, (9, 30), 10, open_px=8.0),
            session_minutes(d1, (9, 30), 10, open_px=8.0),
        ]
    )
    feat = compute_features(df)
    bar_ts = feat["ts_utc"][0]
    avail = feat["available_at"][0]
    assert avail == bar_ts + timedelta(minutes=1)
    assert feat["decision_ts"][0] == avail


def test_feature_uses_completed_bar_not_open():
    """A 09:35 ET bar is not available at 09:35:00."""
    ts = datetime(2024, 1, 3, 9, 35, tzinfo=ET)
    rows = []
    px = 10.0
    for i in range(40):
        t = datetime(2024, 1, 2, 9, 30, tzinfo=ET) + timedelta(minutes=i)
        rows.append((t, px, px, px, px, 1000.0))
    for i in range(20):
        t = datetime(2024, 1, 3, 9, 30, tzinfo=ET) + timedelta(minutes=i)
        rows.append((t, px, px, px, px, 1000.0))
    feat = compute_features(bars_from_et(rows))
    bar = feat.filter(pl.col("ts_et") == ts)
    assert bar.height == 1
    decision = bar["decision_ts"][0]
    # decision is 09:36 ET in UTC naive
    assert decision > bar["ts_utc"][0]
