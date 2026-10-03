from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from quant_edge_lab.models.hypothesis import HypothesisSpec
from quant_edge_lab.outcomes.engine import compute_outcomes
from tests.fixtures.bars import bars_from_et

ET = ZoneInfo("America/New_York")


def _spec() -> HypothesisSpec:
    from quant_edge_lab.hypotheses.loader import load_hypothesis
    from pathlib import Path

    return load_hypothesis(Path("hypotheses/gap_rvol_continuation_v1.yaml"))


def test_forward_return_next_bar_open_entry():
    # entry at 09:36 ET open=10, 1m close of that bar=11 → +10%
    rows = [
        (datetime(2024, 1, 3, 9, 35, tzinfo=ET), 9.0, 9.1, 8.9, 9.0, 1000),
        (datetime(2024, 1, 3, 9, 36, tzinfo=ET), 10.0, 11.2, 9.9, 11.0, 1000),
        (datetime(2024, 1, 3, 9, 37, tzinfo=ET), 11.0, 11.1, 10.8, 11.0, 1000),
    ]
    bars = bars_from_et(rows)
    events = pl.DataFrame(
        {
            "event_id": ["e1"],
            "instrument_id": ["inst-t"],
            "ticker": ["TEST"],
            "session_date": [datetime(2024, 1, 3).date()],
            "entry_ts": [bars["ts_utc"][1]],
            "side": ["long"],
        }
    )
    spec = _spec()
    out = compute_outcomes(events, bars, spec)
    one = out.filter(pl.col("horizon") == "1m")
    assert abs(float(one["forward_return"][0]) - 0.10) < 1e-9
    assert one["execution_status"][0] == "SIGNAL_ONLY"


def test_barrier_ambiguity_same_minute():
    rows = [
        (datetime(2024, 1, 3, 9, 36, tzinfo=ET), 10.0, 10.4, 9.7, 10.01, 1000),
        (datetime(2024, 1, 3, 9, 37, tzinfo=ET), 10.01, 10.02, 10.0, 10.01, 1000),
    ]
    bars = bars_from_et(rows)
    events = pl.DataFrame(
        {
            "event_id": ["e2"],
            "instrument_id": ["inst-t"],
            "ticker": ["TEST"],
            "session_date": [datetime(2024, 1, 3).date()],
            "entry_ts": [bars["ts_utc"][0]],
            "side": ["long"],
        }
    )
    spec = _spec()
    out = compute_outcomes(events, bars, spec)
    assert bool(out.filter(pl.col("horizon") == "1m")["ambiguous"][0]) is True
    assert out.filter(pl.col("horizon") == "1m")["target_first"][0] is None


def test_mfe_mae_long():
    rows = [
        (datetime(2024, 1, 3, 9, 36, tzinfo=ET), 10.0, 10.2, 9.9, 10.05, 1000),
        (datetime(2024, 1, 3, 9, 37, tzinfo=ET), 10.05, 10.5, 10.0, 10.1, 1000),
    ]
    bars = bars_from_et(rows)
    events = pl.DataFrame(
        {
            "event_id": ["e3"],
            "instrument_id": ["inst-t"],
            "ticker": ["TEST"],
            "session_date": [datetime(2024, 1, 3).date()],
            "entry_ts": [bars["ts_utc"][0]],
            "side": ["long"],
        }
    )
    spec = _spec()
    out = compute_outcomes(events, bars, spec)
    five = out.filter(pl.col("horizon") == "5m")
    assert float(five["mfe"][0]) >= 0.049  # 10.5/10 - 1
    assert float(five["mae"][0]) <= -0.009
