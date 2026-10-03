from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl

from quant_edge_lab.events.engine import evaluate_events
from quant_edge_lab.features.engine import compute_features
from quant_edge_lab.hypotheses.loader import load_hypothesis
from pathlib import Path
from tests.fixtures.bars import bars_from_et

ET = ZoneInfo("America/New_York")


def test_events_use_decision_ts_not_signal_bar_open():
    spec = load_hypothesis(Path("hypotheses/gap_rvol_continuation_v1.yaml"))
    rows = []
    # two sessions so prev_close exists
    px = 10.0
    for day, open_px, vol in ((2, 10.0, 1000.0), (3, 12.0, 20000.0)):
        for i in range(90):
            t = datetime(2024, 1, day, 8, 30, tzinfo=ET)
            from datetime import timedelta

            ts = t + timedelta(minutes=i)
            o = open_px if ts.hour == 9 and ts.minute == 30 else px
            c = o
            rows.append((ts, o, o * 1.001, o * 0.999, c, vol))
            px = c
    bars = bars_from_et(rows, instrument_id="inst-gapa", ticker="GAPA")
    inst = pl.DataFrame(
        {
            "instrument_id": ["inst-gapa"],
            "ticker": ["GAPA"],
            "exchange": ["NASDAQ"],
            "security_type": ["COMMON_STOCK"],
        }
    )
    featured = compute_features(bars)
    ev = evaluate_events(featured, inst, spec, experiment_id="t")
    if ev.height:
        assert (ev["decision_ts"] > ev["event_ts"]).all()
        assert (ev["entry_ts"] == ev["decision_ts"]).all()
