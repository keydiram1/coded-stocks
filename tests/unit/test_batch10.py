from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

import warnings

from quant_edge_lab.discovery.catalog import family_by_id
from quant_edge_lab.discovery.correct_decisions import (
    JOBS_TO_RESCORE,
    correct_variant_direction_decisions,
    effective_stage1_decision,
    stage2_eligible_keys,
)
from quant_edge_lab.discovery.eval_batch10 import BATCH10_PRIMARY, BATCH10_VARIANTS, eval_h002, expand_batch10_jobs
from quant_edge_lab.discovery.frozen_direction import expected_direction_for_job, load_frozen_families
from quant_edge_lab.discovery.funnel import decide_stage
from quant_edge_lab.discovery.outcomes_vector import forward_returns_vectorized
from quant_edge_lab.discovery.models import StageCriteria
from quant_edge_lab.hypotheses.loader import load_hypothesis
from quant_edge_lab.outcomes.engine import compute_outcomes
from quant_edge_lab.universe.filters import add_session_columns
from tests.fixtures.bars import bars_from_et

ET = ZoneInfo("America/New_York")


def _inst(ticker: str = "AAA") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [f"ticker:{ticker}"],
            "ticker": [ticker],
            "exchange": ["NASDAQ"],
            "security_type": ["COMMON_STOCK"],
        }
    )


def _rth_day(ticker: str, open_px: float, window_high: float, window_closes: list[float], break_high: float) -> pl.DataFrame:
    rows = []
    day = datetime(2021, 10, 4, 9, 30, tzinfo=ET)
    iid = f"ticker:{ticker}"
    for i in range(90):
        ts = day + timedelta(minutes=i)
        if i == 0:
            o, h, l, c = open_px, max(open_px, window_high * 0.5 + open_px * 0.5), open_px * 0.999, window_closes[0]
            h = max(h, o, c)
        elif i < 15:
            c = window_closes[i]
            o = window_closes[i - 1]
            h = window_high if i >= 1 else max(o, c)
            l = min(o, c, window_closes[i]) * 0.999
            h = max(h, o, c, window_high if i >= 2 else o)
        elif i == 15:
            o = window_closes[-1]
            c = o
            h = max(o, break_high)
            l = min(o, c) * 0.999
        else:
            o = c = open_px
            h, l = o * 1.001, o * 0.999
        rows.append(
            {
                "instrument_id": iid,
                "ticker": ticker,
                "ts_utc": ts.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
                "open": o,
                "high": max(h, o, c),
                "low": min(l, o, c),
                "close": c,
                "volume": 1000.0,
                "transactions": 10,
                "source": "test",
            }
        )
    return add_session_columns(pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us"))))


def test_frozen_h002_hold_near_high_not_open_buffer():
    text = Path("knowledge/families/batch10_frozen_v1.yaml").read_text(encoding="utf-8")
    assert "min(close in W) >= H_w * 0.997" in text
    assert "near_high: 0.003" in text
    assert "H002.v1" in text
    assert "hold_buffer" not in text
    jobs = expand_batch10_jobs(list(BATCH10_PRIMARY), expand_variants=True)
    assert len(jobs) == 10 + sum(len(v) for v in BATCH10_VARIANTS.values())
    assert family_by_id("H002").executable
    assert family_by_id("H004").stage_criteria.kill_if_sign_disagrees is False
    assert family_by_id("H040").stage_criteria.min_ticker_days == 15


def test_h002_requires_close_near_window_high():
    # Gap well above 15%: open 12 vs prev 10. Drive high 12.4. Every close stays near that high.
    closes = [12.38] * 15
    hold = _rth_day("AAA", 12.0, 12.4, closes, 12.45)
    ev = eval_h002(hold, _inst(), {"ticker:AAA": 10.0}, "e", near=0.003)
    assert ev.height == 1
    # Same open/high, but closes only stay above the open (12.01) — not near 12.4 high.
    fade_closes = [12.01] * 15
    lose = _rth_day("BBB", 12.0, 12.4, fade_closes, 12.45)
    ev2 = eval_h002(lose, _inst("BBB"), {"ticker:BBB": 10.0}, "e", near=0.003)
    assert ev2.height == 0


def test_vectorized_forward_matches_compute_outcomes():
    rows = [
        (datetime(2024, 1, 3, 9, 35, tzinfo=ET), 9.0, 9.1, 8.9, 9.0, 1000),
        (datetime(2024, 1, 3, 9, 36, tzinfo=ET), 10.0, 11.2, 9.9, 11.0, 1000),
        (datetime(2024, 1, 3, 9, 37, tzinfo=ET), 11.0, 11.1, 10.8, 11.0, 1000),
        (datetime(2024, 1, 3, 9, 38, tzinfo=ET), 11.0, 11.2, 10.9, 11.05, 1000),
        (datetime(2024, 1, 3, 9, 40, tzinfo=ET), 11.05, 11.3, 11.0, 11.1, 1000),
        (datetime(2024, 1, 3, 9, 50, tzinfo=ET), 11.1, 11.4, 11.0, 11.2, 1000),
    ]
    bars = bars_from_et(rows)
    events = pl.DataFrame(
        {
            "event_id": ["e1"],
            "instrument_id": ["inst-t"],
            "ticker": ["TEST"],
            "session_date": [datetime(2024, 1, 3).date()],
            "entry_ts": [bars["ts_utc"][1]],
        }
    )
    spec = load_hypothesis(Path("hypotheses/gap_rvol_continuation_v1.yaml"))
    ref = compute_outcomes(events, bars, spec)
    vec = forward_returns_vectorized(events, bars, list(spec.outcomes.forward_returns), spec.side)
    for h in spec.outcomes.forward_returns:
        a = ref.filter(pl.col("horizon") == h)["forward_return"]
        b = vec.filter(pl.col("horizon") == h)["forward_return"]
        if a.len() and a[0] is not None and b.len() and b[0] is not None:
            assert abs(float(a[0]) - float(b[0])) < 1e-12
        else:
            assert (a.len() == 0 or a[0] is None) and (b.len() == 0 or b[0] is None)


def test_vectorized_join_asof_has_no_sortedness_warning():
    rows = [
        (datetime(2024, 1, 3, 9, 36, tzinfo=ET), 10.0, 11.2, 9.9, 11.0, 1000),
        (datetime(2024, 1, 3, 9, 37, tzinfo=ET), 11.0, 11.1, 10.8, 11.0, 1000),
    ]
    bars = bars_from_et(rows)
    events = pl.DataFrame(
        {
            "event_id": ["e1"],
            "instrument_id": ["inst-t"],
            "entry_ts": [bars["ts_utc"][0]],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forward_returns_vectorized(events, bars, ["1m"], "long")
    assert not any("Sortedness" in str(w.message) for w in caught)


def test_every_variant_uses_own_frozen_direction():
    fams = load_frozen_families()
    jobs = expand_batch10_jobs(list(BATCH10_PRIMARY), expand_variants=True)
    assert jobs, "batch10 jobs"
    for fid, vid, _ in jobs:
        got = expected_direction_for_job(fid, vid)
        parent = expected_direction_for_job(fid, "default")
        if vid == "default":
            assert got == parent
            continue
        row = next(r for r in fams[fid]["neighborhood_variants"] if r["id"] == vid)
        if "direction" in row:
            assert got == row["direction"]
            if row["direction"] != parent:
                assert got != parent
        else:
            assert got == expected_direction_for_job(fid, "default")
    assert expected_direction_for_job("H007", "H007.v1") == "long"
    assert expected_direction_for_job("H037", "H037.v1") == "long"
    assert expected_direction_for_job("H040", "H040.v1") == "long"
    assert family_by_id("H007").expected_direction == "short"
    assert family_by_id("H037").expected_direction == "short"
    assert family_by_id("H040").expected_direction == "short"
    c = StageCriteria(min_ticker_days=5, min_trading_days=3, primary_horizon="15m")
    wrong, _ = decide_stage(
        expected_direction="short",
        ticker_days=10,
        trading_days=5,
        mean_primary=0.001,
        top_ticker_share=0.01,
        criteria=c,
        stage_complete=True,
    )
    right, _ = decide_stage(
        expected_direction=expected_direction_for_job("H007", "H007.v1"),
        ticker_days=10,
        trading_days=5,
        mean_primary=0.001,
        top_ticker_share=0.01,
        criteria=c,
        stage_complete=True,
    )
    assert wrong == "KILL" and right == "PASS"


def test_direction_correction_preserves_original(tmp_path: Path):
    from quant_edge_lab.config import Paths
    from quant_edge_lab.discovery.knowledge import atomic_write_json

    Paths(tmp_path).ensure()
    batch = "t-correct"
    jobs = {}
    for key, fid, mean in [
        ("H007.v1", "H007", 9.97e-5),
        ("H037.v1", "H037", 0.00126),
        ("H040.v1", "H040", 0.00426),
    ]:
        jobs[key] = {
            "family_id": fid,
            "variant_id": key,
            "decision": "KILL",
            "decision_reason": f"sign disagrees with expected_direction=short mean_15m={mean}",
            "state": "KILLED_STAGE_1",
            "interim_means": {"15m": mean},
            "ticker_days": 100,
            "trading_days_with_events": 20,
            "concentration_top_ticker_share": 0.01,
            "events": 100,
        }
    atomic_write_json(
        Paths(tmp_path).derived / "discovery" / "batches" / batch / "state.json",
        {"batch_id": batch, "jobs": jobs},
    )
    out = correct_variant_direction_decisions(tmp_path, batch)
    st = out["state"]
    for key in JOBS_TO_RESCORE:
        j = st["jobs"][key]
        assert j["decision"] == "KILL"
        assert j["decision_as_recorded"] == "KILL"
        assert j["corrected_decision"] == "PASS"
        assert "sign disagrees" in j["decision_reason_as_recorded"]
        assert "pre-registered" in j["corrected_decision_reason"]
        assert effective_stage1_decision(j) == "PASS"
    assert set(stage2_eligible_keys(st)) == set(JOBS_TO_RESCORE)


def test_jobs_from_keys_roundtrip():
    from quant_edge_lab.discovery.eval_batch10 import jobs_from_keys

    js = jobs_from_keys(["H001.v1", "H007.v1", "H040"])
    assert [x[1] for x in js] == ["H001.v1", "H007.v1", "default"]
    assert js[0][2]["gap"] == 0.20
