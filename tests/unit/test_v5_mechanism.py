from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quant_edge_lab.discovery.v5.campaigns.close_features import (
    close_volume_rvol,
    discrepancy,
    expected_final_5m,
    loo_cross_section,
    resolution_direction,
    signed_resolution,
)
from quant_edge_lab.discovery.v5.causality import assert_available_le_decision
from quant_edge_lab.discovery.v5.partitions import assert_sealed_oos_closed, open_sealed_oos
from quant_edge_lab.features.v4r.beta import BETA_WINDOW, beta_from_history, ols_beta


def test_discrepancy_is_observed_minus_expected():
    assert discrepancy(0.02, 0.005) == pytest.approx(0.015)
    assert discrepancy(-0.01, 0.01) == pytest.approx(-0.02)
    assert discrepancy(None, 0.1) is None


def test_positive_discrepancy_implies_short():
    assert resolution_direction(0.01) == "SHORT"
    assert signed_resolution(0.02, "SHORT") == pytest.approx(-0.02)


def test_negative_discrepancy_implies_long():
    assert resolution_direction(-0.01) == "LONG"
    assert signed_resolution(0.02, "LONG") == pytest.approx(0.02)


def test_expected_is_beta_times_loo_market():
    assert expected_final_5m(1.5, 0.01) == pytest.approx(0.015)
    assert expected_final_5m(None, 0.01) is None


def test_no_beta_fallback_to_one():
    hist = []
    assert beta_from_history(hist, ["a"]) == {}
    assert 1.0 not in beta_from_history(hist, ["a"]).values()


def test_ols_insufficient_is_nan_not_one():
    assert np.isnan(ols_beta(np.array([0.1]), np.array([0.1])))


def test_loo_market_excludes_self():
    r = np.array([0.10, 0.00, -0.10])
    m = loo_cross_section(r)
    assert m[1] == pytest.approx(0.0)
    assert m[0] == pytest.approx(-0.05)


def test_rvol_requires_twenty_prior_observations():
    assert close_volume_rvol(200.0, [100.0] * 19) is None
    assert close_volume_rvol(200.0, [100.0] * 20) == pytest.approx(2.0)
    assert close_volume_rvol(200.0, []) is None


def test_pit_available_must_not_exceed_decision():
    ts = datetime(2024, 6, 3, 20, 0)
    assert_available_le_decision(ts, ts)
    with pytest.raises(AssertionError):
        assert_available_le_decision(datetime(2024, 6, 3, 20, 1), ts)


def test_sealed_oos_stays_closed():
    assert_sealed_oos_closed(
        {"sealed_oos": "inaccessible", "splits": {"sealed_oos": "closed_do_not_open"}}
    )
    with pytest.raises(RuntimeError):
        open_sealed_oos()
    with pytest.raises(AssertionError):
        assert_sealed_oos_closed({"sealed_oos": "inaccessible", "splits": {"OOS": {"start": "x"}}})


def test_beta_window_constant_is_twenty():
    assert BETA_WINDOW == 20


def _mini_panel(*, n_warm: int) -> tuple[dict, list[str]]:
    from datetime import date, timedelta

    d = date(2021, 10, 1)
    days: list[str] = []
    while len(days) < n_warm + 3 + 2 + 2:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    warm = days[:n_warm]
    d1 = days[n_warm : n_warm + 3]
    d2 = days[n_warm + 3 : n_warm + 5]
    d3 = days[n_warm + 5 : n_warm + 7]
    man = {
        "data": {"panel_start": days[0], "panel_end": days[-1]},
        "splits": {
            "warmup_last_day": warm[-1],
            "D1": {"start": d1[0], "end": d1[-1], "n_days": 3},
            "D2": {"start": d2[0], "end": d2[-1], "n_days": 2},
            "D3": {"start": d3[0], "end": d3[-1], "n_days": 2},
        },
        "eligibility": {"warmup_trading_days": 20},
    }
    return man, days


def test_preflight_complete_calendar_passes_and_gap_refuses():
    from quant_edge_lab.discovery.v5.preflight import (
        CalendarError,
        next_session,
        validate_research_calendar,
    )

    man, days = _mini_panel(n_warm=20)
    rec = validate_research_calendar(days, man, parquet_exists=lambda d: True)
    assert rec["counts"]["D1"] == 3
    assert rec["first"] == man["data"]["panel_start"]
    assert rec["last"] == man["data"]["panel_end"]
    d1_end = man["splits"]["D1"]["end"]
    assert next_session(d1_end, days, panel_end=man["data"]["panel_end"]) == man["splits"]["D2"][
        "start"
    ]
    gap = [x for x in days if x != man["splits"]["D2"]["start"]]
    try:
        validate_research_calendar(gap, man, parquet_exists=lambda d: True)
        raise AssertionError("missing middle day must refuse")
    except CalendarError:
        pass


def test_preflight_warmup_count_and_panel_end_guard():
    from quant_edge_lab.discovery.v5.preflight import (
        CalendarError,
        clip_to_panel,
        next_session,
        validate_research_calendar,
    )

    man20, days20 = _mini_panel(n_warm=20)
    validate_research_calendar(days20, man20, parquet_exists=lambda d: True)
    man19, days19 = _mini_panel(n_warm=19)
    try:
        validate_research_calendar(days19, man19, parquet_exists=lambda d: True)
        raise AssertionError("19 warmup days must refuse")
    except CalendarError:
        pass
    frozen = {
        "data": {"panel_start": "2021-10-01", "panel_end": "2026-10-01"},
        "splits": {"warmup_last_day": "2021-10-28"},
        "eligibility": {"warmup_trading_days": 20},
    }
    extra = ["2026-09-30", "2026-10-01", "2026-10-02"]
    clipped = clip_to_panel(extra, frozen)
    assert clipped == ["2026-09-30", "2026-10-01"]
    assert next_session("2026-10-01", clipped, panel_end="2026-10-01") is None
