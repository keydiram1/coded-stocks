from __future__ import annotations

import polars as pl
import pytest

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule
from quant_edge_lab.discovery.v5.campaigns.residual_scale import (
    MIN_PRIOR,
    attach_normalized_dislocation,
    normalized_dislocation,
    prior_calendar_sessions,
    residual_std_scale,
    scale_for_event,
)
from quant_edge_lab.discovery.v5.continuation import continuation_signed
from quant_edge_lab.discovery.v5.models import CandidateRule


def _sessions(n: int, start: int = 1) -> list[str]:
    return [f"2024-01-{i:02d}" for i in range(start, start + n)]


def _z_rule(hid: str, *, require_rvol: bool) -> CandidateRule:
    return CandidateRule(
        hypothesis_id=hid,
        mechanism_id="forced_eod_preclose_flow_continuation",
        role="robustness",
        dislocation_abs_min=None,
        z_abs_min=2.0,
        rvol_min=1.3190530517761354 if require_rvol else None,
        require_rvol=require_rvol,
        direction_policy="continuation_no_flip",
        primary_outcome="next_open_to_15m",
        frozen=True,
    )


def test_same_residual_is_more_abnormal_for_low_vol_name():
    residual = 0.005
    low = residual_std_scale([0.001 + 1e-6 * i for i in range(MIN_PRIOR)])
    high = residual_std_scale([0.02 + 1e-4 * i for i in range(MIN_PRIOR)])
    z_low = normalized_dislocation(residual, low)
    z_high = normalized_dislocation(residual, high)
    assert z_low is not None and z_high is not None
    assert abs(z_low) > abs(z_high)


def test_same_residual_is_less_abnormal_for_high_vol_name():
    residual = 0.005
    high = residual_std_scale([0.04 + 0.02 * ((-1) ** i) for i in range(MIN_PRIOR)])
    z = normalized_dislocation(residual, high)
    assert z is not None
    assert abs(z) < 1.0


def test_insufficient_or_zero_history_is_null():
    assert residual_std_scale([0.01] * (MIN_PRIOR - 1)) is None
    assert residual_std_scale([0.0] * MIN_PRIOR) is None
    assert residual_std_scale([0.02] * MIN_PRIOR) is None
    assert residual_std_scale([0.01] * MIN_PRIOR + [0.02]) is None
    assert normalized_dislocation(0.005, None) is None
    assert normalized_dislocation(None, 0.01) is None
    assert normalized_dislocation(0.005, 0.0) is None


def test_continuation_rules_still_use_absolute_discrepancy():
    ev = pl.DataFrame(
        {
            "discrepancy": [0.005, 0.005],
            "close_volume_rvol": [2.0, 2.0],
            "normalized_dislocation": [3.0, 0.1],
        }
    )
    rule = CandidateRule(
        hypothesis_id="H1_PRIMARY",
        mechanism_id="forced_eod_preclose_flow_continuation",
        role="primary",
        dislocation_abs_min=0.0046049372056016155,
        rvol_min=1.3190530517761354,
        require_rvol=True,
        direction_policy="continuation_no_flip",
        primary_outcome="next_open_to_15m",
        frozen=True,
    )
    got = apply_rule(ev, rule)
    assert got.height == 2


def test_example_a_half_percent_on_ten_bp_scale_qualifies_h4_h5():
    assert normalized_dislocation(0.005, 0.001) == pytest.approx(5.0)
    ev = pl.DataFrame(
        {
            "discrepancy": [0.005],
            "close_volume_rvol": [2.0],
            "normalized_dislocation": [5.0],
        }
    )
    assert apply_rule(ev, _z_rule("H4_Z2_RVOL", require_rvol=True)).height == 1
    assert apply_rule(ev, _z_rule("H5_Z2_ONLY", require_rvol=False)).height == 1


def test_example_b_half_percent_on_fifty_bp_scale_does_not_qualify():
    assert normalized_dislocation(0.005, 0.005) == pytest.approx(1.0)
    ev = pl.DataFrame(
        {
            "discrepancy": [0.005],
            "close_volume_rvol": [2.0],
            "normalized_dislocation": [1.0],
        }
    )
    assert apply_rule(ev, _z_rule("H4_Z2_RVOL", require_rvol=True)).height == 0
    assert apply_rule(ev, _z_rule("H5_Z2_ONLY", require_rvol=False)).height == 0


def test_example_c_negative_z_is_short_continuation():
    z = normalized_dislocation(-0.005, 0.001)
    assert z == pytest.approx(-5.0)
    assert continuation_signed(-0.02, -0.005) == pytest.approx(0.02)


def test_positive_z_positive_next_move_is_positive_continuation():
    assert continuation_signed(0.01, 0.005) == pytest.approx(0.01)


def test_exact_previous_20_sessions_present_scale_available():
    cal = _sessions(21)
    t = cal[-1]
    prior = cal[:-1]
    hist = {d: 0.001 + 1e-5 * i for i, d in enumerate(prior)}
    assert prior_calendar_sessions(t, cal) == prior
    assert scale_for_event(hist, t, cal) is not None


def test_missing_one_of_20_does_not_reach_backwards():
    cal = _sessions(21)
    t = cal[-1]
    prior = cal[:-1]
    hist = {d: 0.001 + 1e-5 * i for i, d in enumerate(prior)}
    del hist[prior[3]]
    hist["2023-12-01"] = 0.05
    hist["2023-11-15"] = 0.04
    assert scale_for_event(hist, t, cal) is None


def test_observation_21_sessions_ago_has_no_effect():
    cal = _sessions(22)
    t = cal[-1]
    window = cal[-21:-1]
    hist = {d: 0.001 + 1e-5 * i for i, d in enumerate(window)}
    clean = scale_for_event(hist, t, cal)
    hist[cal[0]] = 9.0
    poisoned = scale_for_event(hist, t, cal)
    assert clean is not None
    assert poisoned == clean


def test_current_day_residual_does_not_enter_denominator():
    cal = _sessions(21)
    t = cal[-1]
    hist = {d: 0.001 + 1e-5 * i for i, d in enumerate(cal[:-1])}
    a = scale_for_event(hist, t, cal)
    hist[t] = 0.9
    b = scale_for_event(hist, t, cal)
    assert a == b


def test_future_residual_does_not_enter_denominator():
    cal = _sessions(22)
    t = cal[-2]
    hist = {d: 0.001 + 1e-5 * i for i, d in enumerate(cal[:-2])}
    a = scale_for_event(hist, t, cal)
    hist[cal[-1]] = 8.0
    b = scale_for_event(hist, t, cal)
    assert a == b
    rows = pl.DataFrame(
        {
            "instrument_id": ["a"] * 22,
            "trading_date": cal,
            "discrepancy": [0.001 + 1e-5 * i for i in range(21)] + [8.0],
        }
    )
    attached = attach_normalized_dislocation(rows, calendar=cal)
    event = attached.filter(pl.col("trading_date") == t)
    assert event["residual_scale"][0] == a


def test_calendar_not_residual_date_sort_defines_window():
    t = "2024-01-22"
    calendar = [f"2024-01-{i:02d}" for i in range(1, 10)] + [
        f"2024-01-{i:02d}" for i in range(11, 23)
    ]
    assert t in calendar
    window = prior_calendar_sessions(t, calendar)
    assert window is not None
    assert "2024-01-10" not in window
    assert "2024-01-11" in window
    hist = {d: 0.001 + 1e-5 * i for i, d in enumerate(window)}
    hist["2024-01-10"] = 0.2
    del hist["2024-01-11"]
    assert scale_for_event(hist, t, calendar) is None
    hist["2024-01-11"] = 0.001
    assert scale_for_event(hist, t, calendar) is not None
