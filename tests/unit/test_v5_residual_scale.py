from __future__ import annotations

import polars as pl
import pytest

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule
from quant_edge_lab.discovery.v5.campaigns.residual_scale import (
    MIN_PRIOR,
    attach_normalized_dislocation,
    normalized_dislocation,
    prior_residuals_only,
    residual_std_scale,
    update_residual_history,
)
from quant_edge_lab.discovery.v5.continuation import continuation_signed
from quant_edge_lab.discovery.v5.models import CandidateRule


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


def test_denominator_uses_prior_sessions_only():
    hist = [(f"2024-01-{i:02d}", 0.01 + 0.0005 * i) for i in range(1, 22)]
    hist.append(("2024-02-01", 0.99))
    prior = prior_residuals_only(hist, day="2024-02-01")
    assert 0.99 not in prior
    assert residual_std_scale(prior) is not None
    leaked = residual_std_scale(prior + [0.99])
    clean = residual_std_scale(prior)
    assert leaked != clean


def test_no_current_or_future_leakage_in_history_update():
    hist: dict[str, list[tuple[str, float]]] = {}
    scale_before = residual_std_scale(
        prior_residuals_only(hist.get("a", []), day="2024-03-01")
    )
    assert scale_before is None
    update_residual_history(hist, iid="a", day="2024-03-01", residual=0.005)
    still_prior = prior_residuals_only(hist["a"], day="2024-03-01")
    assert still_prior == []
    later = prior_residuals_only(hist["a"], day="2024-03-02")
    assert later == [0.005]


def test_insufficient_or_zero_history_is_null():
    assert residual_std_scale([]) is None
    assert residual_std_scale([0.01] * (MIN_PRIOR - 1)) is None
    assert residual_std_scale([0.0] * MIN_PRIOR) is None
    assert residual_std_scale([0.02] * MIN_PRIOR) is None
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


def test_changing_event_day_residual_does_not_change_denominator():
    hist_days = [f"2024-01-{i:02d}" for i in range(1, 21)]
    hist_vals = [0.001 + 1e-5 * i for i in range(20)]
    base = pl.DataFrame(
        {
            "instrument_id": ["a"] * 20 + ["a"],
            "trading_date": hist_days + ["2024-11-01"],
            "discrepancy": hist_vals + [0.005],
        }
    )
    a = attach_normalized_dislocation(base)
    b = attach_normalized_dislocation(
        base.with_columns(
            pl.when(pl.col("trading_date") == "2024-11-01")
            .then(pl.lit(0.9))
            .otherwise(pl.col("discrepancy"))
            .alias("discrepancy")
        )
    )
    za = a.filter(pl.col("trading_date") == "2024-11-01")
    zb = b.filter(pl.col("trading_date") == "2024-11-01")
    assert za["residual_scale"][0] == zb["residual_scale"][0]
    assert za["normalized_dislocation"][0] != zb["normalized_dislocation"][0]


def test_future_poison_does_not_change_event_day_denominator():
    hist_days = [f"2024-01-{i:02d}" for i in range(1, 21)]
    hist_vals = [0.001 + 1e-5 * i for i in range(20)]
    clean = pl.DataFrame(
        {
            "instrument_id": ["a"] * 21,
            "trading_date": hist_days + ["2024-11-01"],
            "discrepancy": hist_vals + [0.005],
        }
    )
    poisoned = pl.concat(
        [
            clean,
            pl.DataFrame(
                {
                    "instrument_id": ["a"],
                    "trading_date": ["2024-11-02"],
                    "discrepancy": [10.0],
                }
            ),
        ]
    )
    a = attach_normalized_dislocation(clean).filter(pl.col("trading_date") == "2024-11-01")
    b = attach_normalized_dislocation(poisoned).filter(pl.col("trading_date") == "2024-11-01")
    assert a["residual_scale"][0] == b["residual_scale"][0]
    assert a["normalized_dislocation"][0] == b["normalized_dislocation"][0]
