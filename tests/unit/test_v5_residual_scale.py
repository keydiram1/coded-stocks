from __future__ import annotations

import polars as pl

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule
from quant_edge_lab.discovery.v5.campaigns.residual_scale import (
    MIN_PRIOR,
    normalized_dislocation,
    prior_residuals_only,
    residual_std_scale,
    update_residual_history,
)
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
