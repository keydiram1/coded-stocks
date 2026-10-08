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


def test_rvol_uses_prior_days_only():
    assert close_volume_rvol(200.0, [100.0, 100.0]) == pytest.approx(2.0)
    assert close_volume_rvol(200.0, []) is None


def test_pit_available_must_not_exceed_decision():
    ts = datetime(2024, 6, 3, 20, 0)
    assert_available_le_decision(ts, ts)
    with pytest.raises(AssertionError):
        assert_available_le_decision(datetime(2024, 6, 3, 20, 1), ts)


def test_sealed_oos_stays_closed():
    assert_sealed_oos_closed({"sealed_oos": "inaccessible", "splits": {"sealed_oos": "closed_do_not_open"}})
    with pytest.raises(RuntimeError):
        open_sealed_oos()
    with pytest.raises(AssertionError):
        assert_sealed_oos_closed({"sealed_oos": "inaccessible", "splits": {"OOS": {"start": "x"}}})


def test_beta_window_constant_is_twenty():
    assert BETA_WINDOW == 20
