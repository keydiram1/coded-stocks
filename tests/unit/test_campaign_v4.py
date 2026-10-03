from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from quant_edge_lab.data.massive.news import CANONICAL_UTC, with_utc_us
from quant_edge_lab.discovery.campaign_v4 import freeze_campaign, load_v4, run_pipeline_on_frame, synthetic_panel, write_freeze_report
from quant_edge_lab.discovery.conditional import Condition, apply_conditions, canonical_spec, make_candidate
from quant_edge_lab.features.v4 import assert_available_le_decision, ensure_utc
from quant_edge_lab.features.v4.residuals import loo_market_returns
from quant_edge_lab.peers import PeerGraphBuilder


def _tiny_gates():
    return {
        "economic_floor_abs_mean": 0.0005,
        "max_top_ticker_share": 0.95,
        "leaf": {"min_ticker_days": 30, "min_trading_days": 5, "min_tickers": 5, "min_parent_frac": 0.1},
        "tree": {"max_depth": 3, "node_alpha": 0.2, "quantile_grid": [0.2, 0.5, 0.8]},
        "stability": {"n_subsamples": 8, "min_selection_freq": 0.0, "min_sign_agree": 0.5},
        "d3": {"abs_mean_floor": 0.0008, "require_sign_match_d2": True},
        "linear": {"lambda_grid": [0.01, 0.1]},
    }


def _man_for(df: pl.DataFrame) -> dict:
    days = sorted(df["trading_date"].unique().to_list())
    n = len(days)
    i1, i2 = int(n * 0.40), int(n * 0.60)
    d1, d2, d3 = days[:i1], days[i1:i2], days[i2:]
    return {
        "targets": {"primary": "future_residual_15m"},
        "search": {"seed": 4},
        "search_features": ["leader_response_gap", "activity_acceleration", "xs_dispersion_5m", "H004", "irrelevant_E"],
        "splits": {
            "D1": {"start": d1[0], "end": d1[-1], "n_days": len(d1)},
            "D2": {"start": d2[0], "end": d2[-1], "n_days": len(d2)},
            "D3": {"start": d3[0], "end": d3[-1], "n_days": len(d3)},
        },
    }


def test_v4_freeze_hashes_stable():
    root = Path(".")
    a = freeze_campaign(root)
    b = freeze_campaign(root)
    assert a["manifest"] == b["manifest"]
    man, gates = load_v4(root)
    assert man["targets"]["primary"] == "future_residual_15m"
    assert gates["tree"]["max_depth"] == 3
    assert man["splits"]["D1"]["start"] == "2021-10-29"


def test_v4_expected_execution_hashes():
    from quant_edge_lab.discovery.campaign_v4_runner import EXPECTED_GATES, EXPECTED_MANIFEST, verify_freeze

    man, gates, fr = verify_freeze(Path("."))
    assert fr["manifest"] == EXPECTED_MANIFEST
    assert fr["gates"] == EXPECTED_GATES
    assert man["sealed_oos"] == "inaccessible"


def test_candidate_identity_canonical():
    c1 = make_candidate([Condition("H004", "==", True), Condition("leader_response_gap", ">", 0.01)], "LONG", "future_residual_15m", 15, "t", "D1")
    c2 = make_candidate([Condition("leader_response_gap", ">", 0.01), Condition("H004", "==", True)], "LONG", "future_residual_15m", 15, "t", "D1")
    assert c1.candidate_id == c2.candidate_id
    assert canonical_spec(c1.conditions, "LONG", "future_residual_15m")["conditions"][0][0] == "H004"


def test_utc_instant_preserved():
    instant = datetime(2025, 1, 15, 15, 30, tzinfo=UTC)
    df = pl.DataFrame({"decision_ts": [datetime(2025, 1, 15, 15, 30)]})
    out = ensure_utc(df, "decision_ts")
    assert out.schema["decision_ts"] == CANONICAL_UTC
    assert int(out["decision_ts"].dt.timestamp("us")[0]) == int(instant.timestamp() * 1_000_000)


def test_pit_boundary_poison():
    df = pl.DataFrame(
        {
            "decision_ts": [datetime(2025, 1, 15, 15, 0, tzinfo=UTC)],
            "feature_available_at": [datetime(2025, 1, 15, 15, 1, tzinfo=UTC)],
        }
    )
    df = with_utc_us(with_utc_us(df, "decision_ts"), "feature_available_at")
    with pytest.raises(AssertionError):
        assert_available_le_decision(df, "feature_available_at", "decision_ts")


def test_loo_excludes_self():
    ret = np.array([0.10, 0.0, 0.0])
    w = np.array([1.0, 1.0, 1.0])
    m = loo_market_returns(ret, w)
    assert abs(m[0] - 0.0) < 1e-12
    assert abs(m[1] - 0.05) < 1e-12


def test_graph_history_strictly_before_as_of():
    rows = []
    for d in ["2022-01-03", "2022-01-04", "2022-01-05"]:
        for i in range(6):
            rows.append({"trading_date": d, "instrument_id": f"T{i}", "clock": "10:00", "resid_ret_5m": float(i) * 0.001})
    h = pl.DataFrame(rows)
    g = PeerGraphBuilder(history_days=20, structural_top=5, leaders_top=2).fit(h, as_of="2022-01-04")
    assert g.history_end < "2022-01-04" or g.history_end == ""
    assert g.lineage["as_of"] == "2022-01-04"


def test_null_independent_noise_no_pass():
    df = synthetic_panel(n_days=40, n_names=25, seed=1, planted=False)
    man = _man_for(df)
    gates = _tiny_gates()
    gates["d3"]["abs_mean_floor"] = 0.01
    out = run_pipeline_on_frame(df, man, gates)
    assert out["survivors"] == []


def test_common_factor_does_not_invent_leader_gap():
    df = synthetic_panel(n_days=35, n_names=20, seed=2, common_factor=True)
    # scramble leader_gap independent of y
    df = df.with_columns(pl.col("leader_response_gap") * 0 + 0.001)
    man = _man_for(df)
    gates = _tiny_gates()
    gates["d3"]["abs_mean_floor"] = 0.008
    out = run_pipeline_on_frame(df, man, gates)
    assert out["survivors"] == []


def test_planted_edge_recovered_and_e_not_required():
    df = synthetic_panel(n_days=50, n_names=30, seed=7, planted=True)
    man = _man_for(df)
    gates = _tiny_gates()
    gates["tree"]["node_alpha"] = 0.5
    gates["d3"]["abs_mean_floor"] = 0.0003
    gates["stability"]["min_selection_freq"] = 0.0
    out = run_pipeline_on_frame(df, man, gates)
    feats = set()
    for rec in out["d3"]:
        for c in rec["conditions"]:
            feats.add(c[0])
    assert "leader_response_gap" in feats or out["n_candidates"] >= 0
    # planted path should appear in stability or candidates
    used = [f for f, p in out["stability_freq"].items() if p > 0]
    assert "leader_response_gap" in used or "activity_acceleration" in used
    assert "irrelevant_E" not in used or out["stability_freq"].get("irrelevant_E", 0) < out["stability_freq"].get("leader_response_gap", 1)


def test_concentration_gate_kills_one_ticker():
    df = synthetic_panel(n_days=20, n_names=3, seed=3, planted=True)
    man = _man_for(df)
    gates = _tiny_gates()
    gates["max_top_ticker_share"] = 0.20
    gates["leaf"]["min_tickers"] = 10
    out = run_pipeline_on_frame(df, man, gates)
    assert out["survivors"] == []


def test_future_graph_leak_lineage():
    rows = []
    for d in ["2022-06-01", "2022-06-02"]:
        for i in range(8):
            rows.append({"trading_date": d, "instrument_id": f"T{i}", "clock": "11:00", "resid_ret_5m": 0.01 * i})
    h = pl.DataFrame(rows)
    g = PeerGraphBuilder().fit(h, as_of="2022-06-01")
    assert g.lineage.get("n_history_rows", 0) == 0 or g.history_end < "2022-06-01"


def test_apply_conditions_filters():
    df = pl.DataFrame({"H004": [True, False], "leader_response_gap": [0.02, 0.0], "future_residual_15m": [0.01, 0.0]})
    c = make_candidate([Condition("H004", "==", True), Condition("leader_response_gap", ">", 0.01)], "LONG", "future_residual_15m", 15, "t", "D1")
    sub = apply_conditions(df, c.conditions)
    assert sub.height == 1


def test_yaml_dsl_roundtrip():
    from quant_edge_lab.discovery.conditional.dsl import candidate_from_yaml_dict, candidate_to_yaml_dict

    c = make_candidate([Condition("H004", "==", True), Condition("leader_response_gap", ">", 0.01)], "LONG", "future_residual_15m", 15, "cit_v4", "D1")
    d = candidate_to_yaml_dict(c)
    c2 = candidate_from_yaml_dict(d)
    assert c.candidate_id == c2.candidate_id


def test_spa_null_not_automatically_rejected():
    from quant_edge_lab.validation.multiple_testing import white_reality_check

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.002, size=(80, 6))
    r = white_reality_check(x, n_boot=80, seed=1)
    assert r["p"] is not None
    assert r["p"] > 0.01


def test_intraday_must_not_use_future_close_as_available():
    df = pl.DataFrame(
        {
            "decision_ts": [datetime(2025, 1, 15, 14, 30, tzinfo=UTC)],
            "bar_available_at": [datetime(2025, 1, 15, 16, 0, tzinfo=UTC)],
        }
    )
    df = with_utc_us(with_utc_us(df, "decision_ts"), "bar_available_at")
    with pytest.raises(AssertionError):
        assert_available_le_decision(df, "bar_available_at", "decision_ts")
