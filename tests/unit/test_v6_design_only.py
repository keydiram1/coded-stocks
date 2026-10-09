from itertools import product
from pathlib import Path

import yaml


def _load(name: str) -> dict:
    return yaml.safe_load(Path("knowledge/campaigns", name).read_text(encoding="utf-8"))


def test_v6_design_is_not_executable_and_does_not_reopen_closed_families():
    sel = _load("v6_next_campaign_selection.yaml")
    des = _load("v6_shock_retention_design.yaml")
    assert sel["recommended_next_campaign"] == "MARKET_RESIDUAL_SHOCK_RETENTION_V1"
    assert sel["primary_entry"] == "T1_plus_2_open"
    assert sel["skip_open"] == "T1_plus_1_open"
    assert sel["d3_is_globally_untouched"] is False
    assert sel["close_dislocation_family"] == "CLOSED_NO_EDGE"
    assert sel["options_local_rv"] == "DESIGN_ONLY_NOT_READY"
    assert sel["execution_status"] == "NOT_APPROVED"
    assert des["execution_status"] == "NOT_APPROVED"
    assert des["sealed_oos"] == "inaccessible"
    assert "IDIO_SHOCK_RETENTION_V1" not in des
    camp = des["MARKET_RESIDUAL_SHOCK_RETENTION_V1"]
    assert camp["clocks"]["primary_entry"] == "T1_plus_2_open"
    assert camp["clocks"]["skip_open"] == "T1_plus_1_open"
    assert camp["clocks"]["P0"].startswith("completed close")
    assert camp["retention"]["formula"] == "remaining_residual / impulse_residual"
    assert camp["retention"]["clip"] is False
    assert camp["shock_z"]["formula"] == "impulse_residual / prior_same_clock_residual_std"
    assert camp["shock_z"]["ddof"] == 1
    assert camp["shock_z"]["require_all_20_finite"] is True
    assert camp["shock_z"]["reachback"] is False
    assert camp["shock_z"]["imputation"] is False
    assert camp["residual"]["sector_factor"] == "not_used"
    roles = {h["hypothesis_id"]: h["role"] for h in camp["hypotheses"]}
    assert roles["H1_HIGH_RETENTION_CONTINUATION"] == "primary"
    assert roles["H2_LOW_RETENTION_REVERSAL"] == "robustness"
    assert roles["H3_IMPULSE_ONLY_CONTROL"] == "control"
    assert camp["hypotheses"][1]["cannot_rescue_h1"] is True
    assert camp["hypotheses"][2]["cannot_rescue_h1"] is True
    inc = camp["incremental_control"]
    assert inc["method"] == "stratify_then_within_stratum_difference"
    assert inc["required_gate_on_h1"] == "day_delta_mean_gt_0"
    grid = camp["d1_grid"]
    n = 1
    for key in (
        "impulse_window_minutes",
        "wait_window_minutes",
        "min_abs_shock_z",
        "high_retention_min",
        "low_retention_max",
    ):
        n *= len(grid[key])
    assert n == 4
    assert grid["cartesian_combinations"] == 4
    assert grid["total_effective_discovery_trials"] == 4
    assert grid["confirmatory_hypotheses_after_freeze"] == 3
    assert grid["frozen_before_d1"] is True
    assert list(product(grid["impulse_window_minutes"], grid["wait_window_minutes"])) == [
        (5, 5),
        (5, 15),
        (15, 5),
        (15, 15),
    ]
    buckets = camp["time_of_day"]["matching_buckets_et_on_T1"]
    assert buckets[0] == "10:00"
    assert all(isinstance(b, str) for b in buckets)
    assert camp["campaign_success"]["requires"] == "H1_PRIMARY_pass"
    assert "not globally untouched" in camp["splits"]["d3_language"]
    opt = _load("v5_options_local_rv_design.yaml")
    assert opt["execution_status"] == "NOT_APPROVED"
    closed = _load("v5_close_dislocation_family_closed.yaml")
    assert closed["family_status"] == "CLOSED_NO_EDGE"
