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
    assert inc["strata"] == [
        "sign_impulse_residual",
        "abs_shock_z_bucket",
        "T1_30m_et_bucket",
    ]
    assert inc["sign_impulse_residual"] == ["POSITIVE", "NEGATIVE"]
    assert inc["no_cross_sign_matching"] is True
    assert inc["one_sided_strata_do_not_contribute"] is True
    assert inc["required_gate_on_h1"] == "day_delta_mean_gt_0"
    assert inc["d1_requires"]["p_one_sided"] == "not_required"
    assert inc["d2_d3_requires"]["p_one_sided_lt"] == 0.05
    assert inc["d2_d3_requires"]["n_matched_days_min"] == 100
    assert inc["economic_floor_on_delta"] == "none"
    assert inc["hierarchical_not_fourth_hypothesis"] is True
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
    assert grid["d1_parameter_cells"] == 4
    assert [c["id"] for c in grid["frozen_cells"]] == ["W5_W5", "W5_W15", "W15_W5", "W15_W15"]
    assert grid["confirmatory_hypotheses"] == 3
    assert grid["if_no_eligible_cell"] == "KILL_AT_D1"
    assert "H1_equal_day_mean_gt_0" in grid["cell_eligible_for_selection_iff"]
    assert "incremental_matched_day_delta_mean_gt_0" in grid["cell_eligible_for_selection_iff"]
    assert camp["multiplicity"]["discovery_d1_parameter_cells"] == 4
    assert camp["multiplicity"]["confirmatory_hypotheses_after_cell_freeze"] == 3
    assert camp["multiplicity"]["confirmatory_signal_hypotheses"] == 3
    assert camp["multiplicity"]["incremental_mechanism_gate"] == 1
    assert camp["multiplicity"]["incremental_not_in_bh"] is True
    assert camp["multiplicity"]["d2_d3_bh_q"] == 0.10
    promo = camp["promotion"]
    assert promo["d1"]["if_no_eligible_cell"] == "KILL_AT_D1"
    assert promo["d2"]["h1_fails_any_required_primary_gate"] == (
        "KILL_campaign_DO_NOT_OPEN_D3"
    )
    assert promo["d2"]["economic_floor_bp"] == 3
    assert promo["d2"]["statistical_requirement"] == (
        "H1_is_BH_survivor_q_0_10_across_H1_H2_H3"
    )
    assert promo["d2"]["incremental_day_delta_requirement"] == "day_delta_mean_gt_0"
    assert promo["d2"]["incremental_p_one_sided_lt"] == 0.05
    assert promo["d2"]["incremental_n_matched_days_min"] == 100
    assert promo["d2"]["incremental_economic_floor"] == "none"
    assert promo["d3"]["open_only_if"] == "H1_survives_all_D2_primary_gates"
    assert promo["d3"]["economic_floor_bp"] == 12
    assert promo["d3"]["bh_q"] == 0.10
    assert promo["d3"]["incremental_day_delta_gate"] == "day_delta_mean_gt_0"
    assert promo["d3"]["incremental_p_one_sided_lt"] == 0.05
    assert promo["d3"]["incremental_n_matched_days_min"] == 100
    assert (
        promo["d3"]["if_h1_ge_12bp_but_incremental_unsupported"]
        == "PRIMARY_SIGNAL_POSITIVE_BUT_RETENTION_NOT_INCREMENTAL"
    )
    assert "incremental_p_one_sided_lt_0.05" in promo["d3"]["RESEARCH_PASS_requires"]
    assert "H1_PRIMARY_itself_passes" in promo["d3"]["RESEARCH_PASS_requires"]
    assert promo["d3"]["h2_or_h3_pass_cannot_create_RESEARCH_PASS"] is True
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
