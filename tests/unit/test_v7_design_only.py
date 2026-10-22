from pathlib import Path

import yaml


def _load(name: str) -> dict:
    return yaml.safe_load(Path("knowledge/campaigns", name).read_text(encoding="utf-8"))


def test_v7_is_design_only_and_does_not_reopen_closed_families():
    sel = _load("v7_next_campaign_selection.yaml")
    des = _load("v7_response_surface_design.yaml")
    v6 = _load("v6_shock_retention_family_closed.yaml")
    opt = _load("v5_options_local_rv_design.yaml")
    assert sel["status"] == "DESIGN_ONLY"
    assert sel["execution_status"] == "NOT_APPROVED"
    assert sel["v6_shock_retention_family"] == "CLOSED_NO_EDGE"
    assert sel["close_dislocation_family"] == "CLOSED_NO_EDGE"
    assert sel["new_paid_data"] == "forbidden_at_this_stage"
    assert des["status"] == "DESIGN_ONLY"
    assert des["execution_status"] == "NOT_APPROVED"
    assert des["frozen_before_empirical_execution"] is False
    assert des["sealed_oos"] == "inaccessible"
    assert v6["family_status"] == "CLOSED_NO_EDGE"
    assert opt["execution_status"] == "NOT_APPROVED"
    fam = des["MARKET_RELATIVE_SHOCK_RESOLUTION_V1"]
    assert fam["event_grid"]["impulse_window_minutes"] == [5, 15, 30, 60]
    assert fam["event_grid"]["cartesian_event_definitions"] == 4
    assert fam["response_grid_minutes"] == [5, 10, 15, 30, 60, 120]
    assert fam["clocks"]["primary_entry"] == "T_event_plus_2_open"
    assert fam["clocks"]["skip_open"] == "T_event_plus_1_open"
    assert fam["clocks"]["max_impulse_start"] == {
        "W5": "13:54",
        "W15": "13:44",
        "W30": "13:29",
        "W60": "12:59",
    }
    assert fam["event_selection"]["per_ticker_day"] == "first_qualifying_shock_only"
    assert fam["event_selection"]["future_outcomes_not_used_to_choose_or_replace_the_event"] is True
    assert fam["forward_residual"]["not_raw_stock_return"] is True
    assert "eligible" in fam["residual"]["market"]
    assert fam["primary_estimand"] == "equal_weight_trading_day_mean"
    assert des["response_surface"]["common_cohort_only_for_shape_stability_selection"] is True
    assert des["response_surface"]["descriptive_unbalanced_cannot_determine_shape"] is True
    by_id = {r["id"]: r for r in des["shape_rule"]["rules_in_order"]}
    assert by_id["fast_decay"]["anchor_minutes"] == 5
    assert by_id["reversal"]["direction"] == "opposite_to_shock"
    assert by_id["temporary_continuation_then_reversal"]["direction"] == "opposite_to_shock"
    assert by_id["persistent_repricing"]["anchor_minutes"] == 30
    assert des["d1"]["each_frozen_candidate_is_exactly_one_confirmatory_anchor_hypothesis"] is True
    assert des["d2"]["bh"]["across"] == "exactly_the_K_frozen_anchor_hypotheses"
    assert des["d2"]["bh"]["not_across_horizons_of_the_curve"] is True
    assert "shape_conditional_persistence_on_common_cohort" in des["d2"][
        "hierarchical_required_gates_not_in_bh"
    ]
    assert des["multiplicity"]["persistence_is_hierarchical_not_bh"] is True
    assert set(des["mechanism_families"]) == {
        "A_MARKET_RELATIVE_SHOCK_RESOLUTION",
        "B_SHOCK_ABSORPTION_RETENTION",
        "C_ABNORMAL_VOLUME_PRICE_RESPONSE",
        "D_CROSS_SECTIONAL_RELATIVE_DISLOCATION",
    }
    assert des["mechanism_families"]["A_MARKET_RELATIVE_SHOCK_RESOLUTION"][
        "status"
    ].startswith("DESIGN_ONLY")
    for key in ("B_SHOCK_ABSORPTION_RETENTION", "C_ABNORMAL_VOLUME_PRICE_RESPONSE",
                "D_CROSS_SECTIONAL_RELATIVE_DISLOCATION"):
        assert "NOT_IN_FIRST_D1_EXECUTE" in des["mechanism_families"][key]["status"]
    assert des["d1"]["select_at_most"] == 2
    assert des["d1"]["selection"] == "stable_shapes_not_max_mean"
    assert des["d2"]["no_nearby_point_rescue"] is True
    assert des["d2"]["next_action_even_if_pass"] == "STOP_FOR_INDEPENDENT_REVIEW"
    assert des["d3"]["no_automatic_launch"] is True
    assert des["stability"]["cohort"] == "common_cohort_only"
    assert des["stability"]["neighborhood_is_shape_conditional"] is True
    assert "shape_conditional_hierarchical_persistence" in des["stability"]["require_all"]
    assert des["multiplicity"]["first_wave_discovery_curves"] == 4
    assert des["multiplicity"]["horizons_are_descriptors_not_extra_trials"] is True
    assert fam["execute_now"] is False
