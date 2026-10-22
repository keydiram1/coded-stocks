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
    assert fam["primary_estimand"] == "equal_weight_trading_day_mean"
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
    assert des["stability"]["require_all"] == [
        "shape_not_no_stable_effect",
        "sample_and_concentration_on_anchor",
        "neighborhood_same_sign",
        "not_singleton_spike",
        "d1_subperiod_sign_agreement",
    ]
    assert des["multiplicity"]["first_wave_discovery_curves"] == 4
    assert des["multiplicity"]["horizons_are_descriptors_not_extra_trials"] is True
    assert fam["execute_now"] is False
