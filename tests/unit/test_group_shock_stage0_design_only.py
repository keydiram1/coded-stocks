from pathlib import Path
from statistics import median

import yaml

FORBIDDEN_FUTURE = (
    "future_1m_return",
    "future_5m_return",
    "future_30m_return",
    "next_day_return",
    "convergence",
    "reversal",
    "continuation",
    "response_gap_profitability",
    "p_values_on_future_outcomes",
)
SHOCK_MIN = {"H5M": 40, "H30M": 25, "H1D": 12}
NESTED_MIN = {"H5M": 250, "H30M": 150, "H1D": 80}


def _load(name: str) -> dict:
    return yaml.safe_load(Path("knowledge/campaigns", name).read_text(encoding="utf-8"))


def group_move_loo(returns: dict[str, float], target: str) -> float:
    others = [v for k, v in returns.items() if k != target]
    return float(median(others))


def centered_shock_z(move: float, prior: list[float]) -> float:
    loc = float(median(prior))
    mad = float(median([abs(x - loc) for x in prior]))
    return (move - loc) / (1.4826 * mad)


def first_independent_events(times: list[int], cooldown: int) -> list[int]:
    kept: list[int] = []
    last = None
    for t in times:
        if last is None or t >= last + cooldown:
            kept.append(t)
            last = t
    return kept


def stream_events(candidates: list[tuple[int, float]], thresh: float, cooldown: int) -> list[int]:
    times = [t for t, z in candidates if abs(z) >= thresh]
    return first_independent_events(times, cooldown)


def test_stage0_is_design_only_and_does_not_reopen_v6_v7():
    sel = _load("group_shock_response_gap_selection.yaml")
    des = _load("group_shock_response_gap_design.yaml")
    v6 = _load("v6_shock_retention_family_closed.yaml")
    v7 = _load("v7_response_surface_design.yaml")
    v6sci = _load("v6_shock_retention_design.yaml")
    assert sel["execution_status"] == "NOT_APPROVED"
    assert des["execution_status"] == "NOT_APPROVED"
    assert des["execute_now"] is False
    assert des["no_automatic_launch"] is True
    assert des["stage"] == 0
    assert v6["family_status"] == "CLOSED_NO_EDGE"
    assert v6sci["execution_status"] == "FROZEN"
    assert v7["execution_status"] == "NOT_APPROVED"
    assert Path("src/quant_edge_lab/discovery/group_shock_response_gap").exists() is False
    fam = des["GROUP_SHOCK_RESPONSE_GAP_V1"]
    assert fam["empirical_runner_exists"] is False
    assert fam["stage1_not_yet"]["authorized"] is False
    assert fam["stage1_not_yet"]["contemporaneous_multi_group_firing_is_not_independent"] is True


def test_horizons_and_no_future_returns():
    des = _load("group_shock_response_gap_design.yaml")
    fam = des["GROUP_SHOCK_RESPONSE_GAP_V1"]
    ids = [h["id"] for h in fam["horizons"]]
    assert ids == ["H5M", "H30M", "H1D"]
    assert fam["do_not_search_other_horizons"] is True
    forbidden = set(des["forbidden_until_stage1_authorization"])
    assert set(FORBIDDEN_FUTURE) <= forbidden
    assert fam["stage0_outputs"]["stop_before_any_future_return"] is True
    assert fam["clocks"]["entry_or_forward_clock"] == "not_in_stage0"
    assert fam["splits_stage0"]["sealed_oos"] == "inaccessible"
    assert fam["splits_stage0"]["d1_d2_d3_not_opened"] is True


def test_group_sources_are_not_true_pit():
    sel = _load("group_shock_response_gap_selection.yaml")
    fam = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]
    gdef = fam["group_definitions"]
    assert gdef["do_not_cluster_on_returns"] is True
    assert gdef["residual_correlation_peer_graph"] == "forbidden_as_economic_group"
    assert gdef["current_snapshot_membership"] == "forbidden"
    by_id = {x["id"]: x for x in gdef["hierarchy"]}
    assert by_id["GICS_INDUSTRY"]["pit_status"] == "NOT_PIT_READY"
    assert by_id["GICS_INDUSTRY_GROUP"]["pit_status"] == "NOT_PIT_READY"
    assert by_id["SECTOR_ETF_CONSTITUENTS"]["pit_status"] == "NOT_PIT_READY"
    cand = gdef["source_audit_authorized"]
    assert cand["classification_system"] == "SIC_not_GICS"
    assert cand["call_it_pit_gics"] == "forbidden"
    assert cand["call_it_true_pit_sic"] == "forbidden"
    assert cand["not_as_known_on_that_day"] is True
    assert cand["massive_history_semantics"] == "effective_dated_as_currently_maintained"
    assert cand["still_not_pit_ready"] is True
    assert cand["authorized_now"] == "source_audit_and_sample_store_only"
    assert cand["stage0_feasibility"] == "not_authorized"
    assert gdef["assignment_rule"]["run_both_then_pick_better"] == "forbidden"
    assert gdef["assignment_rule"]["gics_industry_is_primary"] is True
    assert gdef["assignment_rule"]["fallback_only_if_primary_eligible_members_lt"] == 8
    assert gdef["assignment_rule"]["membership_store_must_not_choose_fallback"] is True
    assert gdef["assignment_rule"]["decided_in"] == "stage0_at_event_not_in_membership_store"
    assert gdef["assignment_rule"]["fallback_level"] == "sic_major_group"
    assert gdef["assignment_rule"]["sic2_is_not_gics_industry_group"] is True
    assert gdef["assignment_rule"]["fallback_parent_is_all_eligible_sic2_members"] is True
    assert gdef["instrument_id_scheme"] == "figi_then_cik_then_ticker"
    assert gdef["ticker_is_asof_attribute"] is True
    assert sel["massive_sic_is_not_pit_gics"] is True
    assert sel["research_question"] == "APPROVED"
    assert sel["stage0_execution"] == "NOT_APPROVED"
    assert sel["authorized_now"] == "pit_membership_store_and_source_audit_only"
    assert sel["massive_history_is_not_as_known_on_day"] is True
    assert fam["proposed_pass_kill_gates"]["not_pit"] == "NOT_PIT_READY_not_PASS"


def test_loo_median_excludes_target_and_is_not_mean():
    rets = {"a": 0.10, "b": 0.01, "c": 0.02, "d": 0.03, "e": 0.04}
    loo_a = group_move_loo(rets, "a")
    assert loo_a == float(median([0.01, 0.02, 0.03, 0.04]))
    assert loo_a != sum(rets.values()) / len(rets)
    fam = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]
    gm = fam["group_move"]
    assert gm["primary_id"] == "robust_loo_median"
    assert gm["no_self_contamination"] is True
    assert gm["robustness_diagnostic_frozen"]["may_replace_primary_after_seeing_results"] is False


def test_unfinished_window_cannot_qualify_and_reservation_after_close():
    clocks = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]["clocks"]
    assert clocks["reserve_only_after_completed_window"] is True
    assert clocks["unfinished_window_cannot_qualify"] is True
    assert clocks["signal_available_at"] == "immediately_after_window_end_close"


def test_shock_z_is_centered_same_clock():
    prior = [0.01, 0.02, 0.03] * 20
    z = centered_shock_z(0.02, prior)
    assert abs(z) < 1e-12
    sh = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]["group_shock"]
    assert sh["centered"] is True
    assert sh["formula"] == (
        "(group_move_existence - historical_same_clock_median) / (1.4826 * MAD)"
    )
    assert sh["uncentered_group_move_over_MAD"] == "forbidden"
    assert sh["historical_scale"]["current_event_excluded_from_location_and_scale"] is True
    assert sh["primary_severity_for_pass_kill"] == 2.5
    assert sh["do_not_optimize_for_alpha"] is True


def test_severity_streams_have_independent_reservation():
    fam = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]
    ind = fam["independence"]
    assert ind["shared_cooldown_across_severity_ladder"] == "forbidden"
    assert ind["reservation"] == "independent_per_abs_z_threshold"
    assert ind["pass_kill_uses_only_the_2_5_stream"] is True
    roles = {s["abs_z"]: s["role"] for s in ind["streams"]}
    assert roles[2.0] == "descriptive_sample_size_only"
    assert roles[2.5] == "primary_pass_kill"
    assert roles[3.0] == "descriptive_sample_size_only"
    candidates = [(10, 2.1), (15, 3.5)]
    shared = first_independent_events([10, 15], cooldown=30)
    assert shared == [10]
    assert stream_events(candidates, 3.0, cooldown=30) == [15]
    assert stream_events(candidates, 2.0, cooldown=30) == [10]


def test_shock_count_gates_differ_by_horizon():
    gates = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"][
        "proposed_pass_kill_gates"
    ]
    assert gates["uniform_min_40_shocks_every_horizon"] == "forbidden"
    assert gates["min_independent_group_shocks"] == SHOCK_MIN
    assert gates["min_total_independent_h1d_shocks_across_groups"] == 40
    assert gates["min_groups_with_h1d_shocks_ge_8"] == 5
    assert gates["min_stock_level_nested_observations"] == NESTED_MIN
    assert gates["evaluated_at_severity"] == 2.5


def test_dispersion_is_excess_vs_same_clock_baseline():
    fam = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]
    disp = fam["dispersion"]
    assert disp["primary_metric"] == "excess_dispersion"
    assert disp["excess_dispersion"] == "shock_dispersion / historical_same_clock_dispersion"
    assert disp["purpose"] == "abnormal_dispersion_during_group_shock_not_ordinary_tracking_noise"
    scale = fam["residual_scale"]
    assert scale["formula"] == "1.4826 * MAD of the exact 60 prior same-clock residuals"
    assert scale["historical_windows_use_rolling_origin"] is True
    assert scale["do_not_apply_event_time_model_backward"] is True
    assert scale["prior_only"] is True
    assert scale["current_event_excluded"] is True
    assert fam["dispersion"]["historical_baseline_rolling_origin"] is True
    gates = fam["proposed_pass_kill_gates"]
    assert gates["min_median_excess_dispersion"] == 1.30
    assert gates["min_median_mad_of_standardized_residuals"] == 0.35
    assert gates["mad_floor_role"] == "secondary_not_the_phenomenon"
    fam_gates = gates["family_pass_kill"]
    assert fam_gates["cherry_pick_best_group_after_seeing_results"] == "forbidden"
    assert fam_gates["stage1_includes"] == "all_groups_that_pass_that_horizon"
    assert fam_gates["min_groups_pass_per_horizon"] == {"H5M": 8, "H30M": 6, "H1D": 5}
    assert fam_gates["if_zero_horizons_meet_family_gate"] == "FAMILY_KILL_AT_STAGE0"
    assert fam["discovery_package_must_not_exist_until_stage0_authorized"] is True
    assert Path("src/quant_edge_lab/discovery/group_shock_response_gap").exists() is False


def test_beta_stability_is_relative_plus_precision():
    st = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"][
        "sensitivity_stability"
    ]
    iff = st["stable_iff"]
    assert iff["n_pairs_ge"] == 60
    assert iff["abs_tstat_ge"] == 2.0
    assert iff["ols_r_squared_ge"] == 0.10
    assert iff["relative_rolling_beta_iqr_le"] == 0.50
    assert "abs_median_rolling_beta_ge" not in iff
    assert "rolling_beta_iqr_le" not in iff
    assert st["absolute_abs_beta_ge_0_20"] == "not_a_kill_gate"
    assert st["small_stable_beta_is_usable"] is True


def test_leakage_list_and_v4_graph_not_reused_as_groups():
    fam = _load("group_shock_response_gap_design.yaml")["GROUP_SHOCK_RESPONSE_GAP_V1"]
    assert "group_return_excludes_target" in fam["leakage"]
    assert "no_future_returns_in_stage0_outputs" in fam["leakage"]
    assert "v4_residual_correlation_peer_graph" in fam["do_not_reuse_as_groups"]
    assert "V6_MARKET_RESIDUAL_SHOCK_RETENTION" in fam["do_not_reopen"]
    assert "V7_MARKET_RELATIVE_SHOCK_RESOLUTION" in fam["do_not_reopen"]
