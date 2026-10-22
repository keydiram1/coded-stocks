from pathlib import Path

import yaml

NEAR = 1.0
SHAPE_ORDER = (
    "persistent_repricing",
    "delayed_continuation",
    "temporary_continuation_then_reversal",
    "reversal",
    "fast_decay",
    "continuation",
    "no_stable_effect",
)
D3_GATES = [
    "sample_gates_pass",
    "concentration_gate_passes",
    "confirmatory_anchor_mean_gt_0",
    "confirmatory_anchor_mean_ge_12bp",
    "BH_survivor_q_0_10_across_same_frozen_K",
    "frozen_shape_specific_hierarchical_persistence_on_common_cohort",
    "anchor_direction_consistent_with_D2",
    "no_horizon_substitution",
    "no_nearby_point_rescue",
    "no_candidate_replacement",
]


def _load(name: str) -> dict:
    return yaml.safe_load(Path("knowledge/campaigns", name).read_text(encoding="utf-8"))


def _strong(x: float) -> bool:
    return abs(x) >= NEAR


def _bp(d: dict[str, float]) -> dict[str, float]:
    out = {"5": 0.0, "10": 0.0, "15": 0.0, "30": 0.0, "60": 0.0, "120": 0.0, "to_close": 0.0}
    out.update(d)
    return out


def classify_aligned(raw: dict[str, float]) -> str:
    """Frozen first-match on aligned bp. Design test, not an empirical runner."""
    p = _bp(raw)
    late = [p["15"], p["30"], p["60"], p["120"], p["to_close"]]
    if all(_strong(x) and x > 0 for x in late):
        return "persistent_repricing"
    delayed = (
        abs(p["5"]) < NEAR
        and abs(p["10"]) < NEAR
        and _strong(p["30"])
        and p["30"] > 0
        and _strong(p["60"])
        and p["60"] > 0
    )
    if delayed:
        return "delayed_continuation"
    early_pos = (_strong(p["5"]) and p["5"] > 0) or (_strong(p["10"]) and p["10"] > 0)
    late_neg = (_strong(p["60"]) and p["60"] < 0) or (_strong(p["120"]) and p["120"] < 0)
    if early_pos and late_neg:
        return "temporary_continuation_then_reversal"
    if _strong(p["15"]) and p["15"] < 0 and _strong(p["30"]) and p["30"] < 0:
        return "reversal"
    if _strong(p["5"]) and p["5"] > 0 and abs(p["30"]) < NEAR and abs(p["60"]) < NEAR:
        return "fast_decay"
    mid = [h for h in ("15", "30", "60") if _strong(p[h]) and p[h] > 0]
    if len(mid) >= 2 and not (_strong(p["120"]) and p["120"] < 0):
        return "continuation"
    return "no_stable_effect"


def temp_rev_persist(d2: dict[str, float], *, d1_anchor: str) -> bool:
    """Complete temporary shape must recur: early + AND late −."""
    p = _bp(d2)
    early = (_strong(p["5"]) and p["5"] > 0) or (_strong(p["10"]) and p["10"] > 0)
    late_anchor = _strong(p[d1_anchor]) and p[d1_anchor] < 0
    both_late_strong = _strong(p["60"]) and _strong(p["120"])
    both_late_neg = p["60"] < 0 and p["120"] < 0
    if both_late_strong and not both_late_neg:
        return False
    return early and late_anchor


def test_v7_is_design_only_and_does_not_reopen_closed_families():
    sel = _load("v7_next_campaign_selection.yaml")
    des = _load("v7_response_surface_design.yaml")
    v6 = _load("v6_shock_retention_family_closed.yaml")
    opt = _load("v5_options_local_rv_design.yaml")
    assert sel["execution_status"] == "NOT_APPROVED"
    assert des["execution_status"] == "NOT_APPROVED"
    assert des["frozen_before_empirical_execution"] is False
    assert v6["family_status"] == "CLOSED_NO_EDGE"
    assert opt["execution_status"] == "NOT_APPROVED"
    fam = des["MARKET_RELATIVE_SHOCK_RESOLUTION_V1"]
    es = fam["event_selection"]
    assert es["T_event"] == "P1_bar"
    assert es["signal_available_time"] == "immediately_after_P1_close"
    assert es["reservation_time"] == "signal_available_time"
    assert es["reserve_only_after_completed_P1"] is True
    assert es["unfinished_P1_cannot_qualify"] is True
    assert es["reserve_requires_known_impulse_residual_and_shock_z"] is True
    assert es["reserve_first_causal_qualifying_shock"] is True
    assert "common_cohort" not in "".join(es["causal_qualify_iff"])
    assert es["later_signal_cannot_replace_failed_reserved_event"] is True
    ids = [r["id"] for r in des["shape_rule"]["rules_in_order"]]
    assert ids == list(SHAPE_ORDER)
    assert ids.index("temporary_continuation_then_reversal") < ids.index("reversal")
    curve_a = {"5": 2.0, "10": 0.0, "15": -2.0, "30": -3.0, "60": -4.0}
    curve_b = {"5": 0.0, "10": 0.0, "15": -2.0, "30": -3.0, "60": -4.0}
    assert classify_aligned(curve_a) == "temporary_continuation_then_reversal"
    assert classify_aligned(curve_b) == "reversal"
    assert temp_rev_persist(curve_a, d1_anchor="60") is True
    d2_lost_early = {"5": 0.2, "10": 0.0, "15": -2.0, "30": -3.0, "60": -4.0}
    assert temp_rev_persist(d2_lost_early, d1_anchor="60") is False
    delayed_curve = {
        "5": 0.2,
        "10": -0.4,
        "15": 0.5,
        "30": 4.0,
        "60": 5.0,
        "120": 0.8,
        "to_close": 0.3,
    }
    assert classify_aligned(delayed_curve) == "delayed_continuation"
    d3 = des["d3"]
    assert d3["preregistered_before_d1"] is True
    assert d3["no_automatic_launch"] is True
    assert d3["economic_floor_bp"] == 12
    assert d3["bh_across"] == "exactly_the_same_frozen_K_hypotheses_as_d2"
    assert d3["required_gates"] == D3_GATES
    assert "confirmatory_signed_anchor" in d3["same_as_d2_freeze"]
    assert "anchor_horizon" in d3["same_as_d2_freeze"]
    assert "anchor_direction_consistent_with_D2" in d3["required_gates"]
    assert "VALIDATED_SUBTHRESHOLD_PHENOMENON" in d3["outcomes"]
    assert d3["confirmatory_h1"] == des["d2"]["confirmatory_h1"]
    by_id = {r["id"]: r for r in des["shape_rule"]["rules_in_order"]}
    assert by_id["temporary_continuation_then_reversal"]["confirmatory_signed_anchor"] == (
        "neg_aligned_anchor_return"
    )
    assert not Path("src/quant_edge_lab/discovery/v7").exists()
    assert fam["execute_now"] is False
    assert fam["clocks"]["signal_available_at"] == "immediately_after_P1_close"
    assert fam["aligned_forward_residual"]["formula"] == (
        "sign(impulse_residual) * forward_residual"
    )
    assert des["shape_classes"] == list(SHAPE_ORDER)
