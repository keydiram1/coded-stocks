from pathlib import Path

import yaml

NEAR = 1.0
SHAPE_ORDER = (
    "persistent_repricing",
    "delayed_continuation",
    "reversal",
    "temporary_continuation_then_reversal",
    "fast_decay",
    "continuation",
    "no_stable_effect",
)


def _load(name: str) -> dict:
    return yaml.safe_load(Path("knowledge/campaigns", name).read_text(encoding="utf-8"))


def _strong(x: float) -> bool:
    return abs(x) >= NEAR


def classify_aligned(bp: dict[str, float]) -> str:
    """Frozen first-match on aligned bp. Design test, not an empirical runner."""
    p = bp
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
    if _strong(p["15"]) and p["15"] < 0 and _strong(p["30"]) and p["30"] < 0:
        return "reversal"
    early_pos = (_strong(p["5"]) and p["5"] > 0) or (_strong(p["10"]) and p["10"] > 0)
    late_neg = (_strong(p["60"]) and p["60"] < 0) or (_strong(p["120"]) and p["120"] < 0)
    if early_pos and late_neg:
        return "temporary_continuation_then_reversal"
    if _strong(p["5"]) and p["5"] > 0 and abs(p["30"]) < NEAR and abs(p["60"]) < NEAR:
        return "fast_decay"
    mid = [h for h in ("15", "30", "60") if _strong(p[h]) and p[h] > 0]
    if len(mid) >= 2 and not (_strong(p["120"]) and p["120"] < 0):
        return "continuation"
    return "no_stable_effect"


def test_v7_is_design_only_and_does_not_reopen_closed_families():
    sel = _load("v7_next_campaign_selection.yaml")
    des = _load("v7_response_surface_design.yaml")
    v6 = _load("v6_shock_retention_family_closed.yaml")
    opt = _load("v5_options_local_rv_design.yaml")
    assert sel["status"] == "DESIGN_ONLY"
    assert sel["execution_status"] == "NOT_APPROVED"
    assert des["status"] == "DESIGN_ONLY"
    assert des["execution_status"] == "NOT_APPROVED"
    assert des["frozen_before_empirical_execution"] is False
    assert v6["family_status"] == "CLOSED_NO_EDGE"
    assert opt["execution_status"] == "NOT_APPROVED"
    fam = des["MARKET_RELATIVE_SHOCK_RESOLUTION_V1"]
    es = fam["event_selection"]
    assert "qualify_iff" not in es
    assert "common_cohort_P0_through_1559" not in es["causal_qualify_iff"]
    assert all("1559" not in x and "common_cohort" not in x for x in es["causal_qualify_iff"])
    assert es["common_cohort_not_in_causal_qualify_iff"] is True
    assert es["reserve_first_at_T_event"] is True
    assert es["reserve_using_only_information_available_by_T_event"] is True
    post = es["post_reservation_common_cohort_filter"]
    assert "every_target_minute_P0_through_1559" in post["requires"]
    assert post["do_not_search_later_shock"] is True
    assert post["do_not_re_reserve"] is True
    assert es["later_signal_cannot_replace_failed_reserved_event"] is True
    assert fam["aligned_forward_residual"]["formula"] == (
        "sign(impulse_residual) * forward_residual"
    )
    assert fam["aligned_forward_residual"]["used_everywhere_in_v7_shape_and_confirmation"] is True
    ids = [r["id"] for r in des["shape_rule"]["rules_in_order"]]
    assert ids == list(SHAPE_ORDER)
    by_id = {r["id"]: r for r in des["shape_rule"]["rules_in_order"]}
    for sid in (
        "persistent_repricing",
        "delayed_continuation",
        "fast_decay",
        "continuation",
    ):
        assert by_id[sid]["confirmatory_signed_anchor"] == "aligned_anchor_return"
    for sid in ("reversal", "temporary_continuation_then_reversal"):
        assert by_id[sid]["confirmatory_signed_anchor"] == "neg_aligned_anchor_return"
    assert des["d2"]["continuation_like_confirmatory_signed_anchor"] == "aligned_anchor_return"
    assert des["d2"]["reversal_like_confirmatory_signed_anchor"] == "neg_aligned_anchor_return"
    assert des["d2"]["confirmatory_h1"] == "mean_confirmatory_signed_anchor_gt_0"
    assert des["shape_rule"]["uses_aligned_coordinates_only"] is True
    for rec in des["shape_rule"]["rules_in_order"]:
        blob = yaml.safe_dump(rec)
        assert "same_as_shock" not in blob
        assert "opposite_to_shock" not in blob
    delayed_curve = {
        "5": 0.2, "10": -0.4, "15": 0.5, "30": 4.0, "60": 5.0, "120": 0.8, "to_close": 0.3,
    }
    assert classify_aligned(delayed_curve) == "delayed_continuation"
    assert des["shape_classes"] == list(SHAPE_ORDER)
    assert not Path("src/quant_edge_lab/discovery/v7").exists()
    assert fam["execute_now"] is False
    assert des["d1"]["select_at_most"] == 2
    assert des["d2"]["no_nearby_point_rescue"] is True
    assert des["d3"]["no_automatic_launch"] is True
    assert fam["clocks"]["max_impulse_start"]["W5"] == "13:54"
    assert "eligible" in fam["residual"]["market"]
    assert fam["forward_residual"]["not_raw_stock_return"] is True
    assert des["response_surface"]["descriptive_unbalanced_cannot_determine_shape"] is True
    assert des["d2"]["bh"]["not_across_horizons_of_the_curve"] is True
