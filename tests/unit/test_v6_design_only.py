from pathlib import Path

import yaml


def test_v6_design_is_not_executable_and_does_not_reopen_closed_families():
    sel = yaml.safe_load(
        Path("knowledge/campaigns/v6_next_campaign_selection.yaml").read_text(
            encoding="utf-8"
        )
    )
    des = yaml.safe_load(
        Path("knowledge/campaigns/v6_shock_retention_design.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert sel["recommended_next_campaign"] == "IDIO_SHOCK_RETENTION_V1"
    assert sel["close_dislocation_family"] == "CLOSED_NO_EDGE"
    assert sel["options_local_rv"] == "DESIGN_ONLY_NOT_READY"
    assert sel["execution_status"] == "NOT_APPROVED"
    assert des["execution_status"] == "NOT_APPROVED"
    assert des["sealed_oos"] == "inaccessible"
    camp = des["IDIO_SHOCK_RETENTION_V1"]
    assert camp["trial_count"] == 3
    assert camp["shock_abs_quantile"] == "TO_BE_SELECTED_IN_DISCOVERY"
    roles = {h["hypothesis_id"]: h["role"] for h in camp["hypotheses"]}
    assert roles["H1_HIGH_RETENTION_CONTINUATION"] == "primary"
    assert roles["H3_IMPULSE_ONLY_CONTROL"] == "control"
    assert camp["hypotheses"][2]["do_not_promote_to_primary"] is True
    opt = yaml.safe_load(
        Path("knowledge/campaigns/v5_options_local_rv_design.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert opt["execution_status"] == "NOT_APPROVED"
    closed = yaml.safe_load(
        Path("knowledge/campaigns/v5_close_dislocation_family_closed.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert closed["family_status"] == "CLOSED_NO_EDGE"
