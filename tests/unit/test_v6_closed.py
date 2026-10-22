from pathlib import Path

import yaml

from quant_edge_lab.discovery.v6.d1_pin import FROZEN_CELL, FROZEN_D1_IDENTITY, FROZEN_D1_JSON_SHA256
from quant_edge_lab.hashing import sha256_file


def _load(name: str) -> dict:
    return yaml.safe_load(Path("knowledge/campaigns", name).read_text(encoding="utf-8"))


def test_v6_is_permanently_closed_without_mutating_science_yaml():
    closed = _load("v6_shock_retention_family_closed.yaml")
    des = _load("v6_shock_retention_design.yaml")
    sel = _load("v6_next_campaign_selection.yaml")
    assert closed["family_status"] == "CLOSED_NO_EDGE"
    assert closed["d1"]["decision"] == "FREEZE"
    assert closed["d1"]["selected_id"] == "W15_W15"
    assert closed["d1"]["frozen_cell"] == FROZEN_CELL
    assert closed["d2"]["decision"] == "KILL_AT_D2"
    assert closed["d2"]["failure_reasons"] == [
        "below_d2_floor",
        "bh_not_survivor",
        "incremental_mean",
    ]
    assert closed["d3_opened"] is False
    assert closed["d2"]["d3_opened"] is False
    assert closed["sealed_oos"] == "inaccessible"
    assert closed["no_d3"] is True
    assert closed["no_rescue"] is True
    assert sha256_file(Path("knowledge/campaigns/v6_shock_retention_design.yaml")) == (
        FROZEN_D1_IDENTITY["design"]
    )
    assert sha256_file(Path("knowledge/campaigns/v6_next_campaign_selection.yaml")) == (
        FROZEN_D1_IDENTITY["selection"]
    )
    assert des["execution_status"] == "FROZEN"
    assert sel["execution_status"] == "NOT_APPROVED"
    assert closed["identities"]["d1_json_sha256"] == FROZEN_D1_JSON_SHA256
    assert closed["identities"]["calendar"] == FROZEN_D1_IDENTITY["calendar"]
    assert closed["implementation_sha"].startswith("77ddadd")
