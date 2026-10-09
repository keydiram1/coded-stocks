"""V6 orchestration. Default is readiness-only. --execute is required to research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_edge_lab.discovery.v5.partitions import assert_sealed_oos_closed
from quant_edge_lab.discovery.v6.design import (
    CAMPAIGN_KEY,
    SCIENCE_ID,
    campaign_block,
    freeze_v6,
    frozen_cells,
    load_v6,
)

DEFAULT_RUN_ID = "v6-market-residual-shock-retention"
APPROVED = frozenset({"APPROVED", "FROZEN"})


def assert_execution_approved(des: dict[str, Any]) -> None:
    st = str(des.get("execution_status") or "NOT_APPROVED")
    if st not in APPROVED:
        raise RuntimeError(
            f"execute_v6 refused: execution_status={st} (must be APPROVED or FROZEN)."
        )


def readiness_v6(root: Path) -> dict[str, Any]:
    des = load_v6(root)
    ident = freeze_v6(root)
    assert_sealed_oos_closed(des)
    camp = campaign_block(des)
    cells = frozen_cells(des)
    return {
        "mode": "READINESS",
        "campaign_id": des["campaign_id"],
        "science_id": SCIENCE_ID,
        "campaign_key": CAMPAIGN_KEY,
        "identity": ident,
        "git": ident["git"],
        "design_hash": ident["design"],
        "selection_hash": ident["selection"],
        "data_manifest_hash": ident["data_manifest"],
        "science": ident["science"],
        "primary_entry": camp["clocks"]["primary_entry"],
        "skip_open": camp["clocks"]["skip_open"],
        "primary_outcome": camp["primary_outcome"],
        "d1_parameter_cells": 4,
        "confirmatory_signal_hypotheses": 3,
        "incremental_mechanism_gate": 1,
        "frozen_cells": cells,
        "d1_selection": camp["d1_grid"]["selection_among_eligible"],
        "d1_if_none": camp["d1_grid"]["if_no_eligible_cell"],
        "d2_open_d3_if_h1_fail": camp["promotion"]["d2"]["h1_fails_any_required_primary_gate"],
        "d2_incremental_p_lt": camp["promotion"]["d2"]["incremental_p_one_sided_lt"],
        "d3_incremental_p_lt": camp["promotion"]["d3"]["incremental_p_one_sided_lt"],
        "d3_if_h1_without_incremental": camp["promotion"]["d3"][
            "if_h1_ge_12bp_but_incremental_unsupported"
        ],
        "hypotheses": [h["hypothesis_id"] for h in camp["hypotheses"]],
        "splits": camp["splits"],
        "sealed_oos": des["sealed_oos"],
        "status": des["status"],
        "execution_status": des.get("execution_status"),
        "launch_command": "python -m quant_edge_lab discovery campaign-v6 --execute",
        "note": (
            "Default command does not run research. SIGNAL_ONLY. Sealed OOS closed. "
            "No D1 returns or event counts were computed."
        ),
    }


def execute_v6(root: Path, *, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    des = load_v6(root)
    assert_execution_approved(des)
    raise RuntimeError("execute_v6 reached unexpected path; campaign is not approved")


def run_campaign_v6(
    root: Path, *, execute: bool = False, run_id: str = DEFAULT_RUN_ID
) -> dict[str, Any]:
    if not execute:
        return readiness_v6(root)
    return execute_v6(root, run_id=run_id)
