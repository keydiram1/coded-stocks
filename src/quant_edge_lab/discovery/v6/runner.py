"""V6 orchestration. Default is readiness-only. --execute is required to research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v5.eligibility import instruments_path
from quant_edge_lab.discovery.v5.partitions import assert_sealed_oos_closed
from quant_edge_lab.discovery.v6.calendar import (
    calendar_manifest,
    d1_loadable_days,
    require_calendar_identity,
    v6_calendar_report,
)
from quant_edge_lab.discovery.v6.checkpoint import require_resume_identity, v6_run_dir
from quant_edge_lab.discovery.v6.d1_pin import (
    FROZEN_CELL,
    FROZEN_D1_IDENTITY,
    FROZEN_D1_JSON_SHA256,
    FROZEN_SELECTED_ID,
    load_d1_pin,
)
from quant_edge_lab.discovery.v6.design import (
    CAMPAIGN_KEY,
    SCIENCE_ID,
    campaign_block,
    freeze_v6,
    frozen_cells,
    load_v6,
)
from quant_edge_lab.discovery.v6.pipeline import walk_d1, walk_d2

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
    cal = v6_calendar_report(root, des)
    ident = {**ident, "calendar": cal["calendar_hash_actual"]}
    d1_path = v6_run_dir(root, DEFAULT_RUN_ID) / "d1.json"
    d1_sha = None
    d1_pin_ok = False
    d1_selected = None
    if d1_path.exists():
        from quant_edge_lab.hashing import sha256_file

        d1_sha = sha256_file(d1_path)
        d1_pin_ok = d1_sha == FROZEN_D1_JSON_SHA256
        d1_selected = FROZEN_SELECTED_ID if d1_pin_ok else None
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
        "primary_estimand": camp["primary_estimand"],
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
        "calendar_hash_actual": cal["calendar_hash_actual"],
        "calendar_hash_expected": cal["calendar_hash_expected"],
        "calendar_identity_match": cal["calendar_identity_match"],
        "calendar_n_days": cal["calendar_n_days"],
        "sealed_oos": des["sealed_oos"],
        "status": des["status"],
        "execution_status": des.get("execution_status"),
        "frozen_before_empirical_execution": des.get("frozen_before_empirical_execution"),
        "launch_command": "python -m quant_edge_lab discovery campaign-v6 --execute",
        "d2_launch_command": "python -m quant_edge_lab discovery campaign-v6 --execute-d2",
        "d1_json_sha256": d1_sha,
        "d1_json_sha256_expected": FROZEN_D1_JSON_SHA256,
        "d1_pin_ok": d1_pin_ok,
        "d2_frozen_cell": FROZEN_CELL,
        "d2_selected_id": d1_selected,
        "d2_start": camp["splits"]["validation_D2"]["start"],
        "d2_end": camp["splits"]["validation_D2"]["end"],
        "d2_payloads_read": False,
        "d3_opened": False,
        "note": (
            "Default command does not run research. SIGNAL_ONLY. Sealed OOS closed. "
            "No D1/D2/D3 minute payloads were read. Calendar listed, payloads unread."
        ),
    }


def run_d1_execute(root: Path, des: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    """D1-only empirical walk. Caller must already have passed approval + calendar pin."""
    from quant_edge_lab.data.massive.flatfiles import local_parquet_path

    cal = v6_calendar_report(root, des)
    require_calendar_identity(cal)
    ident = freeze_v6(root)
    ident = {**ident, "calendar": cal["calendar_hash_actual"]}
    require_resume_identity(root, run_id, ident)
    cal_man = calendar_manifest(des)
    days = d1_loadable_days(cal["days"], cal_man)
    inst_p = instruments_path(root)
    if not inst_p.exists():
        raise RuntimeError("instruments.parquet missing; refuse execute")
    inst = pl.read_parquet(inst_p)
    d1_end = cal_man["splits"]["D1"]["end"]

    def load_day(day: str) -> pl.DataFrame | None:
        if day > d1_end:
            raise RuntimeError(f"D1 execute refuses to read {day}")
        p = local_parquet_path(root, day)
        if not p.exists():
            return None
        return pl.read_parquet(p)

    out = walk_d1(
        days,
        load_day,
        calendar=cal["days"],
        cal_man=cal_man,
        des=des,
        instruments=inst,
        emit_splits=frozenset({"D1"}),
    )
    payload = {
        **out,
        "identity": ident,
        "mode": "D1_ONLY",
        "sealed_oos": "inaccessible",
        "d2_opened": False,
        "d3_opened": False,
    }
    atomic_write_json(v6_run_dir(root, run_id) / "d1.json", payload)
    return payload


def execute_v6(root: Path, *, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    des = load_v6(root)
    assert_execution_approved(des)
    return run_d1_execute(root, des, run_id=run_id)


def run_d2_execute(
    root: Path,
    des: dict[str, Any],
    *,
    run_id: str,
    expected_d1_sha256: str = FROZEN_D1_JSON_SHA256,
    expected_identity: dict[str, str] | None = None,
) -> dict[str, Any]:
    """D2-only empirical walk of the D1-frozen cell. Never opens D3."""
    from quant_edge_lab.data.massive.flatfiles import local_parquet_path

    cal = v6_calendar_report(root, des)
    require_calendar_identity(cal)
    ident = freeze_v6(root)
    ident = {**ident, "calendar": cal["calendar_hash_actual"]}
    pin = load_d1_pin(
        v6_run_dir(root, run_id),
        expected_sha256=expected_d1_sha256,
        expected_identity=expected_identity or FROZEN_D1_IDENTITY,
        current_identity=ident,
    )
    cal_man = calendar_manifest(des)
    d2_end = cal_man["splits"]["D2"]["end"]
    inst_p = instruments_path(root)
    if not inst_p.exists():
        raise RuntimeError("instruments.parquet missing; refuse execute")
    inst = pl.read_parquet(inst_p)

    def load_day(day: str) -> pl.DataFrame | None:
        if day > d2_end:
            raise RuntimeError(f"D2 execute refuses to read {day}")
        p = local_parquet_path(root, day)
        if not p.exists():
            return None
        return pl.read_parquet(p)

    out = walk_d2(
        cal["days"],
        load_day,
        cal_man=cal_man,
        des=des,
        instruments=inst,
        cell=pin["frozen_cell"],
    )
    payload = {
        **out,
        "identity": ident,
        "d1_sha256": pin["d1_sha256"],
        "d1_selected_id": pin["selected_id"],
        "sealed_oos": "inaccessible",
    }
    atomic_write_json(v6_run_dir(root, run_id) / "d2.json", payload)
    return payload


def execute_v6_d2(root: Path, *, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    des = load_v6(root)
    assert_execution_approved(des)
    return run_d2_execute(root, des, run_id=run_id)


def run_campaign_v6(
    root: Path,
    *,
    execute: bool = False,
    execute_d2: bool = False,
    run_id: str = DEFAULT_RUN_ID,
) -> dict[str, Any]:
    if execute and execute_d2:
        raise RuntimeError("refuse: cannot combine --execute and --execute-d2")
    if execute_d2:
        return execute_v6_d2(root, run_id=run_id)
    if not execute:
        return readiness_v6(root)
    return execute_v6(root, run_id=run_id)
