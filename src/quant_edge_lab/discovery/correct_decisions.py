"""Re-score Stage-1 kill/pass using frozen variant directions. Does not rescan bars."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.catalog import family_by_id
from quant_edge_lab.discovery.frozen_direction import expected_direction_for_job
from quant_edge_lab.discovery.funnel import decide_stage
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json, knowledge_dir
from quant_edge_lab.discovery.runner import load_state, save_state

JOBS_TO_RESCORE = ("H007.v1", "H037.v1", "H040.v1")


def effective_stage1_decision(job: dict[str, Any]) -> str:
    if job.get("corrected_decision"):
        return str(job["corrected_decision"])
    return str(job.get("decision") or "")


def stage2_eligible_keys(state: dict[str, Any]) -> list[str]:
    keys = []
    for key, job in state.get("jobs", {}).items():
        if job.get("state") == "ERROR":
            continue
        if effective_stage1_decision(job) == "PASS":
            keys.append(key)
    return sorted(keys)


def correct_variant_direction_decisions(
    root: Path,
    batch_id: str,
    *,
    job_keys: tuple[str, ...] = JOBS_TO_RESCORE,
) -> dict[str, Any]:
    """Keep original decisions; write corrected PASS/KILL from frozen variant direction + stored means."""
    root = Path(root)
    state = load_state(root, batch_id)
    if not state.get("jobs"):
        raise FileNotFoundError(f"no state for {batch_id}")
    at = datetime.now(UTC).isoformat()
    audit: list[dict[str, Any]] = []
    for key in job_keys:
        job = state["jobs"][key]
        fid = job.get("family_id") or key.split(".")[0]
        vid = job.get("variant_id") or key
        fam = family_by_id(fid)
        frozen_dir = expected_direction_for_job(fid, vid, root=root)
        parent_dir = fam.expected_direction
        means = job.get("interim_means") or {}
        primary = means.get(fam.stage_criteria.primary_horizon)
        n_td = int(job.get("ticker_days") or len(job.get("tdays") or []))
        n_days = int(job.get("trading_days_with_events") or len(job.get("dayset") or []))
        top_share = job.get("concentration_top_ticker_share")
        new_dec, new_reason = decide_stage(
            expected_direction=frozen_dir,
            ticker_days=n_td,
            trading_days=n_days,
            mean_primary=primary,
            top_ticker_share=top_share,
            criteria=fam.stage_criteria,
            stage_complete=True,
        )
        original = {
            "decision": job.get("decision"),
            "decision_reason": job.get("decision_reason"),
            "state": job.get("state"),
            "expected_direction": parent_dir,
        }
        if "decision_as_recorded" not in job:
            job["decision_as_recorded"] = original["decision"]
            job["decision_reason_as_recorded"] = original["decision_reason"]
            job["state_as_recorded"] = original["state"]
        job["corrected_decision"] = new_dec
        job["corrected_decision_reason"] = new_reason
        job["corrected_state"] = (
            "PASSED_STAGE_1" if new_dec == "PASS" else "KILLED_STAGE_1" if new_dec == "KILL" else job.get("state")
        )
        job["direction_correction"] = {
            "at": at,
            "means_unchanged": True,
            "sample_unchanged": True,
            "original_expected_direction": parent_dir,
            "frozen_expected_direction": frozen_dir,
            "original_decision": original["decision"],
            "original_reason": original["decision_reason"],
            "corrected_decision": new_dec,
            "corrected_reason": new_reason,
            "note": "sign test only; frozen YAML not modified; bars not rescanned",
        }
        rec = {
            "type": "stage_result_direction_correction",
            "batch_id": batch_id,
            "family_id": fid,
            "variant_id": vid,
            "job_key": key,
            "stage": "fast",
            "original_decision": original["decision"],
            "original_reason": original["decision_reason"],
            "corrected_decision": new_dec,
            "corrected_reason": new_reason,
            "original_expected_direction": parent_dir,
            "frozen_expected_direction": frozen_dir,
            "events": job.get("events"),
            "ticker_days": n_td,
            "means": means,
            "means_unchanged": True,
            "note": "SIGNAL_ONLY; not an edge; original KILL kept in audit",
        }
        rec["recorded_at"] = at
        append_jsonl(knowledge_dir(root) / "index.jsonl", rec)
        audit.append(rec)
    state["direction_correction_at"] = at
    state["direction_correction_jobs"] = list(job_keys)
    save_state(root, state)
    summary_path = knowledge_dir(root) / "experiments" / f"{batch_id}-direction-correction.json"
    atomic_write_json(
        summary_path,
        {
            "batch_id": batch_id,
            "at": at,
            "jobs": audit,
            "stage2_eligible": stage2_eligible_keys(state),
        },
    )
    return {"state": state, "audit": audit, "stage2_eligible": stage2_eligible_keys(state)}
