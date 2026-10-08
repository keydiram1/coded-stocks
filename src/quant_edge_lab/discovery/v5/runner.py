"""V5 orchestration. Default is readiness-only. --execute is required to research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v5.campaigns.close_dislocation import (
    CAMPAIGN_ID,
    apply_rule,
    build_events_from_days,
    freeze_rules_from_d1,
)
from quant_edge_lab.discovery.v5.checkpoint import refuse_recompute_if_complete, require_resume_identity, v5_run_dir, write_ckpt, write_state_table
from quant_edge_lab.discovery.v5.evaluation import evaluate_frozen, trial_count
from quant_edge_lab.discovery.v5.identity import ResumeIdentityError
from quant_edge_lab.discovery.v5.manifest import campaign_block, freeze_v5, load_v5
from quant_edge_lab.discovery.v5.models import TelemetrySnapshot
from quant_edge_lab.discovery.v5.partitions import SPLIT_ORDER, assert_no_future_split_in_estimation, assert_sealed_oos_closed, filter_days, split_for_day
from quant_edge_lab.discovery.v5.telemetry import StageClock, eta_seconds, percent, persist_snapshot

DEFAULT_RUN_ID = "v5-close-dislocation"


def readiness_v5(root: Path) -> dict[str, Any]:
    man, gates = load_v5(root)
    ident = freeze_v5(root)
    assert_sealed_oos_closed(man)
    camp = campaign_block(man)
    n_trials = trial_count(camp)
    return {
        "mode": "READINESS",
        "identity": ident,
        "git": ident["git"],
        "manifest_hash": ident["manifest"],
        "gates_hash": ident["gates"],
        "data_manifest_hash": ident["data_manifest"],
        "science": ident["science"],
        "campaigns": man["campaigns"],
        "primary_campaign": man["primary_campaign"],
        "partitions": man["splits"],
        "primary_outcome": camp["primary_outcome"],
        "trial_count": n_trials,
        "hypotheses": [h["hypothesis_id"] for h in camp["hypotheses"]],
        "sealed_oos": man["sealed_oos"],
        "status": man["status"],
        "reviewer_approval_required": camp.get("reviewer_approval_required", True),
        "launch_command": "python -m quant_edge_lab discovery campaign-v5 --execute",
        "note": "Default command does not run research. SIGNAL_ONLY. Sealed OOS closed.",
    }


def _days_from_bars(bars_by_day: dict[str, pl.DataFrame]) -> list[str]:
    return sorted(bars_by_day)


def run_on_bars(
    bars_by_day: dict[str, pl.DataFrame],
    man: dict[str, Any],
    gates: dict[str, Any],
    *,
    ident: dict[str, str],
    root: Path | None = None,
    run_id: str = DEFAULT_RUN_ID,
    persist: bool = False,
) -> dict[str, Any]:
    assert_sealed_oos_closed(man)
    camp = campaign_block(man)
    days = _days_from_bars(bars_by_day)
    ordered = sorted(days)
    clock = StageClock()

    def load_day(day: str) -> pl.DataFrame | None:
        return bars_by_day.get(day)

    def load_next(day: str) -> pl.DataFrame | None:
        i = ordered.index(day)
        if i + 1 >= len(ordered):
            return None
        return bars_by_day.get(ordered[i + 1])

    def progress(done: int, total: int, day: str, n_ev: int) -> None:
        if not persist or root is None:
            return
        persist_snapshot(
            root,
            run_id,
            TelemetrySnapshot(
                stage="EVENTS",
                substage="close_rows",
                current_unit=day,
                completed=done,
                total=total,
                percent_complete=percent(done, total),
                elapsed_s=clock.elapsed(),
                eta_s=eta_seconds(done, total, clock.elapsed()),
                events_generated=n_ev,
                checkpoint_state="OK",
                last_successful_checkpoint=day,
                sealed_oos="inaccessible",
            ),
            extra={"campaign_id": CAMPAIGN_ID},
        )

    if persist and root is not None:
        require_resume_identity(root, run_id, ident)

    events, rvol_hist = build_events_from_days(
        ordered,
        load_day,
        load_next,
        rvol_lookback=int(man["rvol"]["lookback_completed_days"]),
        min_rth_minutes=int((man.get("session") or {}).get("min_rth_minutes_full_day") or 380),
        progress=progress if persist else None,
    )
    d1_days = [d for d in ordered if split_for_day(d, man) == "D1"]
    assert_no_future_split_in_estimation("D1", d1_days, man)
    d1 = events.filter(pl.col("trading_date").is_in(d1_days)) if events.height else events
    if persist and root is not None and refuse_recompute_if_complete(root, run_id, "d1_rules", ident):
        from quant_edge_lab.discovery.v5.checkpoint import load_ckpt
        from quant_edge_lab.discovery.v5.models import CandidateRule

        rec = load_ckpt(root, run_id, "d1_rules", ident) or {}
        rules = [CandidateRule.model_validate(r) for r in rec.get("rules") or []]
    else:
        rules = freeze_rules_from_d1(d1, camp)
        if persist and root is not None:
            write_ckpt(root, run_id, "d1_rules", {"complete": True, "rules": [r.model_dump() for r in rules]}, ident)
            write_state_table(root, run_id, "events", events, ident)

    results: dict[str, Any] = {"rules": [r.model_dump() for r in rules], "trial_count": trial_count(camp)}
    d2_means: dict[str, float] = {}
    for split in SPLIT_ORDER:
        sub_days = filter_days(ordered, split, man)
        sub = events.filter(pl.col("trading_date").is_in(sub_days)) if events.height else events
        ev = evaluate_frozen(sub, rules, split=split, gates=gates, d2_means=d2_means)
        results[split] = ev
        if split == "D2":
            for row in ev:
                m = row["stats"].get("mean")
                if m is not None:
                    d2_means[row["rule"]["hypothesis_id"]] = m
        if persist and root is not None:
            write_ckpt(root, run_id, f"{split.lower()}_eval", {"complete": True, "rows": ev}, ident)
    unused = rvol_hist
    _ = unused
    return results


def execute_v5(root: Path, *, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    """Full campaign against on-disk minute parquet. Must not be called by default CLI."""
    from quant_edge_lab.data.massive.flatfiles import local_parquet_path
    from quant_edge_lab.discovery.campaign_v4 import research_days

    man, gates = load_v5(root)
    ident = freeze_v5(root)
    if ident["data_manifest"] == "MISSING":
        raise ResumeIdentityError("data manifest missing; refuse execute")
    require_resume_identity(root, run_id, ident)
    days = research_days(root)
    bars: dict[str, pl.DataFrame] = {}
    for d in days:
        p = local_parquet_path(root, d)
        if p.exists():
            bars[d] = pl.read_parquet(p)
    out = run_on_bars(bars, man, gates, ident=ident, root=root, run_id=run_id, persist=True)
    atomic_write_json(v5_run_dir(root, run_id) / "final.json", {**out, "identity": ident, "sealed_oos": "inaccessible"})
    return out


def run_campaign_v5(root: Path, *, execute: bool = False, run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    if not execute:
        return readiness_v5(root)
    return execute_v5(root, run_id=run_id)
