"""V5 orchestration. Default is readiness-only. --execute is required to research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v5.campaigns.close_dislocation import (
    CAMPAIGN_ID,
    build_events_from_days,
    freeze_rules_from_d1,
)
from quant_edge_lab.discovery.v5.checkpoint import (
    DayStore,
    refuse_recompute_if_complete,
    require_resume_identity,
    v5_run_dir,
    write_ckpt,
)
from quant_edge_lab.discovery.v5.evaluation import d2_survivor_ids, evaluate_frozen, trial_count
from quant_edge_lab.discovery.v5.identity import ResumeIdentityError
from quant_edge_lab.discovery.v5.manifest import campaign_block, freeze_v5, load_v5
from quant_edge_lab.discovery.v5.models import CandidateRule, TelemetrySnapshot
from quant_edge_lab.discovery.v5.partitions import (
    SPLIT_ORDER,
    assert_no_future_split_in_estimation,
    assert_sealed_oos_closed,
    filter_days,
    split_for_day,
)
from quant_edge_lab.discovery.v5.telemetry import StageClock, eta_seconds, percent, persist_snapshot

DEFAULT_RUN_ID = "v5-close-dislocation"
APPROVED = frozenset({"APPROVED", "FROZEN"})


def assert_execution_approved(man: dict[str, Any], gates: dict[str, Any]) -> None:
    ms = str(man.get("execution_status") or "NOT_APPROVED")
    gs = str(gates.get("execution_status") or "NOT_APPROVED")
    if ms not in APPROVED or gs not in APPROVED:
        raise RuntimeError(
            f"execute_v5 refused: manifest.execution_status={ms} gates.execution_status={gs} "
            "(both must be APPROVED or FROZEN)."
        )


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
        "execution_status": man.get("execution_status") or gates.get("execution_status"),
        "reviewer_approval_required": camp.get("reviewer_approval_required", True),
        "launch_command": "python -m quant_edge_lab discovery campaign-v5 --execute",
        "note": "Default command does not run research. SIGNAL_ONLY. Sealed OOS closed.",
    }


def _days_from_bars(bars_by_day: dict[str, pl.DataFrame]) -> list[str]:
    return sorted(bars_by_day)


def _hypothesis_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for row in rows:
        hid = row["rule"]["hypothesis_id"]
        st = row["stats"]
        dec = row["decision"]
        ev = st.get("events")
        bp = st.get("mean_bp")
        bps = f"{bp:+.1f}bp" if isinstance(bp, int | float) else "n/a"
        lines.append(f"{hid} events={ev} effect={bps} status={dec.get('label')}")
    return lines


def run_on_bars(
    bars_by_day: dict[str, pl.DataFrame],
    man: dict[str, Any],
    gates: dict[str, Any],
    *,
    ident: dict[str, str],
    root: Path | None = None,
    run_id: str = DEFAULT_RUN_ID,
    persist: bool = False,
    interrupt_after: int | None = None,
) -> dict[str, Any]:
    assert_sealed_oos_closed(man)
    camp = campaign_block(man)
    ordered = _days_from_bars(bars_by_day)
    clock = StageClock()
    store = DayStore(root, run_id, ident) if persist and root is not None else None

    def load_day(day: str) -> pl.DataFrame | None:
        return bars_by_day.get(day)

    def load_next(day: str) -> pl.DataFrame | None:
        i = ordered.index(day)
        if i + 1 >= len(ordered):
            return None
        return bars_by_day.get(ordered[i + 1])

    def progress(
        done, total, day, n_ev, *, cumulative_events=0, cumulative_rows=0, checkpoint_ok=False
    ):
        if not persist or root is None:
            return
        sp = split_for_day(day, man)
        persist_snapshot(
            root,
            run_id,
            TelemetrySnapshot(
                stage="EVENTS",
                substage=sp,
                current_unit=day,
                completed=done,
                total=total,
                percent_complete=percent(done, total),
                elapsed_s=clock.elapsed(),
                eta_s=eta_seconds(done, total, clock.elapsed()),
                rows_processed=cumulative_rows,
                events_generated=cumulative_events,
                checkpoint_state="OK" if checkpoint_ok else "pending",
                last_successful_checkpoint=day if checkpoint_ok else None,
                sealed_oos="inaccessible",
            ),
            extra={"campaign_id": CAMPAIGN_ID, "current_split": sp},
        )

    events, _rvol = build_events_from_days(
        ordered,
        load_day,
        load_next,
        rvol_lookback=int(man["rvol"]["lookback_completed_days"]),
        min_rth_minutes=int((man.get("session") or {}).get("min_rth_minutes_full_day") or 380),
        progress=progress if persist else None,
        store=store,
        interrupt_after=interrupt_after,
    )
    return _finish_splits(events, ordered, man, gates, camp, ident, root, run_id, persist, clock)


def _finish_splits(
    events: pl.DataFrame,
    ordered: list[str],
    man: dict[str, Any],
    gates: dict[str, Any],
    camp: dict[str, Any],
    ident: dict[str, str],
    root: Path | None,
    run_id: str,
    persist: bool,
    clock: StageClock,
) -> dict[str, Any]:
    d1_days = [d for d in ordered if split_for_day(d, man) == "D1"]
    assert_no_future_split_in_estimation("D1", d1_days, man)
    d1 = events.filter(pl.col("trading_date").is_in(d1_days)) if events.height else events
    if (
        persist
        and root is not None
        and refuse_recompute_if_complete(root, run_id, "d1_rules", ident)
    ):
        from quant_edge_lab.discovery.v5.checkpoint import load_ckpt

        rec = load_ckpt(root, run_id, "d1_rules", ident) or {}
        rules = [CandidateRule.model_validate(r) for r in rec.get("rules") or []]
    else:
        rules = freeze_rules_from_d1(d1, camp)
        if persist and root is not None:
            write_ckpt(
                root,
                run_id,
                "d1_rules",
                {"complete": True, "rules": [r.model_dump() for r in rules]},
                ident,
            )

    results: dict[str, Any] = {
        "rules": [r.model_dump() for r in rules],
        "trial_count": trial_count(camp),
        "events": events.height,
    }
    d2_means: dict[str, float] = {}
    survivors: set[str] = set()
    for split in SPLIT_ORDER:
        if (
            persist
            and root is not None
            and refuse_recompute_if_complete(root, run_id, f"{split.lower()}_eval", ident)
        ):
            from quant_edge_lab.discovery.v5.checkpoint import load_ckpt

            rec = load_ckpt(root, run_id, f"{split.lower()}_eval", ident) or {}
            ev = rec.get("rows") or []
        else:
            sub_days = filter_days(ordered, split, man)
            sub = events.filter(pl.col("trading_date").is_in(sub_days)) if events.height else events
            ev = evaluate_frozen(
                sub,
                rules,
                split=split,
                gates=gates,
                d2_means=d2_means,
                d2_survivors=survivors if split == "D3" else None,
                preregistered_trial_count=trial_count(camp),
            )
            if persist and root is not None:
                write_ckpt(
                    root, run_id, f"{split.lower()}_eval", {"complete": True, "rows": ev}, ident
                )
        results[split] = ev
        if split == "D2":
            survivors = d2_survivor_ids(ev)
            for row in ev:
                m = row["stats"].get("mean")
                if m is not None:
                    d2_means[row["rule"]["hypothesis_id"]] = m
        if persist and root is not None:
            stage = split
            persist_snapshot(
                root,
                run_id,
                TelemetrySnapshot(
                    stage=stage,
                    completed=len(ordered),
                    total=len(ordered),
                    percent_complete=100.0,
                    elapsed_s=clock.elapsed(),
                    events_generated=int(events.height) if events.height else 0,
                    checkpoint_state="OK",
                    last_successful_checkpoint=f"{split}_eval",
                    hypothesis_lines=_hypothesis_lines(ev),
                    allowed_split_mean_effect_bp=(ev[0]["stats"].get("mean_bp") if ev else None),
                    sealed_oos="inaccessible",
                ),
                extra={"campaign_id": CAMPAIGN_ID, "current_split": split},
            )
    return results


def execute_v5(
    root: Path, *, run_id: str = DEFAULT_RUN_ID, interrupt_after: int | None = None
) -> dict[str, Any]:
    """Stream one parquet day at a time. Must not be called by default CLI."""
    from quant_edge_lab.data.massive.flatfiles import local_parquet_path
    from quant_edge_lab.discovery.campaign_v4 import research_days
    from quant_edge_lab.discovery.v5.eligibility import cfg_from_manifest, instruments_path
    from quant_edge_lab.discovery.v5.preflight import (
        require_next_session_file,
        validate_research_calendar,
    )

    man, gates = load_v5(root)
    assert_execution_approved(man, gates)
    inst_p = instruments_path(root)
    if not inst_p.exists():
        raise ResumeIdentityError("instruments.parquet missing; refuse execute")
    inst = pl.read_parquet(inst_p)
    ident = freeze_v5(root)
    if ident["data_manifest"] == "MISSING" or ident.get("instruments") == "MISSING":
        raise ResumeIdentityError("data or instruments identity missing; refuse execute")
    days = research_days(root)

    def exists(d: str) -> bool:
        return local_parquet_path(root, d).exists()

    pf = validate_research_calendar(days, man, parquet_exists=exists)
    ident = {**ident, "calendar": pf["calendar_hash"]}
    require_resume_identity(root, run_id, ident)
    clock = StageClock()
    store = DayStore(root, run_id, ident)
    camp = campaign_block(man)

    def load_day(day: str) -> pl.DataFrame | None:
        p = local_parquet_path(root, day)
        if not p.exists():
            return None
        return pl.read_parquet(p)

    def load_next(day: str) -> pl.DataFrame | None:
        nxt = require_next_session_file(day, days, exists)
        if nxt is None:
            return None
        return pl.read_parquet(local_parquet_path(root, nxt))

    def progress(
        done, total, day, n_ev, *, cumulative_events=0, cumulative_rows=0, checkpoint_ok=False
    ):
        persist_snapshot(
            root,
            run_id,
            TelemetrySnapshot(
                stage="EVENTS",
                substage=split_for_day(day, man),
                current_unit=day,
                completed=done,
                total=total,
                percent_complete=percent(done, total),
                elapsed_s=clock.elapsed(),
                eta_s=eta_seconds(done, total, clock.elapsed()),
                rows_processed=cumulative_rows,
                events_generated=cumulative_events,
                checkpoint_state="OK" if checkpoint_ok else "pending",
                last_successful_checkpoint=day if checkpoint_ok else None,
                sealed_oos="inaccessible",
            ),
            extra={"campaign_id": CAMPAIGN_ID},
        )

    events, _rvol = build_events_from_days(
        days,
        load_day,
        load_next,
        rvol_lookback=int(man["rvol"]["lookback_completed_days"]),
        min_rth_minutes=int((man.get("session") or {}).get("min_rth_minutes_full_day") or 380),
        progress=progress,
        store=store,
        interrupt_after=interrupt_after,
        instruments=inst,
        elig_cfg=cfg_from_manifest(man),
    )
    out = _finish_splits(events, days, man, gates, camp, ident, root, run_id, True, clock)
    atomic_write_json(
        v5_run_dir(root, run_id) / "final.json",
        {**out, "identity": ident, "sealed_oos": "inaccessible"},
    )
    return out


def run_campaign_v5(
    root: Path, *, execute: bool = False, run_id: str = DEFAULT_RUN_ID
) -> dict[str, Any]:
    if not execute:
        return readiness_v5(root)
    return execute_v5(root, run_id=run_id)
