"""CLOSE_DISLOCATION_CONTINUATION_V1. Default is readiness-only. D3 payload stays sealed."""

from __future__ import annotations

from typing import Any

import polars as pl

from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v5.campaigns.residual_scale import attach_normalized_dislocation
from quant_edge_lab.discovery.v5.continuation import (
    CAMPAIGN_ID,
    D3,
    DEFAULT_RUN_ID,
    SCALE_LOOKBACK,
    SCIENCE_ID,
    SOURCE_CAMPAIGN,
    SOURCE_IDENTITY,
    SOURCE_RUN_ID,
    Z_ABS,
    assert_continuation_negates_reversal,
    assert_execution_approved,
    assert_frozen_thresholds,
    assert_source_identity,
    attach_continuation,
    campaign_block,
    freeze_continuation,
    frozen_rules,
    load_continuation,
    load_continuation_source_events,
    new_run_dir,
    refuse_discovery_split,
)
from quant_edge_lab.discovery.v5.evaluation import day_block_stats, evaluate_frozen, trial_count
from quant_edge_lab.discovery.v5.models import CandidateRule
from quant_edge_lab.discovery.v5.partitions import assert_sealed_oos_closed

DIAGNOSTIC_HORIZONS = ("next_open_to_5m", "next_open_to_30m", "close_to_next_open")
DISCOVERY_PROVENANCE = {
    "note": (
        "The continuation hypothesis arose because the preregistered reversal "
        "hypothesis produced the opposite sign in D1 and D2. Those splits are "
        "discovery, not confirmation. The volatility-normalization concern was "
        "identified before D3 inspection; H4/H5 were preregistered before D3 was "
        "opened. D1/D2 must not choose or optimize z=2.0. H1 remains primary. "
        "H4/H5 are robustness/mechanism-cleanliness tests, not promoted primary rules."
    ),
    "reversal_campaign_published_primary_signed_mean_bp": {
        "H1_PRIMARY": {"D1": -6.587840, "D2": -9.848435},
        "H2_ABS_Q95": {"D1": -8.380926, "D2": -15.416081},
        "H3_DISLOCATION_ONLY": {"D1": -6.726542, "D2": -7.481214},
    },
}


def readiness_v5_continuation(root) -> dict[str, Any]:
    man, gates = load_continuation(root)
    ident = freeze_continuation(root)
    assert_sealed_oos_closed(man)
    camp = campaign_block(man)
    rules = frozen_rules(camp)
    assert_frozen_thresholds(rules)
    return {
        "mode": "READINESS",
        "campaign_id": man["campaign_id"],
        "science_id": man["science_id"],
        "source_campaign": SOURCE_CAMPAIGN,
        "source_run_identity": dict(man["source_run"]),
        "required_source_identity": SOURCE_IDENTITY,
        "identity": ident,
        "git": ident["git"],
        "manifest_hash": ident["manifest"],
        "gates_hash": ident["gates"],
        "data_manifest_hash": ident["data_manifest"],
        "science": ident["science"],
        "campaigns": man["campaigns"],
        "primary_campaign": man["primary_campaign"],
        "partitions": man["splits"],
        "evaluable_split": "D3",
        "discovery_splits_not_confirmation": ["D1", "D2"],
        "primary_outcome": camp["primary_outcome"],
        "economic_floor_bp": gates["d3"]["signed_mean_floor_bp"],
        "trial_count": trial_count(camp),
        "hypotheses": [h["hypothesis_id"] for h in camp["hypotheses"]],
        "h1_role": next(
            h["role"] for h in camp["hypotheses"] if h["hypothesis_id"] == "H1_PRIMARY"
        ),
        "absolute_family_trial_count": 3,
        "normalized_family_trial_count": 2,
        "z_threshold_abs": Z_ABS,
        "scale_lookback_sessions": SCALE_LOOKBACK,
        "normalized_scale_type": "same_clock_residual_std",
        "bh_q": gates["inference"]["bh_q"],
        "d3_start": man["splits"]["D3"]["start"],
        "d3_end": man["splits"]["D3"]["end"],
        "sealed_oos": man["sealed_oos"],
        "status": man["status"],
        "execution_status": man.get("execution_status") or gates.get("execution_status"),
        "reviewer_approval_required": True,
        "launch_command": "python -m quant_edge_lab discovery campaign-v5-continuation --execute",
        "note": (
            "Default command does not run confirmation. SIGNAL_ONLY. "
            "D3 payload is not loaded. Sealed OOS closed. execution_status=NOT_APPROVED."
        ),
    }


def evaluate_continuation_d3(
    events: pl.DataFrame,
    rules: list[CandidateRule],
    gates: dict[str, Any],
    *,
    split: str = "D3",
) -> list[dict[str, Any]]:
    refuse_discovery_split(split)
    if events.height and "trading_date" in events.columns:
        events = events.filter(
            (pl.col("trading_date") >= D3[0]) & (pl.col("trading_date") <= D3[1])
        )
    ids = {r.hypothesis_id for r in rules}
    return evaluate_frozen(
        events,
        rules,
        split="D3",
        gates=gates,
        d2_survivors=ids,
        preregistered_trial_count=len(rules),
    )


def _public_hypothesis(row: dict[str, Any]) -> dict[str, Any]:
    st = row["stats"]
    dec = row["decision"]
    return {
        "hypothesis_id": row["rule"]["hypothesis_id"],
        "role": row["rule"]["role"],
        "events": st.get("events"),
        "ticker_days": st.get("ticker_days"),
        "trading_days": st.get("trading_days"),
        "tickers": st.get("tickers"),
        "top_ticker_share": st.get("top_ticker_share"),
        "mean": st.get("mean"),
        "mean_bp": st.get("mean_bp"),
        "median": st.get("median"),
        "win_rate": st.get("win_rate"),
        "se": st.get("se"),
        "ci": st.get("ci"),
        "p_one_sided": st.get("p_one_sided"),
        "bh_rejected": st.get("bh_rejected"),
        "bh_q": st.get("bh_q"),
        "decision_label": dec.get("label"),
        "decision_reason": dec.get("reason"),
    }


def diagnostic_horizons(events: pl.DataFrame) -> dict[str, Any]:
    if events.height and "trading_date" in events.columns:
        events = events.filter(
            (pl.col("trading_date") >= D3[0]) & (pl.col("trading_date") <= D3[1])
        )
    out: dict[str, Any] = {}
    for col in DIAGNOSTIC_HORIZONS:
        if col not in events.columns:
            continue
        signed_name = f"continuation_signed_{col}"
        work = events
        if signed_name not in work.columns:
            from quant_edge_lab.discovery.v5.continuation import continuation_signed

            fwd = work[col].to_list()
            disc = work["discrepancy"].to_list()
            work = work.with_columns(
                pl.Series(
                    signed_name,
                    [continuation_signed(f, d) for f, d in zip(fwd, disc, strict=True)],
                    dtype=pl.Float64,
                )
            )
        st = day_block_stats(work, col=signed_name, n_boot=20, seed=42)
        st.pop("daily_means", None)
        out[col] = st
    return out


def execute_v5_continuation(
    root,
    *,
    run_id: str = DEFAULT_RUN_ID,
    events: pl.DataFrame | None = None,
) -> dict[str, Any]:
    if run_id == SOURCE_RUN_ID:
        raise RuntimeError("refusing to write into frozen reversal run directory")
    man, gates = load_continuation(root)
    assert_execution_approved(man, gates)
    assert_sealed_oos_closed(man)
    camp = campaign_block(man)
    rules = frozen_rules(camp)
    assert_frozen_thresholds(rules)
    source = assert_source_identity(root)
    if events is None:
        events = load_continuation_source_events(root)
    events = attach_normalized_dislocation(events)
    events = attach_continuation(events)
    assert_continuation_negates_reversal(events)
    rows = evaluate_continuation_d3(events, rules, gates, split="D3")
    public = [_public_hypothesis(r) for r in rows]
    ident = freeze_continuation(root)
    report = {
        "mode": "CONFIRMATION",
        "source_campaign": SOURCE_CAMPAIGN,
        "source_run_identity": source,
        "campaign_id": man["campaign_id"],
        "science_id": SCIENCE_ID,
        "campaign": CAMPAIGN_ID,
        "rules": [r.model_dump() for r in rules],
        "trial_count": trial_count(camp),
        "tested_this_stage": len(rows),
        "D3": public,
        "primary_outcome": camp["primary_outcome"],
        "economic_floor_bp": gates["d3"]["signed_mean_floor_bp"],
        "sealed_oos": "inaccessible",
        "status": man["status"],
        "execution_status": man.get("execution_status"),
        "diagnostic_horizons": diagnostic_horizons(events),
        "provenance": DISCOVERY_PROVENANCE,
        "identity": ident,
        "absolute_family_trial_count": 3,
        "normalized_family_trial_count": 2,
        "z_threshold_abs": Z_ABS,
        "scale_lookback_sessions": SCALE_LOOKBACK,
        "note": (
            "D3 only. D1/D2 were not evaluated as confirmation. "
            "H4/H5 are a separate residual-z family. Secondary horizons are diagnostic only."
        ),
    }
    out_dir = new_run_dir(root, run_id)
    atomic_write_json(out_dir / "final.json", report)
    return report


def run_campaign_v5_continuation(root, *, execute: bool = False, events=None) -> dict[str, Any]:
    if not execute:
        return readiness_v5_continuation(root)
    return execute_v5_continuation(root, events=events)
