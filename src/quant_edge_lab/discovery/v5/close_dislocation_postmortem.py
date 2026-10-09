"""Descriptive equal-day D1/D2 recompute. Does not change campaign decisions or load D3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule
from quant_edge_lab.discovery.v5.continuation import (
    CAMPAIGN_ID,
    D3,
    SOURCE_CAMPAIGN,
    SOURCE_IDENTITY,
    attach_continuation,
    freeze_continuation,
    frozen_rules,
    load_continuation,
    load_pre_d3_event_partitions,
)
from quant_edge_lab.discovery.v5.continuation_stats import equal_weight_day_stats
from quant_edge_lab.discovery.v5.evaluation import day_block_stats
from quant_edge_lab.hashing import sha256_file

D1 = ("2021-10-29", "2023-10-17")
D2 = ("2023-10-18", "2024-10-10")
HIDS = ("H1_PRIMARY", "H2_ABS_Q95", "H3_DISLOCATION_ONLY")

EXISTING_D3_MEAN_BP = {
    "H1_PRIMARY": -1.1968567947566027,
    "H2_ABS_Q95": -0.8608738226450076,
    "H3_DISLOCATION_ONLY": -1.8223760003603973,
    "H4_Z2_RVOL": -2.4432018905369812,
    "H5_Z2_ONLY": -2.1024154838464537,
}


def _split_frame(events: pl.DataFrame, bounds: tuple[str, str]) -> pl.DataFrame:
    lo, hi = bounds
    return events.filter((pl.col("trading_date") >= lo) & (pl.col("trading_date") <= hi))


def _stats_for(events: pl.DataFrame) -> dict[str, Any]:
    eq = equal_weight_day_stats(events, col="primary_signed", n_boot=20, seed=42)
    eq.pop("daily_means", None)
    ev = day_block_stats(events, col="primary_signed", n_boot=20, seed=42)
    ev.pop("daily_means", None)
    return {
        "equal_weight_trading_day_mean_bp": eq.get("mean_bp"),
        "equal_weight_trading_day_mean": eq.get("mean"),
        "equal_day_trading_days": eq.get("trading_days"),
        "equal_day_events": eq.get("events"),
        "event_weighted_mean_bp": ev.get("mean_bp"),
        "event_weighted_mean": ev.get("mean"),
    }


def run_close_dislocation_postmortem(root: Path) -> dict[str, Any]:
    """Load D1/D2 event partitions only. Do not load D3 parquet."""
    man, gates = load_continuation(root)
    ident = freeze_continuation(root)
    rules = {r.hypothesis_id: r for r in frozen_rules(man[man["primary_campaign"]])}
    pre = load_pre_d3_event_partitions(root)
    pre = attach_continuation(pre)
    by_h: dict[str, Any] = {}
    for hid in HIDS:
        rule = rules[hid]
        hit = apply_rule(pre, rule)
        d1 = _stats_for(_split_frame(hit, D1))
        d2 = _stats_for(_split_frame(hit, D2))
        eq1 = d1["equal_weight_trading_day_mean_bp"]
        eq2 = d2["equal_weight_trading_day_mean_bp"]
        d3 = EXISTING_D3_MEAN_BP[hid]
        by_h[hid] = {
            "role": rule.role,
            "D1_equal_day_mean_bp": eq1,
            "D2_equal_day_mean_bp": eq2,
            "D3_equal_day_mean_bp_existing": d3,
            "D1_event_weighted_mean_bp": d1["event_weighted_mean_bp"],
            "D2_event_weighted_mean_bp": d2["event_weighted_mean_bp"],
            "apparent_continuation_under_equal_day": bool(
                eq1 is not None and eq2 is not None and eq1 > 0 and eq2 > 0
            ),
            "sample": {
                "D1": {
                    k: d1[k] for k in d1 if k.endswith("events") or "trading_days" in k
                },
                "D2": {
                    k: d2[k] for k in d2 if k.endswith("events") or "trading_days" in k
                },
            },
        }
    from quant_edge_lab.discovery.v5.manifest import GATES_REL as RG
    from quant_edge_lab.discovery.v5.manifest import MANIFEST_REL as RM

    report = {
        "family_status": "CLOSED_NO_EDGE",
        "purpose": "DESCRIPTIVE_POST_MORTEM_ONLY",
        "does_not": [
            "alter_failed_campaign",
            "reopen_hypothesis_selection",
            "change_any_decision",
            "generate_new_close_dislocation_candidate",
            "use_D3_to_select_conditions",
        ],
        "reversal": {
            "campaign": SOURCE_CAMPAIGN,
            "freeze_git": SOURCE_IDENTITY["git"],
            "manifest": SOURCE_IDENTITY["manifest"],
            "gates": SOURCE_IDENTITY["gates"],
            "data_manifest": SOURCE_IDENTITY["data_manifest"],
            "calendar": SOURCE_IDENTITY["calendar"],
            "file_hashes_match": sha256_file(root / RM) == SOURCE_IDENTITY["manifest"]
            and sha256_file(root / RG) == SOURCE_IDENTITY["gates"],
            "outcome": "FAILED",
            "d3_evaluated": False,
        },
        "continuation": {
            "campaign": CAMPAIGN_ID,
            "freeze_git": ident["git"],
            "manifest": ident["manifest"],
            "gates": ident["gates"],
            "data_manifest": ident["data_manifest"],
            "execution_status": man.get("execution_status"),
            "sealed_oos": man.get("sealed_oos"),
            "d3_opened_once": True,
            "d3_bounds": list(D3),
            "hypothesis_count": 5,
            "outcome": "FAILED",
            "h1_h5": "ALL_KILL",
            "bh_survivor": False,
            "research_pass": False,
            "validated_subthreshold_phenomenon": False,
            "primary_estimand_d3": "equal_weight_trading_day_mean",
        },
        "method_note": (
            "D1/D2 published continuation evidence used the older event-weighted "
            "reporting path. Frozen continuation D3 used equal_weight_trading_day_mean. "
            "This file recomputes D1/D2 on that same equal-day estimand. D3 mean_bp "
            "values are copied from the frozen continuation final.json and were not recomputed."
        ),
        "equal_day_d1_d2": by_h,
        "d3_existing_mean_bp": EXISTING_D3_MEAN_BP,
        "sealed_oos": "inaccessible",
    }
    out_dir = Paths(root).derived / "discovery" / "v5" / "v5-close-dislocation-family-closed"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "postmortem.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written"] = str(dest)
    return report
