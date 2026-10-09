"""Quotes feasibility for local-RV. No returns, no hypothesis, no sealed OOS."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.options_quotes import (
    AUDIT_DAY,
    MAX_FULL_GET_BYTES,
    contrast_entitled_aggregates,
    list_quotes_inventory,
    probe_quotes_get,
    storage_plan,
)

UNOBSERVED = "NOT_OBSERVED_QUOTES_GET_FORBIDDEN"


def _unobserved(reason: str) -> dict[str, str]:
    return {"status": "NOT_OBSERVED", "reason": reason}


def timestamp_semantics_unknown() -> dict[str, Any]:
    return {
        "observed_columns": None,
        "clocks_observed": [],
        "causal_availability_clock": None,
        "uncertainty": (
            "quotes_v1 LIST succeeds but HEAD/GET return 403. "
            "No quote timestamp column was observed in a file body. "
            "Cannot assign value_time or first_available_at."
        ),
        "rule": "No signal may use a quote before first_available_at once a clock is observed.",
    }


def quote_quality_unobserved() -> dict[str, Any]:
    keys = (
        "total_quote_rows",
        "unique_contracts",
        "unique_underlyings",
        "bid_gt_0_fraction",
        "ask_gt_0_fraction",
        "ask_ge_bid_fraction",
        "locked_fraction",
        "crossed_fraction",
        "two_sided_fraction",
        "dollar_spread",
        "relative_spread",
        "quote_count_per_contract",
        "duplicate_contract_timestamps",
        "out_of_order_timestamp_frequency",
        "spread_by_mid_bucket",
    )
    return {k: UNOBSERVED for k in keys}


def synchronized_surface_unobserved() -> dict[str, Any]:
    return {
        "underlyings_inspected": ["AAPL", "SPY", "QQQ"],
        "purpose": "data_quality_only_not_hypothesis_selection",
        "can_reconstruct_synchronous_local_surface": None,
        "detail": _unobserved(
            "Need event quotes available at-or-before a clock. File body not readable."
        ),
        "quote_age_snapshots": _unobserved("no quotes loaded"),
    }


def underlying_sync_assessment() -> dict[str, Any]:
    return {
        "stock_data": "minute aggregates only",
        "option_quotes": "entitled for LIST, not for GET",
        "preferred_research_clock": "completed_minute_boundaries",
        "proposed_join_if_quotes_become_readable": {
            "option_snapshot": (
                "latest quote with first_available_at <= minute_boundary"
            ),
            "underlying": (
                "last completed stock minute bar whose first_available_at "
                "<= the same minute_boundary (no unfinished minute)"
            ),
            "implementable_in_code": "yes_once_quote_clock_is_observed",
            "validated_on_quotes": False,
        },
        "resolution_mismatch": (
            "Option quotes, if event-level, can be sub-second. Underlying is "
            "minute-level. That blocks sub-minute options RV. It does not "
            "necessarily block minute-clock RV."
        ),
    }


def own_iv_feasibility() -> dict[str, Any]:
    return {
        "vendor_iv_greeks": "do_not_use",
        "can_compute_own_iv_from_observed_quotes": False,
        "reason": "quotes file body not observed",
        "inputs_still_needed": [
            "option bid/ask mid from two-sided quotes",
            "causal first_available_at",
            "contemporaneous completed-minute underlying",
            "strike and expiration from OCC ticker",
            "call/put from OCC ticker",
            "risk_free_rate_source",
            "dividend_treatment",
            "american_exercise_treatment",
            "adjusted_contract_exclusion",
        ],
        "simplest_defensible_v1_if_quotes_arrive": {
            "rate": "constant_or_published_SOFR_asof_session_explicitly_lagged",
            "dividends": "zero_for_short_dated_equity_index_and_flag_as_assumption",
            "exercise": "european_on_unadjusted_short_dated_only_or_binomial_later",
            "price": "mid_of_valid_two_sided_quote_not_trade_close",
            "reliability": "unknown_until_quotes_are_readable",
        },
    }


def representation_recommendation() -> dict[str, Any]:
    return {
        "A_local_iv_residual": "target_IV - locally_fitted_same_expiry_IV_smile",
        "B_butterfly_convexity": "direct strike convexity using synchronized mids",
        "fewer_assumptions": "B",
        "more_robust_to_underlying_price_error": "B",
        "maps_better_to_tradeable_rv": "B_as_calendar_or_butterfly; A_for_smile_relative_value",
        "PRIMARY_for_V1": "B",
        "reason": (
            "Butterfly/convexity uses synchronized mids and strike geometry. "
            "It does not need a vol model, rate, or dividend. Local IV residual "
            "needs all of those plus a smile fit. Price-space linear interpolation "
            "of option prices is rejected: it confuses legitimate convexity with "
            "mispricing and, on trade-minute closes, is asynchronous."
        ),
        "rejected_primary": "trade_aggregate_price_close_linear_interpolation",
        "not_tested_on_forward_returns": True,
    }


def adjusted_contract_risk() -> dict[str, Any]:
    return {
        "observed_in_quotes_file": False,
        "flat_file_has_shares_per_contract": "unknown_not_observed",
        "conservative_exclusion_if_only_OCC_ticker": (
            "exclude roots containing digits; do not mix into a standard 100-share "
            "surface. This is a heuristic, not a proof of standard deliverable."
        ),
        "blocker": (
            "Cannot confirm adjusted vs standard from an unread quotes file. "
            "Need contract metadata (multiplier/deliverable) or a conservative "
            "root filter plus reference."
        ),
    }


def run_quotes_feasibility(root: Path) -> dict[str, Any]:
    inv = list_quotes_inventory(root)
    probe = probe_quotes_get(root, AUDIT_DAY)
    listed = inv.get("audit_day_compressed_bytes")
    if listed and int(listed) > MAX_FULL_GET_BYTES:
        probe["full_object_too_large"] = True
        probe["listed_compressed_bytes"] = int(listed)
    entitled = contrast_entitled_aggregates(root, AUDIT_DAY)
    get_ok = bool((probe.get("range_get") or {}).get("ok"))
    readiness = "READY_TO_DESIGN_OPTIONS_EXPERIMENT" if get_ok else "NOT_READY"
    if get_ok and probe.get("full_object_too_large"):
        readiness = "READY_WITH_LIMITATIONS"
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "science": "quotes_feasibility_not_a_hypothesis",
        "audit_day_requested": AUDIT_DAY,
        "inventory": inv,
        "entitlement_probe": probe,
        "other_opra_head": entitled,
        "schema": probe.get("schema_from_prefix"),
        "timestamp_availability": timestamp_semantics_unknown()
        if not get_ok
        else probe.get("schema_from_prefix"),
        "quote_quality": quote_quality_unobserved()
        if not get_ok
        else _unobserved("prefix-only; full-day quality not computed"),
        "synchronized_surface": synchronized_surface_unobserved(),
        "staleness": _unobserved("no quote ages without a quote clock"),
        "underlying_synchronization": underlying_sync_assessment(),
        "own_iv": own_iv_feasibility(),
        "representation": representation_recommendation(),
        "adjusted_contracts": adjusted_contract_risk(),
        "storage": storage_plan(inv.get("audit_day_compressed_bytes"), inv.get("size_bytes") or {}),
        "readiness": readiness,
        "blockers": [
            "quotes_v1 HEAD/GET HTTP 403 with current S3 keys (LIST is allowed)",
            "audit day compressed size ~116 GiB; full-day GET refused even if entitled",
            "trades_v1 also HEAD 403",
            "quote schema, clocks, quality, surface, and staleness not observed",
        ]
        if not get_ok
        else ["full_day_quotes_object_too_large_for_this_round"],
        "did_not": [
            "bulk_download_quotes",
            "run_options_hypothesis",
            "compute_returns_or_convergence",
            "open_sealed_oos",
            "use_trade_minute_price_space_as_primary",
        ],
    }
    out_dir = Paths(root).derived / "discovery" / "v5" / "v5-options-quotes-feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "feasibility.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written"] = str(dest)
    return report
