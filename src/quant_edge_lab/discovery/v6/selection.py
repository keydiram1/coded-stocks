"""D1 cell eligibility and freeze. Discovery only; no incremental p required."""

from __future__ import annotations

from typing import Any

from quant_edge_lab.discovery.v6.incremental import incremental_d1_eligible


def sample_ok(stats: dict[str, Any], sample: dict[str, Any]) -> bool:
    if (
        int(stats.get("ticker_days") or 0) < int(sample["min_ticker_days"])
        or int(stats.get("trading_days") or 0) < int(sample["min_trading_days"])
        or int(stats.get("tickers") or 0) < int(sample["min_tickers"])
    ):
        return False
    share = stats.get("top_ticker_share")
    if share is not None and float(share) > float(sample["max_top_ticker_share"]):
        return False
    wr = stats.get("win_rate")
    if wr is not None and float(wr) < float(sample["min_win_rate"]):
        return False
    return True


def cell_eligible_d1(
    h1_stats: dict[str, Any],
    incremental: dict[str, Any],
    sample: dict[str, Any],
) -> bool:
    mean = h1_stats.get("mean")
    if mean is None or float(mean) <= 0:
        return False
    if not sample_ok(h1_stats, sample):
        return False
    return incremental_d1_eligible(incremental)


def select_d1_cell(cell_rows: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for row in cell_rows:
        if cell_eligible_d1(row["h1_stats"], row["incremental"], sample):
            eligible.append(row)
    if not eligible:
        return {"decision": "KILL_AT_D1", "frozen_cell": None, "eligible_ids": []}
    best = max(eligible, key=lambda r: float(r["h1_stats"]["mean"]))
    return {
        "decision": "FREEZE",
        "frozen_cell": best["cell"],
        "eligible_ids": [r["cell"]["id"] for r in eligible],
        "selected_id": best["cell"]["id"],
    }
