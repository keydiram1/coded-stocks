"""Equal-day H1/H2/H3 stats plus hierarchical incremental gate."""

from __future__ import annotations

from typing import Any

import polars as pl

from quant_edge_lab.discovery.v5.continuation_stats import ESTIMAND, equal_weight_day_stats
from quant_edge_lab.discovery.v6.events import filter_hypothesis, incremental_frames
from quant_edge_lab.discovery.v6.incremental import incremental_stats, matched_day_deltas
from quant_edge_lab.statistics.summarize import benjamini_hochberg

HIDS = (
    "H1_HIGH_RETENTION_CONTINUATION",
    "H2_LOW_RETENTION_REVERSAL",
    "H3_IMPULSE_ONLY_CONTROL",
)


def cell_incremental(events: pl.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    treated, control = incremental_frames(events)
    deltas = matched_day_deltas(treated, control)
    return incremental_stats(deltas, n_boot=n_boot, seed=seed)


def evaluate_hypotheses(
    events: pl.DataFrame, *, n_boot: int = 1000, seed: int = 42, bh_q: float = 0.10
) -> dict[str, Any]:
    rows = []
    pvals = []
    for hid in HIDS:
        sub = filter_hypothesis(events, hid)
        st = equal_weight_day_stats(sub, n_boot=n_boot, seed=seed)
        if st.get("primary_estimand") != ESTIMAND:
            raise AssertionError("V6 hypotheses must use equal_weight_trading_day_mean")
        pvals.append(float(st["p_one_sided"]))
        rows.append({"hypothesis_id": hid, "stats": st})
    bh = benjamini_hochberg(pvals, q=bh_q)
    for rec, b in zip(rows, bh, strict=True):
        rec["bh_rejected"] = bool(b["bh_rejected"])
        rec["stats"]["bh_rejected"] = bool(b["bh_rejected"])
    h1 = next(r for r in rows if r["hypothesis_id"] == HIDS[0])
    return {
        "hypotheses": rows,
        "h1_stats": h1["stats"],
        "h1_bh_survivor": h1["bh_rejected"],
        "incremental": cell_incremental(events, n_boot=n_boot, seed=seed),
    }
