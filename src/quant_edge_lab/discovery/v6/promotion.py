"""Frozen D1→D2→D3 H1-only promotion. Incremental is hierarchical, not BH."""

from __future__ import annotations

from typing import Any

from quant_edge_lab.discovery.v6.incremental import incremental_confirmatory_ok
from quant_edge_lab.discovery.v6.selection import sample_ok

H1_ID = "H1_HIGH_RETENTION_CONTINUATION"
LABEL_NOT_INCREMENTAL = "PRIMARY_SIGNAL_POSITIVE_BUT_RETENTION_NOT_INCREMENTAL"


def h1_survives_d2(
    h1: dict[str, Any],
    incremental: dict[str, Any],
    *,
    sample: dict[str, Any],
    bh_survivor: bool,
    economic_floor: float = 0.0003,
) -> tuple[bool, str | None]:
    if not sample_ok(h1, sample):
        return False, "sample_or_concentration"
    mean = h1.get("mean")
    if mean is None or float(mean) <= 0:
        return False, "direction_failed"
    if float(mean) < economic_floor:
        return False, "below_d2_floor"
    if bh_survivor is not True:
        return False, "bh_not_survivor"
    ok, why = incremental_confirmatory_ok(incremental)
    if not ok:
        return False, why
    return True, None


def d3_campaign_decision(
    h1: dict[str, Any],
    incremental: dict[str, Any],
    *,
    sample: dict[str, Any],
    bh_survivor: bool,
    d2_mean: float | None,
    d3_floor: float = 0.0012,
) -> dict[str, Any]:
    if not sample_ok(h1, sample):
        return {"label": "KILL", "reason": "sample_or_concentration"}
    mean = h1.get("mean")
    if mean is None or float(mean) <= 0:
        return {"label": "KILL", "reason": "direction_failed"}
    if d2_mean is None or float(d2_mean) <= 0:
        return {"label": "KILL", "reason": "sign_mismatch_d2"}
    if bh_survivor is not True:
        return {"label": "KILL", "reason": "d3_not_significant"}
    inc_ok, why = incremental_confirmatory_ok(incremental)
    if float(mean) >= d3_floor and not inc_ok:
        return {
            "label": LABEL_NOT_INCREMENTAL,
            "reason": why,
            "h3_cannot_win": True,
        }
    if not inc_ok:
        return {"label": "KILL", "reason": why}
    if float(mean) < d3_floor:
        return {
            "label": "VALIDATED_SUBTHRESHOLD_PHENOMENON",
            "reason": "below_d3_floor",
            "v6_success": False,
        }
    return {"label": "RESEARCH_PASS", "reason": "h1_and_incremental", "v6_success": True}


def refuse_open_d3(d2_h1_ok: bool) -> None:
    if not d2_h1_ok:
        raise RuntimeError("D3 refused: H1 did not survive all D2 primary gates")
