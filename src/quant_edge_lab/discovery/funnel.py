"""Pre-registered kill/promote. Criteria must exist before the stage runs."""

from __future__ import annotations

from typing import Any

from quant_edge_lab.discovery.models import StageCriteria


def decide_stage(
    *,
    expected_direction: str,
    ticker_days: int,
    trading_days: int,
    mean_primary: float | None,
    top_ticker_share: float | None,
    criteria: StageCriteria,
    stage_complete: bool,
) -> tuple[str, str]:
    """Return (PASS|KILL|CONTINUE, reason). CONTINUE only if stage not complete."""
    if not stage_complete:
        return "CONTINUE", "stage incomplete"
    if ticker_days < criteria.min_ticker_days or trading_days < criteria.min_trading_days:
        return "KILL", (
            f"insufficient clustered sample ticker_days={ticker_days} "
            f"trading_days={trading_days} (min {criteria.min_ticker_days}/{criteria.min_trading_days})"
        )
    if top_ticker_share is not None and top_ticker_share > criteria.max_top_ticker_share:
        return "KILL", f"concentration top_ticker_share={top_ticker_share:.2f} > {criteria.max_top_ticker_share}"
    if mean_primary is None:
        return "KILL", "no primary-horizon returns"
    if criteria.kill_if_sign_disagrees and expected_direction in {"long", "short"}:
        bad = (expected_direction == "long" and mean_primary < 0) or (
            expected_direction == "short" and mean_primary > 0
        )
        if bad:
            return "KILL", (
                f"sign disagrees with expected_direction={expected_direction} "
                f"mean_{criteria.primary_horizon}={mean_primary}"
            )
    if criteria.min_abs_mean_to_kill_as_too_small is not None:
        if abs(mean_primary) < criteria.min_abs_mean_to_kill_as_too_small:
            return "KILL", f"abs mean {mean_primary} below {criteria.min_abs_mean_to_kill_as_too_small}"
    return "PASS", "pre-registered stage criteria satisfied (not an edge claim)"
