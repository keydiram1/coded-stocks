"""Frozen directional promotion gates. Load YAML; never infer from results."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from quant_edge_lab.hashing import sha256_file

GATES_REL = Path("knowledge/families/directional_gates_v1.yaml")
BATCH_REL = Path("knowledge/families/directional_batch_v1.yaml")


class DirStageGate(BaseModel):
    min_ticker_days: int
    min_trading_days: int
    min_abs_mean_15m: float
    min_win_rate: float | None = None
    min_frac_days_agree: float | None = None
    min_median_same_sign: bool = False
    require_ci_excludes_zero: bool = False
    max_top_ticker_share: float = 0.50
    kill_if_sign_disagrees: bool = True
    require_outlier_robust: bool = False
    outlier_floor: float = 0.0008


class DirGates(BaseModel):
    version: str
    primary_horizon: str = "15m"
    stages: dict[str, DirStageGate]


def load_gates_from(root: Path, rel: Path) -> DirGates:
    raw = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
    stages = {k: DirStageGate.model_validate(v) for k, v in raw["stages"].items()}
    return DirGates(version=raw["version"], primary_horizon=raw.get("primary_horizon", "15m"), stages=stages)


def load_gates(root: Path) -> DirGates:
    raw = yaml.safe_load((root / GATES_REL).read_text(encoding="utf-8"))
    stages = {k: DirStageGate.model_validate(v) for k, v in raw["stages"].items()}
    return DirGates(version=raw["version"], primary_horizon=raw.get("primary_horizon", "15m"), stages=stages)


def freeze_hashes(root: Path) -> dict[str, str]:
    return {
        "gates": sha256_file(root / GATES_REL),
        "batch": sha256_file(root / BATCH_REL),
        "gates_path": str(GATES_REL).replace("\\", "/"),
        "batch_path": str(BATCH_REL).replace("\\", "/"),
    }


def decide_directional(
    *,
    mean_primary: float | None,
    median_primary: float | None,
    win_rate: float | None,
    ticker_days: int,
    trading_days: int,
    top_ticker_share: float | None,
    frac_days_agree: float | None,
    ci: list[float] | None,
    gate: DirStageGate,
    stage_complete: bool,
    extras: dict | None = None,
) -> tuple[str, str]:
    """Returns are signed to the algorithm's LONG/SHORT decision (positive = correct)."""
    if not stage_complete:
        return "CONTINUE", "stage incomplete"
    if ticker_days < gate.min_ticker_days or trading_days < gate.min_trading_days:
        return "KILL", (
            f"insufficient clustered sample ticker_days={ticker_days} "
            f"trading_days={trading_days} (min {gate.min_ticker_days}/{gate.min_trading_days})"
        )
    if top_ticker_share is not None and top_ticker_share > gate.max_top_ticker_share:
        return "KILL", f"concentration top_ticker_share={top_ticker_share:.2f} > {gate.max_top_ticker_share}"
    if mean_primary is None:
        return "KILL", "no primary-horizon returns"
    if gate.kill_if_sign_disagrees and mean_primary < 0:
        return "KILL", f"sign disagrees (decision-signed mean_15m={mean_primary})"
    floor = (extras or {}).get("economic_floor")
    if floor is None:
        floor = gate.min_abs_mean_15m
    if abs(mean_primary) < floor:
        return "KILL", f"abs mean {mean_primary} below economic floor {floor}"
    if gate.min_win_rate is not None and (win_rate is None or win_rate < gate.min_win_rate):
        return "KILL", f"win_rate {win_rate} < {gate.min_win_rate}"
    if gate.min_median_same_sign and (median_primary is None or median_primary < 0):
        return "KILL", f"median {median_primary} not same sign as decision"
    if gate.min_frac_days_agree is not None:
        if frac_days_agree is None or frac_days_agree < gate.min_frac_days_agree:
            return "KILL", f"frac_days_agree {frac_days_agree} < {gate.min_frac_days_agree}"
    if gate.require_ci_excludes_zero:
        if not ci or len(ci) != 2 or ci[0] is None or ci[1] is None:
            return "KILL", "missing day-block CI"
        if not (ci[0] > 0 and ci[1] > 0):
            return "KILL", f"CI does not exclude 0 on the profitable side {ci}"
    if gate.require_outlier_robust:
        om = extras.get("outlier_mean") if extras else None
        dm = extras.get("drop_best_day_mean") if extras else None
        if om is None or dm is None:
            return "KILL", "missing outlier-robust diagnostics"
        if om < gate.outlier_floor or dm < gate.outlier_floor:
            return "KILL", (
                f"outlier-robust means om={om} day={dm} below floor {gate.outlier_floor}"
            )
    return "PASS", "pre-registered directional stage criteria satisfied (not an edge claim)"


def frac_days_agree(fwd, primary: str = "15m") -> float | None:
    import polars as pl

    if fwd is None or fwd.height == 0:
        return None
    g = fwd.filter((pl.col("horizon") == primary) & pl.col("forward_return").is_not_null())
    if g.height == 0 or "session_date" not in g.columns:
        return None
    by = g.group_by("session_date").agg(pl.col("forward_return").mean().alias("m"))
    if by.height == 0:
        return None
    return float((by["m"] > 0).mean())
