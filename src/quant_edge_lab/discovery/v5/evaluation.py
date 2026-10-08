"""Day-aware evaluation and frozen D1→D2→D3 decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.models import CandidateRule, ScientificDecision
from quant_edge_lab.statistics.summarize import _block_mean_distribution


def trial_count(campaign_yaml: dict[str, Any]) -> int:
    return len(campaign_yaml.get("hypotheses") or [])


def concentration(events: pl.DataFrame) -> dict[str, Any]:
    if events.height == 0:
        return {
            "events": 0,
            "ticker_days": 0,
            "trading_days": 0,
            "tickers": 0,
            "top_ticker_share": None,
        }
    td = events.select(["instrument_id", "trading_date"]).unique().height
    days = events["trading_date"].n_unique()
    tickers = events["instrument_id"].n_unique()
    vc = events.group_by("instrument_id").len().sort("len", descending=True)
    top = float(vc["len"][0] / events.height)
    return {
        "events": events.height,
        "ticker_days": int(td),
        "trading_days": int(days),
        "tickers": int(tickers),
        "top_ticker_share": top,
    }


def day_block_stats(events: pl.DataFrame, col: str = "primary_signed", *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    conc = concentration(events)
    if events.height == 0 or col not in events.columns:
        return {**conc, "mean": None, "median": None, "win_rate": None, "se": None, "ci": None, "mean_bp": None}
    ydf = events.filter(pl.col(col).is_not_null())
    y = ydf[col].to_numpy().astype(float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {**conc, "mean": None, "median": None, "win_rate": None, "se": None, "ci": None, "mean_bp": None}
    mean = float(np.mean(y))
    by = ydf.group_by("trading_date").agg(pl.col(col).mean().alias("m"))
    dm = by["m"].drop_nulls().to_numpy().astype(float)
    se = float(np.std(dm, ddof=1) / np.sqrt(dm.size)) if dm.size > 1 else None
    ci = None
    if ydf.height and "trading_day" not in ydf.columns:
        blocks = ydf["trading_date"].cast(pl.Utf8).to_numpy()
    else:
        blocks = ydf["trading_date"].cast(pl.Utf8).to_numpy()
    if y.size > 1:
        dist = _block_mean_distribution(y, blocks, n_boot, seed)
        ci = [float(np.nanquantile(dist, 0.025)), float(np.nanquantile(dist, 0.975))]
    return {
        **conc,
        "mean": mean,
        "median": float(np.median(y)),
        "win_rate": float(np.mean(y > 0)),
        "se": se,
        "ci": ci,
        "mean_bp": mean * 10_000.0,
        "direction_agreement": float(np.mean(np.sign(dm) == np.sign(mean))) if dm.size else None,
    }


def decide(
    *,
    split: str,
    hypothesis_id: str,
    stats: dict[str, Any],
    gates: dict[str, Any],
    d2_mean: float | None = None,
    trial_index: int | None = None,
) -> ScientificDecision:
    sample = gates["sample"]
    floor = float(gates["economic_floor_abs_mean"]["value"])
    if stats["ticker_days"] < sample["min_ticker_days"] or stats["trading_days"] < sample["min_trading_days"] or stats["tickers"] < sample["min_tickers"]:
        return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="KILL", reason="sample", mean_primary=stats.get("mean"), trial_index=trial_index)
    if stats.get("top_ticker_share") is not None and stats["top_ticker_share"] > sample["max_top_ticker_share"]:
        return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="KILL", reason="concentration", mean_primary=stats.get("mean"), trial_index=trial_index)
    mean = stats.get("mean")
    wr = stats.get("win_rate")
    if mean is None:
        return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="KILL", reason="no_mean", trial_index=trial_index)
    if wr is not None and wr < sample["min_win_rate"] and split != "D1":
        return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="KILL", reason="win_rate", mean_primary=mean, trial_index=trial_index)
    if split == "D3" and gates["d3"]["require_sign_match_d2"]:
        if d2_mean is None or np.sign(mean) != np.sign(d2_mean) or mean == 0 or d2_mean == 0:
            return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="KILL", reason="sign_mismatch_d2", mean_primary=mean, trial_index=trial_index)
    if split in {"D2", "D3"} and abs(mean) < floor:
        if split == "D3":
            return ScientificDecision(
                hypothesis_id=hypothesis_id,
                split=split,
                label="VALIDATED_SUBTHRESHOLD_PHENOMENON",
                reason="below_economic_floor",
                mean_primary=mean,
                trial_index=trial_index,
            )
        return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="KILL", reason="below_economic_floor", mean_primary=mean, trial_index=trial_index)
    if split == "D3" and abs(mean) >= floor:
        return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="RESEARCH_PASS", reason="d3_floor_and_sign", mean_primary=mean, trial_index=trial_index)
    return ScientificDecision(hypothesis_id=hypothesis_id, split=split, label="ACTIVE", reason="measured", mean_primary=mean, trial_index=trial_index)


def evaluate_frozen(
    events: pl.DataFrame,
    rules: list[CandidateRule],
    *,
    split: str,
    gates: dict[str, Any],
    d2_means: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule

    out = []
    for i, rule in enumerate(rules, start=1):
        sub = apply_rule(events, rule)
        st = day_block_stats(
            sub,
            n_boot=int(gates["inference"]["bootstrap_draws"]),
            seed=int(gates["inference"]["bootstrap_seed"]),
        )
        dec = decide(
            split=split,
            hypothesis_id=rule.hypothesis_id,
            stats=st,
            gates=gates,
            d2_mean=(d2_means or {}).get(rule.hypothesis_id),
            trial_index=i,
        )
        out.append({"rule": rule.model_dump(), "stats": st, "decision": dec.model_dump()})
    return out
