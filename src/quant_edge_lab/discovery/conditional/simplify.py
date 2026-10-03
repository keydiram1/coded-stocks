"""D2 honest estimation, parsimony, threshold-neighbor robustness."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.conditional import AlgorithmCandidate, apply_conditions, signed_target


def sample_stats(df: pl.DataFrame, target: str, direction: str) -> dict[str, Any]:
    if df.height == 0 or target not in df.columns:
        return {"n": 0, "mean": None, "se": None, "ticker_days": 0, "trading_days": 0, "tickers": 0, "top_ticker_share": None}
    y = signed_target(df, target, direction).to_numpy().astype(float)
    y = y[np.isfinite(y)]
    days = df["trading_date"].n_unique() if "trading_date" in df.columns else 0
    tickers = df["instrument_id"].n_unique() if "instrument_id" in df.columns else 0
    td = df.select(["instrument_id", "trading_date"]).unique().height if {"instrument_id", "trading_date"} <= set(df.columns) else df.height
    top = None
    if "instrument_id" in df.columns and df.height:
        vc = df.group_by("instrument_id").len().sort("len", descending=True)
        top = float(vc["len"][0] / df.height)
    mean = float(np.mean(y)) if y.size else None
    se = float(np.std(y, ddof=1) / np.sqrt(max(len(y), 1))) if y.size > 1 else None
    # day-block SE if possible
    if "trading_date" in df.columns and y.size:
        tmp = df.with_columns(signed_target(df, target, direction).alias("_y"))
        by = tmp.group_by("trading_date").agg(pl.col("_y").mean())
        dm = by["_y"].drop_nulls().to_numpy()
        if dm.size > 1:
            se = float(np.std(dm, ddof=1) / np.sqrt(dm.size))
    return {
        "n": int(y.size),
        "mean": mean,
        "median": float(np.median(y)) if y.size else None,
        "se": se,
        "ticker_days": int(td),
        "trading_days": int(days),
        "tickers": int(tickers),
        "top_ticker_share": top,
        "win_rate": float(np.mean(y > 0)) if y.size else None,
    }


def estimate_candidate(df: pl.DataFrame, cand: AlgorithmCandidate, *, min_td: int, min_days: int, min_tickers: int, max_conc: float) -> dict[str, Any]:
    sub = apply_conditions(df, cand.conditions)
    st = sample_stats(sub, cand.target, cand.direction)
    st["candidate_id"] = cand.candidate_id
    st["n_conditions"] = len(cand.conditions)
    st["direction"] = cand.direction
    kill = None
    if st["ticker_days"] < min_td or st["trading_days"] < min_days or st["tickers"] < min_tickers:
        kill = "sample"
    elif st["top_ticker_share"] is not None and st["top_ticker_share"] > max_conc:
        kill = "concentration"
    st["kill"] = kill
    return st


def simplify_candidate(
    cand: AlgorithmCandidate,
    d2: pl.DataFrame,
    *,
    min_td: int,
    min_days: int,
    min_tickers: int,
    max_conc: float,
) -> tuple[AlgorithmCandidate, list[dict[str, Any]]]:
    log: list[dict[str, Any]] = []
    current = cand
    while len(current.conditions) > 1:
        variants = [current.without(c) for c in current.conditions]
        est = [estimate_candidate(d2, v, min_td=min_td, min_days=min_days, min_tickers=min_tickers, max_conc=max_conc) | {"cand": v} for v in variants]
        est.append(estimate_candidate(d2, current, min_td=min_td, min_days=min_days, min_tickers=min_tickers, max_conc=max_conc) | {"cand": current})
        valid = [e for e in est if e.get("mean") is not None and not e.get("kill")]
        log.append({"step": current.candidate_id, "n_variants": len(variants), "valid": len(valid)})
        if not valid:
            break
        best = max(valid, key=lambda e: abs(e["mean"]))
        se = best.get("se") or 0.0
        acceptable = [
            e
            for e in valid
            if abs(e["mean"]) + 1e-12 >= abs(best["mean"]) - se
            and (e["mean"] > 0) == (best["mean"] > 0)
        ]
        simpler = [e for e in acceptable if e["n_conditions"] < len(current.conditions)]
        if not simpler:
            break
        pick = min(simpler, key=lambda e: (e["n_conditions"], -abs(e["mean"])))
        current = pick["cand"]
        log.append({"chose": current.candidate_id, "n_conditions": len(current.conditions)})
    return current, log


def neighbor_ok(central_mean: float, neighbor_means: list[float | None], frac: float = 0.50) -> bool:
    if central_mean == 0:
        return False
    sign = central_mean > 0
    ok = 0
    for m in neighbor_means:
        if m is None:
            continue
        if (m > 0) == sign and abs(m) >= frac * abs(central_mean):
            ok += 1
    return ok >= len([m for m in neighbor_means if m is not None]) and ok > 0
