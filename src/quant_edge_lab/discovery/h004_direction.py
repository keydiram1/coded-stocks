"""H004 signed baseline from cached full-panel returns. No extra tape read."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.knowledge import atomic_write_json, knowledge_dir

console = Console()


def _scan_cached(root: Path, job_key: str) -> pl.LazyFrame | None:
    d = Paths(root).derived / "discovery" / "batches" / "batch10-full" / "returns" / job_key
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return None
    return pl.scan_parquet([str(f) for f in files])


def _day_block_ci(day_means: np.ndarray, n_boot: int = 400, seed: int = 42) -> list[float] | None:
    if day_means.size < 5:
        return None
    rng = np.random.default_rng(seed)
    boots = np.array([float(np.mean(rng.choice(day_means, size=day_means.size, replace=True))) for _ in range(n_boot)])
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def write_h004_signed_baseline(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "h004_signed_baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "source": "batch10-full compact parquet (lazy aggregations)",
        "note": "SIGNAL_ONLY signed long-convention returns from H004 events. Not a directional edge.",
        "jobs": {},
    }
    for key in ("H004", "H004.v1"):
        console.print(f"  aggregating {key} cached returns (streaming)…")
        lf = _scan_cached(root, key)
        if lf is None:
            out["jobs"][key] = {"error": "no cached returns"}
            continue
        valid = lf.filter(pl.col("forward_return").is_not_null())
        by_h = (
            valid.group_by("horizon")
            .agg(
                pl.len().alias("n"),
                pl.col("forward_return").mean().alias("mean"),
                pl.col("forward_return").median().alias("median"),
                pl.col("forward_return").std().alias("std"),
                (pl.col("forward_return") > 0).mean().alias("pct_positive"),
                (pl.col("forward_return") < 0).mean().alias("pct_negative"),
                pl.col("forward_return").quantile(0.1).alias("q10"),
                pl.col("forward_return").quantile(0.25).alias("q25"),
                pl.col("forward_return").quantile(0.75).alias("q75"),
                pl.col("forward_return").quantile(0.9).alias("q90"),
                pl.col("forward_return").abs().mean().alias("abs_mean"),
            )
            .collect()
        )
        p15 = valid.filter(pl.col("horizon") == "15m")
        day_df = p15.group_by("trading_day").agg(pl.col("forward_return").mean().alias("m")).collect()
        day_means = day_df["m"].to_numpy() if day_df.height else np.array([])
        ci = _day_block_ci(day_means)
        ev = p15.unique(subset=["event_id"]) if "event_id" in p15.collect_schema() else p15
        sample_df = ev.group_by("ticker").agg(pl.len().alias("n")).collect()
        tot = int(sample_df["n"].sum()) if sample_df.height else 0
        top = float(sample_df["n"].max() / tot) if tot else None
        n_days = day_df.height
        years = []
        if "trading_day" in (p15.collect_schema() or {}):
            ydf = (
                p15.with_columns(pl.col("trading_day").cast(pl.Utf8).str.slice(0, 4).alias("year"))
                .group_by("year")
                .agg(
                    pl.col("forward_return").mean().alias("mean"),
                    pl.col("forward_return").median().alias("median"),
                    pl.len().alias("n"),
                )
                .sort("year")
                .collect()
            )
            years = ydf.to_dicts()
        horizons = by_h.sort("horizon").to_dicts()
        prim = next((h for h in horizons if str(h.get("horizon")) == "15m"), None)
        if prim:
            prim = {
                **prim,
                "win_rate": prim.get("pct_positive"),
                "bootstrap_ci_95": ci,
                "day_block_note": "CI from bootstrap of daily means (equal-weight days)",
            }
        out["jobs"][key] = {
            "signed_horizons": horizons,
            "signed_primary": prim,
            "unsigned_primary": {"mean": prim.get("abs_mean") if prim else None, "horizon": "15m"},
            "sample": {
                "ticker_days": tot,
                "unique_tickers": sample_df.height,
                "trading_days": n_days,
                "top_ticker_share": top,
                "frac_days_positive_mean": float(np.mean(day_means > 0)) if day_means.size else None,
            },
            "by_year": years,
            "n_horizon_groups": by_h.height,
        }
        console.print(f"  {key} 15m signed mean={None if not prim else prim.get('mean')} days={n_days}")
    path = knowledge_dir(root) / "experiments" / "h004_signed_baseline.json"
    atomic_write_json(path, out)
    out["path"] = str(path)
    return out
