"""Stage-1+ forward-return summaries. SIGNAL_ONLY; no MFE/MAE required."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from quant_edge_lab.statistics.summarize import _block_mean_distribution, benjamini_hochberg


def summarize_forwards(
    fwd: pl.DataFrame,
    *,
    primary: str = "15m",
    abs_primary: bool = False,
    n_boot: int = 1000,
    bootstrap_primary_only: bool = False,
) -> dict[str, Any]:
    if fwd.height == 0:
        return {"horizons": [], "primary": None, "n_events": 0}
    out_h = []
    pvals: list[float] = []
    labels: list[str] = []
    for horizon, grp in fwd.partition_by("horizon", as_dict=True).items():
        h = str(horizon[0] if isinstance(horizon, tuple) else horizon)
        g = grp.filter(pl.col("forward_return").is_not_null())
        rets = g["forward_return"].to_numpy()
        if abs_primary and h == primary:
            metric = np.abs(rets)
        else:
            metric = rets
        if metric.size == 0:
            continue
        mean = float(np.mean(metric))
        median = float(np.median(metric))
        std = float(np.std(metric, ddof=1)) if metric.size > 1 else 0.0
        q = np.quantile(metric, [0.1, 0.25, 0.5, 0.75, 0.9])
        win = float(np.mean(metric > 0))
        tstat, p = stats.ttest_1samp(metric, 0.0) if metric.size > 1 else (np.nan, 1.0)
        p = float(p) if np.isfinite(p) else 1.0
        if "trading_day" in g.columns and (not bootstrap_primary_only or h == primary):
            blocks = g["trading_day"].cast(pl.String).to_numpy()
            boot = _block_mean_distribution(metric, blocks, n_boot=n_boot, seed=42)
            ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            p_boot = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
        else:
            ci_lo = ci_hi = p_boot = None
        n_td = None
        if "ticker" in g.columns and "event_id" in g.columns:
            ev = g.unique(subset=["event_id"])
            tot = ev.height
            tc = ev.group_by("ticker").len()
            top_share = float(tc["len"].max() / tot) if tot else None
            ticker_n = tc.height
            n_td = ev.select(pl.concat_str(["ticker", "session_date"], separator="|")).n_unique() if "session_date" in ev.columns else tot
            n_days = ev["session_date"].n_unique() if "session_date" in ev.columns else None
        row = {
            "horizon": h,
            "n": int(metric.size),
            "mean": mean,
            "median": median,
            "std": std,
            "q10": float(q[0]),
            "q25": float(q[1]),
            "q50": float(q[2]),
            "q75": float(q[3]),
            "q90": float(q[4]),
            "win_rate": win,
            "ttest_p": p,
            "bootstrap_p": p_boot,
            "bootstrap_ci_95": [ci_lo, ci_hi] if ci_lo is not None else None,
            "abs_metric": bool(abs_primary and h == primary),
        }
        out_h.append(row)
        pvals.append(p_boot if p_boot is not None else p)
        labels.append(h)
    bh = benjamini_hochberg(pvals, q=0.05)
    for item, lab in zip(bh, labels, strict=False):
        item["horizon"] = lab
    primary_row = next((x for x in out_h if x["horizon"] == primary), None)
    # ticker concentration from 15m unique events
    sample = {}
    p15 = fwd.filter((pl.col("horizon") == primary) & pl.col("forward_return").is_not_null())
    if p15.height and "ticker" in p15.columns:
        ev = p15.unique(subset=["event_id"]) if "event_id" in p15.columns else p15
        tot = ev.height
        tc = ev.group_by("ticker").len()
        sample = {
            "ticker_days": tot,
            "unique_tickers": tc.height,
            "trading_days": ev["session_date"].n_unique() if "session_date" in ev.columns else None,
            "top_ticker_share": float(tc["len"].max() / tot) if tot else None,
            "frac_days_positive_mean": None,
        }
        if "session_date" in ev.columns and tot:
            by_d = ev.group_by("session_date").agg(pl.col("forward_return").mean().alias("m"))
            if abs_primary:
                sample["frac_days_positive_mean"] = float((by_d["m"].abs() > 0).mean())
            else:
                sample["frac_days_positive_mean"] = float((by_d["m"] > 0).mean())
    return {
        "horizons": out_h,
        "primary": primary_row,
        "benjamini_hochberg_horizons": bh,
        "sample": sample,
        "n_rows": fwd.height,
        "label": "SIGNAL_ONLY",
    }


def load_job_returns(returns_root: "Path", job_key: str) -> pl.DataFrame:
    from pathlib import Path

    d = Path(returns_root) / job_key
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pl.DataFrame()
    return pl.scan_parquet([str(f) for f in files]).collect()
