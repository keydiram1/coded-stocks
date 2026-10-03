from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


def benjamini_hochberg(pvalues: list[float], q: float = 0.05) -> list[dict]:
    n = len(pvalues)
    if n == 0:
        return []
    order = np.argsort(pvalues)
    ranked = [(int(i), float(pvalues[i])) for i in order]
    thresh = [q * (k + 1) / n for k in range(n)]
    max_k = -1
    for k, (_, p) in enumerate(ranked):
        if p <= thresh[k]:
            max_k = k
    rejected = {ranked[k][0] for k in range(max_k + 1)} if max_k >= 0 else set()
    out = []
    for orig_i, p in enumerate(pvalues):
        out.append(
            {
                "index": orig_i,
                "p": p,
                "bh_rejected": orig_i in rejected,
                "q_target": q,
            }
        )
    return out


def _block_mean_distribution(
    values: np.ndarray,
    blocks: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    means = np.empty(n_boot)
    grouped = {b: values[blocks == b] for b in unique}
    for i in range(n_boot):
        draw = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([grouped[b] for b in draw]) if len(draw) else np.array([np.nan])
        means[i] = float(np.nanmean(sample))
    return means


def summarize_outcomes(
    events: pl.DataFrame,
    outcomes: pl.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    q_target: float = 0.05,
) -> dict:
    n_events = events.height
    n_symbol_days = events.select(pl.concat_str(["ticker", "session_date"], separator="|")).n_unique() if n_events else 0
    n_symbols = events["instrument_id"].n_unique() if n_events else 0
    n_days = events["session_date"].n_unique() if n_events else 0

    horizons = []
    pvals: list[float] = []
    labels: list[str] = []

    if outcomes.height:
        for horizon, grp in outcomes.partition_by("horizon", as_dict=True).items():
            h = horizon[0] if isinstance(horizon, tuple) else horizon
            rets = grp.filter(pl.col("forward_return").is_not_null())["forward_return"].to_numpy()
            if rets.size == 0:
                continue
            mean = float(np.mean(rets))
            median = float(np.median(rets))
            std = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
            se = std / np.sqrt(rets.size) if rets.size else None
            q = np.quantile(rets, [0.1, 0.25, 0.5, 0.75, 0.9])
            pos_rate = float(np.mean(rets > 0))
            tstat, p = stats.ttest_1samp(rets, 0.0) if rets.size > 1 else (np.nan, 1.0)
            p = float(p) if np.isfinite(p) else 1.0
            blocks = grp.filter(pl.col("forward_return").is_not_null())["trading_day"].cast(pl.String).to_numpy()
            boot = _block_mean_distribution(rets, blocks, n_boot=n_boot, seed=seed)
            ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            # two-sided bootstrap p vs 0
            p_boot = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
            p_boot = min(1.0, p_boot)
            mfe = float(grp["mfe"].drop_nulls().mean() or 0)
            mae = float(grp["mae"].drop_nulls().mean() or 0)
            total = float(np.sum(rets))
            order = np.argsort(-np.abs(rets))
            top1 = max(1, int(np.ceil(0.01 * rets.size)))
            concentration_top1_share = float(np.sum(rets[order[:top1]]) / total) if total != 0 else 0.0
            amb_rate = float(grp["ambiguous"].mean())
            horizons.append(
                {
                    "horizon": h,
                    "n": int(rets.size),
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "se": se,
                    "q10": float(q[0]),
                    "q25": float(q[1]),
                    "q50": float(q[2]),
                    "q75": float(q[3]),
                    "q90": float(q[4]),
                    "positive_event_rate": pos_rate,
                    "ttest_p": p,
                    "bootstrap_p": p_boot,
                    "bootstrap_ci_95": [ci_lo, ci_hi],
                    "mfe_mean": mfe,
                    "mae_mean": mae,
                    "concentration_top1pct_share_of_sum": concentration_top1_share,
                    "ambiguous_rate": amb_rate,
                    "effect_size_mean_over_std": (mean / std) if std else None,
                }
            )
            pvals.append(p_boot)
            labels.append(str(h))

    bh = benjamini_hochberg(pvals, q=q_target)
    for item, lab in zip(bh, labels, strict=False):
        item["horizon"] = lab

    chrono = []
    if events.height:
        chrono = (
            events.group_by("session_date")
            .agg(pl.len().alias("n_events"))
            .sort("session_date")
            .to_dicts()
        )
        for row in chrono:
            row["session_date"] = str(row["session_date"])

    by_symbol = []
    if events.height:
        by_symbol = (
            events.group_by(["ticker", "instrument_id"])
            .agg(pl.len().alias("n_events"))
            .sort("n_events", descending=True)
            .to_dicts()
        )

    return {
        "n_raw_events": n_events,
        "n_symbol_days": n_symbol_days,
        "n_symbols": n_symbols,
        "n_trading_days": n_days,
        "horizons": horizons,
        "benjamini_hochberg": bh,
        "chronological": chrono,
        "by_symbol": by_symbol,
        "labels": {
            "execution": "SIGNAL_ONLY",
            "stage": "EXPLORATORY",
            "sample_is_research_evidence": False,
        },
    }
