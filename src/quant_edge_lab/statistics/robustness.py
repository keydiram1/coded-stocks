"""Concentration and calendar/bucket robustness. Does not change event definitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _horizon_frame(outcomes: pl.DataFrame, horizon: str | None = None) -> pl.DataFrame:
    if outcomes.height == 0:
        return outcomes
    h = horizon or str(outcomes["horizon"][0])
    return outcomes.filter(pl.col("horizon") == h)


def _mean_stats(rets: np.ndarray) -> dict[str, float | None]:
    if rets.size == 0:
        return {"n": 0, "mean": None, "median": None, "win_rate": None}
    return {
        "n": int(rets.size),
        "mean": float(np.mean(rets)),
        "median": float(np.median(rets)),
        "win_rate": float(np.mean(rets > 0)),
        "std": float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0,
    }


def robustness_tables(events: pl.DataFrame, outcomes: pl.DataFrame) -> dict[str, Any]:
    primary = _horizon_frame(outcomes)
    if primary.height == 0:
        return {"horizon": None, "note": "no outcomes"}
    horizon = str(primary["horizon"][0])
    df = primary.filter(pl.col("forward_return").is_not_null())
    if "prev_close" in events.columns:
        df = df.join(
            events.select(["event_id", "prev_close", "session_date"]).unique(subset=["event_id"]),
            on="event_id",
            how="left",
        )
    else:
        df = df.join(events.select(["event_id", "session_date"]).unique(subset=["event_id"]), on="event_id", how="left")

    df = df.with_columns(
        pl.col("trading_day").cast(pl.Utf8).str.slice(0, 4).alias("year"),
        pl.col("trading_day").cast(pl.Utf8).str.slice(0, 7).alias("year_month"),
    )
    if "prev_close" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("prev_close") < 5)
            .then(pl.lit("1-5"))
            .when(pl.col("prev_close") < 10)
            .then(pl.lit("5-10"))
            .when(pl.col("prev_close") <= 20)
            .then(pl.lit("10-20"))
            .otherwise(pl.lit("other"))
            .alias("price_bucket")
        )
    if "dollar_volume" in events.columns:
        evd = events.select(["event_id", "dollar_volume"]).unique(subset=["event_id"])
        df = df.join(evd, on="event_id", how="left")
        if df["dollar_volume"].drop_nulls().len():
            qs = [
                float(df.select(pl.col("dollar_volume").quantile(q)).item())
                for q in (0.25, 0.5, 0.75)
            ]
            df = df.with_columns(
                pl.when(pl.col("dollar_volume") <= qs[0])
                .then(pl.lit("dvol_q1"))
                .when(pl.col("dollar_volume") <= qs[1])
                .then(pl.lit("dvol_q2"))
                .when(pl.col("dollar_volume") <= qs[2])
                .then(pl.lit("dvol_q3"))
                .otherwise(pl.lit("dvol_q4"))
                .alias("dvol_bucket")
            )

    def group_means(col: str) -> list[dict[str, Any]]:
        if col not in df.columns:
            return []
        rows = []
        for key, grp in df.partition_by(col, as_dict=True).items():
            k = key[0] if isinstance(key, tuple) else key
            rets = grp["forward_return"].to_numpy()
            rows.append({"bucket": str(k), **_mean_stats(rets)})
        rows.sort(key=lambda r: str(r["bucket"]))
        return rows

    rets = df["forward_return"].to_numpy()
    order = np.argsort(-rets)
    def exclude_top_frac(frac: float) -> dict[str, float | None]:
        n = max(1, int(np.ceil(frac * rets.size))) if rets.size else 0
        if rets.size <= n:
            return _mean_stats(np.array([]))
        keep = np.ones(rets.size, dtype=bool)
        keep[order[:n]] = False
        return _mean_stats(rets[keep]) | {"excluded": int(n)}

    by_ticker = (
        df.group_by("ticker")
        .agg(
            pl.len().alias("n"),
            pl.col("forward_return").sum().alias("sum_ret"),
            pl.col("forward_return").mean().alias("mean_ret"),
        )
        .sort("sum_ret", descending=True)
        .head(15)
        .to_dicts()
    )
    by_day = (
        df.group_by("trading_day")
        .agg(
            pl.len().alias("n"),
            pl.col("forward_return").sum().alias("sum_ret"),
            pl.col("forward_return").mean().alias("mean_ret"),
        )
        .sort("sum_ret", descending=True)
        .head(15)
        .to_dicts()
    )
    for row in by_day:
        row["trading_day"] = str(row["trading_day"])

    return {
        "horizon": horizon,
        "by_year": group_means("year"),
        "by_month": group_means("year_month"),
        "by_price_bucket": group_means("price_bucket"),
        "by_dvol_bucket": group_means("dvol_bucket"),
        "top_tickers_by_sum_return": by_ticker,
        "top_days_by_sum_return": by_day,
        "exclude_top_1pct_profitable": exclude_top_frac(0.01),
        "exclude_top_5pct_profitable": exclude_top_frac(0.05),
        "note": "Forward returns are SIGNAL_ONLY next-bar-open to horizon close. Not fills.",
    }
