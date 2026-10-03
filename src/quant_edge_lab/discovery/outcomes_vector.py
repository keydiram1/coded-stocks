"""Vectorized SIGNAL_ONLY forward returns. Same convention as compute_outcomes (no barriers)."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from quant_edge_lab.outcomes.engine import parse_horizon_minutes


def forward_returns_vectorized(
    events: pl.DataFrame,
    bars: pl.DataFrame,
    horizons: list[str],
    side: str,
) -> pl.DataFrame:
    """One row per event × horizon with forward_return and entry_price.

    entry = open of first bar with ts_utc >= entry_ts
    horizon Nm close = close of bar with ts_utc == entry_ts + (N-1) minutes
    """
    if events.height == 0 or bars.height == 0:
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "horizon": pl.String,
                "forward_return": pl.Float64,
                "entry_price": pl.Float64,
            }
        )
    side_col = "side" if "side" in events.columns else None
    ev_cols = ["event_id", "instrument_id", "entry_ts"] + ([side_col] if side_col else [])
    ev = events.select(ev_cols).sort(["instrument_id", "entry_ts"])
    b = bars.select(["instrument_id", "ts_utc", "open", "close"]).sort(["instrument_id", "ts_utc"])
    entered = ev.join_asof(
        b.rename({"ts_utc": "entry_bar_ts", "open": "entry_price"}).drop("close"),
        left_on="entry_ts",
        right_on="entry_bar_ts",
        by="instrument_id",
        strategy="forward",
        check_sortedness=False,
    )
    frames = []
    for h in horizons:
        mins = parse_horizon_minutes(h)
        tgt = entered.with_columns(
            (pl.col("entry_ts") + pl.duration(minutes=mins - 1)).alias("tgt_ts"),
            pl.lit(h).alias("horizon"),
        )
        hit = tgt.join(
            b.rename({"ts_utc": "tgt_ts", "close": "exit_close"}).drop("open"),
            on=["instrument_id", "tgt_ts"],
            how="left",
        )
        long_ret = pl.col("exit_close") / pl.col("entry_price") - 1
        short_ret = 1 - pl.col("exit_close") / pl.col("entry_price")
        if side_col:
            ret = pl.when(pl.col("side") == "long").then(long_ret).otherwise(short_ret)
        elif side == "long":
            ret = long_ret
        else:
            ret = short_ret
        frames.append(hit.select("event_id", "horizon", ret.alias("forward_return"), "entry_price"))
    return pl.concat(frames, how="vertical")


def means_by_horizon(fwd: pl.DataFrame, *, abs_mean: bool = False) -> dict[str, float | None]:
    if fwd.height == 0:
        return {}
    col = pl.col("forward_return").abs() if abs_mean else pl.col("forward_return")
    g = fwd.filter(pl.col("forward_return").is_not_null()).group_by("horizon").agg(col.mean().alias("m"))
    return {str(r["horizon"]): (float(r["m"]) if r["m"] is not None else None) for r in g.iter_rows(named=True)}
