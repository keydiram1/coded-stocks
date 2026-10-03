from __future__ import annotations

from datetime import time

import polars as pl

from quant_edge_lab.features.engine import FEATURE_VERSION
from quant_edge_lab.models.hypothesis import HypothesisSpec
from quant_edge_lab.universe.filters import apply_universe, parse_hhmm

OPS = {
    ">=": lambda c, v: pl.col(c) >= v,
    "<=": lambda c, v: pl.col(c) <= v,
    ">": lambda c, v: pl.col(c) > v,
    "<": lambda c, v: pl.col(c) < v,
    "==": lambda c, v: pl.col(c) == v,
}


def _in_window(t: time, start: time, end: time) -> bool:
    return start <= t <= end


def evaluate_events(
    featured: pl.DataFrame,
    instruments: pl.DataFrame,
    spec: HypothesisSpec,
    experiment_id: str,
) -> pl.DataFrame:
    df = apply_universe(featured, instruments, spec.universe)
    start = parse_hhmm(spec.trigger.time_et.start)
    end = parse_hhmm(spec.trigger.time_et.end)

    if spec.trigger.session == "RTH":
        df = df.filter(pl.col("is_rth"))
    elif spec.trigger.session == "PREMARKET":
        df = df.filter(pl.col("is_premarket"))

    # decision_ts is bar close; trigger window is evaluated at decision time (ET).
    df = df.with_columns(pl.col("decision_ts").alias("decision_ts_naive"))
    # decision_ts stored naive UTC; compare using ts_et + 1 minute.
    df = df.with_columns((pl.col("ts_et") + pl.duration(minutes=1)).alias("decision_et"))
    df = df.filter(
        (pl.col("decision_et").dt.time() >= start) & (pl.col("decision_et").dt.time() <= end)
    )

    mask = pl.lit(True)
    for cond in spec.trigger.conditions:
        if cond.feature not in df.columns:
            raise ValueError(f"Unknown feature '{cond.feature}' in hypothesis {spec.id}")
        mask = mask.and_(OPS[cond.op](cond.feature, cond.value))
        mask = mask.and_(pl.col(cond.feature).is_not_null())

    hit = df.filter(mask)
    feat_cols = list(spec.features.keys())
    for f in feat_cols:
        if f not in hit.columns:
            raise ValueError(f"Feature '{f}' declared but not computed")

    if hit.height == 0:
        return pl.DataFrame(
            schema={
                "event_id": pl.String,
                "experiment_id": pl.String,
                "instrument_id": pl.String,
                "ticker": pl.String,
                "event_ts": pl.Datetime("us"),
                "decision_ts": pl.Datetime("us"),
                "feature_version": pl.String,
                "side": pl.String,
                "session_date": pl.Date,
                "entry_ts": pl.Datetime("us"),
                **{f: pl.Float64 for f in feat_cols},
            }
        )

    hit = hit.with_columns(
        pl.lit(experiment_id).alias("experiment_id"),
        pl.lit(FEATURE_VERSION).alias("feature_version"),
        pl.lit(spec.side).alias("side"),
        pl.col("ts_utc").alias("event_ts"),
        pl.col("decision_ts"),
        pl.col("decision_ts").alias("entry_ts"),
    )
    hit = hit.with_columns(
        (
            pl.lit("evt-")
            + pl.col("instrument_id")
            + pl.lit("-")
            + pl.col("decision_ts").cast(pl.String)
        ).alias("event_id")
    )
    cols = [
        "event_id",
        "experiment_id",
        "instrument_id",
        "ticker",
        "event_ts",
        "decision_ts",
        "feature_version",
        "side",
        "session_date",
        "entry_ts",
        *feat_cols,
    ]
    return hit.select(cols).unique(subset=["event_id"]).sort(["decision_ts", "instrument_id"])
