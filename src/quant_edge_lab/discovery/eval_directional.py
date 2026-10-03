"""Frozen directional evaluators. Causal bars only; side assigned at decision bar."""

from __future__ import annotations

from datetime import time
from typing import Any

import polars as pl

from quant_edge_lab.discovery.eval_batch10 import _map_num, _stamp, _univ
from quant_edge_lab.universe.filters import apply_universe


def prepare_rth(
    sessioned: pl.DataFrame,
    instruments: pl.DataFrame,
    prev_close: dict[str, float],
    typ: dict[str, float],
    yday_ret: dict[str, float] | None = None,
    ah_ret: dict[str, float] | None = None,
    med20: dict[str, float] | None = None,
) -> pl.DataFrame:
    df = sessioned.sort(["instrument_id", "ts_utc"])
    df = _map_num(df, prev_close, "prev_close")
    df = apply_universe(df, instruments, _univ())
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1).alias("ret_1m"),
        (pl.col("close") / pl.col("close").shift(5).over("instrument_id") - 1).alias("ret_5m"),
        (pl.col("close") / pl.col("close").shift(15).over("instrument_id") - 1).alias("ret_15m"),
        pl.col("volume").rolling_sum(5, min_samples=1).over("instrument_id").alias("roll_vol_5m"),
        pl.col("volume").rolling_sum(10, min_samples=1).over("instrument_id").alias("roll_vol_10m"),
        pl.col("volume").rolling_sum(15, min_samples=1).over("instrument_id").alias("roll_vol_15m"),
        (
            (pl.col("close") * pl.col("volume")).rolling_sum(15, min_samples=5).over("instrument_id")
            / pl.col("volume").rolling_sum(15, min_samples=5).over("instrument_id")
        ).alias("vwap_15m"),
        pl.col("high").rolling_max(15, min_samples=15).over("instrument_id").alias("hi_15"),
        pl.col("low").rolling_min(15, min_samples=15).over("instrument_id").alias("lo_15"),
    )
    df = _map_num(df, typ, "typ_1m_vol")
    if yday_ret is not None:
        df = _map_num(df, yday_ret, "yday_ret")
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("yday_ret"))
    if ah_ret is not None:
        df = _map_num(df, ah_ret, "ah_ret")
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("ah_ret"))
    if med20 is not None:
        df = _map_num(df, med20, "med20")
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("med20"))
    rth = df.filter(pl.col("is_rth"))
    rth = rth.with_columns(
        pl.col("open").first().over("instrument_id").alias("rth_open"),
        pl.col("high").cum_max().over("instrument_id").alias("rth_high_so_far"),
        pl.col("low").cum_min().over("instrument_id").alias("rth_low_so_far"),
        (pl.col("time_et").dt.hour() * 60 + pl.col("time_et").dt.minute() - (9 * 60 + 30)).alias("minutes_from_open"),
    )
    rth = rth.with_columns(
        (pl.col("rth_open") / pl.col("prev_close") - 1).alias("gap_pct"),
        (pl.col("roll_vol_5m") / (pl.col("typ_1m_vol") * 5)).alias("rvol_5m"),
        (pl.col("close") / pl.col("rth_open") - 1).alias("ret_from_open"),
        (
            (pl.col("close") - pl.col("rth_low_so_far"))
            / (pl.col("rth_high_so_far") - pl.col("rth_low_so_far") + 1e-12)
        ).alias("range_loc"),
        ((pl.col("hi_15") - pl.col("lo_15")) / pl.col("prev_close")).alias("range_15m"),
    )
    return rth


def market_open_ret(rth: pl.DataFrame, until: time = time(10, 0)) -> float | None:
    snap = rth.filter(pl.col("time_et") <= until)
    if snap.height == 0:
        return None
    last = snap.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="last")
    last = last.filter(pl.col("rth_open").is_not_null() & (pl.col("rth_open") > 0))
    if last.height < 20:
        return None
    return float((last["close"] / last["rth_open"] - 1).mean())


def xs_open_ret(rth: pl.DataFrame, at: time = time(10, 15)) -> pl.DataFrame:
    snap = rth.filter(pl.col("time_et") <= at)
    if snap.height == 0:
        return pl.DataFrame()
    last = snap.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="last")
    q20 = last["ret_from_open"].quantile(0.20)
    return last.with_columns(pl.lit(q20).alias("xs_p20"))


def _assign_h004_side(ev: pl.DataFrame, job_id: str) -> pl.DataFrame:
    if ev.height == 0:
        return ev
    ev = ev.with_columns(
        (
            2.0 * pl.col("gap_pct").fill_null(0)
            + 1.5 * pl.col("ret_5m").fill_null(0)
            + 0.5 * pl.col("ret_15m").fill_null(0)
            + 0.02 * pl.min_horizontal(pl.col("rvol_5m").fill_null(0), pl.lit(10.0)) / 10
            - 0.0004 * pl.col("minutes_from_open").fill_null(0)
        ).alias("score")
    )
    if job_id == "D004.mom5":
        ev = ev.filter(pl.col("ret_5m") != 0).with_columns(
            pl.when(pl.col("ret_5m") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
    elif job_id == "D004.fade5":
        ev = ev.filter(pl.col("ret_5m") != 0).with_columns(
            pl.when(pl.col("ret_5m") > 0).then(pl.lit("short")).otherwise(pl.lit("long")).alias("side")
        )
    elif job_id == "D004.gap":
        ev = ev.filter(pl.col("gap_pct") != 0).with_columns(
            pl.when(pl.col("gap_pct") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
    elif job_id == "D004.gap_fade":
        ev = ev.filter(pl.col("gap_pct") != 0).with_columns(
            pl.when(pl.col("gap_pct") > 0).then(pl.lit("short")).otherwise(pl.lit("long")).alias("side")
        )
    elif job_id == "D004.range":
        ev = ev.with_columns(
            pl.when(pl.col("range_loc") >= 0.60).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
    elif job_id == "D004.select":
        ev = ev.filter((pl.col("ret_5m").abs() >= 0.004) & (pl.col("rvol_5m") >= 6)).with_columns(
            pl.when(pl.col("ret_5m") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
    elif job_id == "D004.score":
        ev = ev.filter(pl.col("score").abs() >= 0.012).with_columns(
            pl.when(pl.col("score") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
    else:
        raise RuntimeError(job_id)
    return ev


def eval_d004(
    rth: pl.DataFrame,
    instruments: pl.DataFrame,
    prev_close: dict[str, float],
    typ: dict[str, float],
    exp_id: str,
    job_id: str,
) -> pl.DataFrame:
    hits = rth.filter(
        (pl.col("time_et") >= time(9, 40))
        & (pl.col("time_et") <= time(14, 30))
        & (pl.col("rvol_5m") >= 5.0)
        & (pl.col("ret_5m").abs() <= 0.012)
    )
    raw = _stamp(hits, exp_id)
    if raw.height == 0:
        return raw
    return _assign_h004_side(raw, job_id)


def eval_d101(rth: pl.DataFrame, exp_id: str, *, fade: bool = False) -> pl.DataFrame:
    hits = rth.filter(
        (pl.col("time_et") >= time(9, 50))
        & (pl.col("time_et") <= time(14, 30))
        & (pl.col("gap_pct") >= 0.05)
        & (pl.col("rvol_5m") >= 4)
        & (pl.col("ret_5m").abs() <= 0.02)
        & (pl.col("close") >= pl.col("vwap_15m"))
    )
    side = "short" if fade else "long"
    return _stamp(hits, exp_id, side=side)


def eval_d103(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    df = rth.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    if df.height == 0:
        return df
    df = df.with_columns(
        (pl.col("med20").is_not_null() & (pl.col("range_15m") <= 0.40 * pl.col("med20"))).alias("compress"),
        (pl.col("close") > pl.col("hi_15").shift(1).over("instrument_id")).alias("brk_up"),
        (pl.col("close") < pl.col("lo_15").shift(1).over("instrument_id")).alias("brk_dn"),
        pl.col("hi_15").shift(1).over("instrument_id").alias("brk_hi"),
        pl.col("lo_15").shift(1).over("instrument_id").alias("brk_lo"),
    )
    df = df.with_columns(
        pl.col("compress").shift(5).over("instrument_id").alias("c5"),
        pl.col("brk_up").shift(5).over("instrument_id").alias("u5"),
        pl.col("brk_dn").shift(5).over("instrument_id").alias("d5"),
        pl.col("brk_hi").shift(5).over("instrument_id").alias("hi5"),
        pl.col("brk_lo").shift(5).over("instrument_id").alias("lo5"),
    )
    up = df.filter(pl.col("c5") & pl.col("u5") & (pl.col("close") > pl.col("hi5")))
    dn = df.filter(pl.col("c5") & pl.col("d5") & (pl.col("close") < pl.col("lo5")))
    up = up.with_columns(pl.lit("long").alias("side"))
    dn = dn.with_columns(pl.lit("short").alias("side"))
    hits = pl.concat([up, dn], how="diagonal") if up.height or dn.height else df.head(0)
    return _stamp(hits, exp_id)


def eval_d105(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    xs = xs_open_ret(rth, time(10, 15))
    if xs.height == 0:
        return xs
    mkt = market_open_ret(rth, time(10, 15))
    if mkt is None or mkt < 0.002:
        return rth.head(0)
    hits = xs.filter(
        (pl.col("ret_from_open") <= pl.col("xs_p20"))
        & (pl.col("rvol_5m") >= 2)
        & (pl.col("time_et") >= time(10, 10))
        & (pl.col("time_et") <= time(10, 20))
    )
    return _stamp(hits, exp_id, side="long")


def eval_d106(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    hits = rth.filter(
        (pl.col("time_et") >= time(9, 50))
        & (pl.col("time_et") <= time(10, 30))
        & (pl.col("yday_ret") > 0)
        & (pl.col("gap_pct") <= -0.03)
        & (pl.col("close") >= pl.col("rth_open"))
    )
    return _stamp(hits, exp_id, side="long")


def eval_d107(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    mkt = market_open_ret(rth, time(10, 0))
    if mkt is None:
        return rth.head(0)
    if mkt > 0.003:
        side = "short"
    elif mkt < -0.003:
        side = "long"
    else:
        return rth.head(0)
    hits = rth.filter(
        (pl.col("time_et") >= time(10, 5))
        & (pl.col("time_et") <= time(10, 20))
        & (pl.col("gap_pct") >= 0.04)
    )
    return _stamp(hits, exp_id, side=side)


def eval_d108(rth: pl.DataFrame, exp_id: str, *, fade: bool = False) -> pl.DataFrame:
    df = rth.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    if df.height == 0:
        return df
    df = df.with_columns(
        pl.col("ret_5m").shift(15).over("instrument_id").alias("imp"),
        pl.col("close").shift(15).over("instrument_id").alias("c_imp"),
        pl.col("close").shift(5).over("instrument_id").alias("c_pause"),
        pl.col("roll_vol_10m").shift(5).over("instrument_id").alias("vol_pause"),
        pl.col("high").shift(15).over("instrument_id").alias("h_imp"),
        pl.col("low").shift(15).over("instrument_id").alias("l_imp"),
    )
    df = df.filter(pl.col("imp").abs() >= 0.02)
    df = df.filter((pl.col("c_pause") / pl.col("c_imp") - 1).abs() <= 0.006)
    df = df.filter(pl.col("roll_vol_5m") >= 1.5 * (pl.col("vol_pause") / 2.0))
    if fade:
        df = df.filter(
            ((pl.col("imp") > 0) & (pl.col("high") < pl.col("h_imp")))
            | ((pl.col("imp") < 0) & (pl.col("low") > pl.col("l_imp")))
        )
        df = df.with_columns(
            pl.when(pl.col("imp") > 0).then(pl.lit("short")).otherwise(pl.lit("long")).alias("side")
        )
    else:
        df = df.with_columns(
            pl.when(pl.col("imp") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
    return _stamp(df, exp_id)


def eval_d111(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    orb = rth.filter((pl.col("time_et") >= time(9, 30)) & (pl.col("time_et") < time(9, 45)))
    if orb.height == 0:
        return orb
    levels = orb.group_by("instrument_id").agg(
        pl.col("high").max().alias("or_h"),
        pl.col("low").min().alias("or_l"),
    )
    later = rth.filter(pl.col("time_et") >= time(9, 50)).join(levels, on="instrument_id")
    later = later.with_columns(
        (pl.col("close") > pl.col("or_h")).alias("out_up"),
        (pl.col("close") < pl.col("or_l")).alias("out_dn"),
        ((pl.col("close") <= pl.col("or_h")) & (pl.col("close") >= pl.col("or_l"))).alias("inside"),
    )
    first_out = later.filter(pl.col("out_up") | pl.col("out_dn")).sort(["instrument_id", "ts_utc"]).unique(
        subset=["instrument_id"], keep="first"
    )
    if first_out.height == 0:
        return first_out
    fo = first_out.select(
        [
            "instrument_id",
            pl.col("ts_utc").alias("out_ts"),
            "out_up",
            "or_h",
            "or_l",
        ]
    )
    joined = later.join(fo, on="instrument_id")
    fail = joined.filter(
        (pl.col("ts_utc") > pl.col("out_ts"))
        & (pl.col("ts_utc") <= pl.col("out_ts") + pl.duration(minutes=20))
        & pl.col("inside")
    )
    hits = fail.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
    hits = hits.with_columns(
        pl.when(pl.col("out_up")).then(pl.lit("short")).otherwise(pl.lit("long")).alias("side")
    )
    return _stamp(hits, exp_id)


def eval_d112(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    hits = rth.filter(
        (pl.col("time_et") >= time(9, 36))
        & (pl.col("time_et") <= time(10, 15))
        & (pl.col("ah_ret").abs() >= 0.03)
        & pl.col("ret_5m").is_not_null()
    )
    hits = hits.with_columns(
        pl.when(pl.col("ah_ret") * pl.col("ret_5m") > 0)
        .then(pl.when(pl.col("ah_ret") > 0).then(pl.lit("long")).otherwise(pl.lit("short")))
        .otherwise(pl.when(pl.col("ah_ret") > 0).then(pl.lit("short")).otherwise(pl.lit("long")))
        .alias("side")
    )
    return _stamp(hits, exp_id)


def eval_d110(rth: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    df = rth.filter(
        (pl.col("time_et") >= time(9, 50))
        & (pl.col("time_et") <= time(14, 30))
        & (pl.col("rvol_5m") >= 3)
        & (pl.col("gap_pct").abs() >= 0.02)
    )
    df = df.with_columns(
        (
            1.5 * pl.col("gap_pct").fill_null(0)
            + 1.2 * pl.col("ret_5m").fill_null(0)
            + 0.4 * pl.col("yday_ret").fill_null(0)
            + 0.015 * pl.min_horizontal(pl.col("rvol_5m").fill_null(0), pl.lit(12.0)) / 12
        ).alias("score")
    )
    df = df.filter(pl.col("score").abs() >= 0.02).with_columns(
        pl.when(pl.col("score") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
    )
    return _stamp(df, exp_id)


WORKER_A_IDS = [
    "D004.mom5",
    "D004.fade5",
    "D004.gap",
    "D004.gap_fade",
    "D004.range",
    "D004.select",
    "D004.score",
]
WORKER_B_IDS = ["D101", "D102", "D103", "D105", "D106", "D107", "D108", "D109", "D111", "D112", "D110"]


def eval_job(job_id: str, rth: pl.DataFrame, *, instruments: pl.DataFrame, prev_close: dict, typ: dict, exp_id: str) -> pl.DataFrame:
    if job_id.startswith("D004."):
        return eval_d004(rth, instruments, prev_close, typ, exp_id, job_id)
    if job_id == "D101":
        return eval_d101(rth, exp_id, fade=False)
    if job_id == "D102":
        return eval_d101(rth, exp_id, fade=True)
    if job_id == "D103":
        return eval_d103(rth, exp_id)
    if job_id == "D105":
        return eval_d105(rth, exp_id)
    if job_id == "D106":
        return eval_d106(rth, exp_id)
    if job_id == "D107":
        return eval_d107(rth, exp_id)
    if job_id == "D108":
        return eval_d108(rth, exp_id, fade=False)
    if job_id == "D109":
        return eval_d108(rth, exp_id, fade=True)
    if job_id == "D111":
        return eval_d111(rth, exp_id)
    if job_id == "D112":
        return eval_d112(rth, exp_id)
    if job_id == "D110":
        return eval_d110(rth, exp_id)
    raise RuntimeError(f"unknown directional job {job_id}")
