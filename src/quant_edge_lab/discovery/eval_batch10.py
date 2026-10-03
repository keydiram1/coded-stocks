"""Batch-10 family evaluators. Uses only bars with available_at <= decision_ts by construction."""

from __future__ import annotations

from datetime import time

import polars as pl

from quant_edge_lab.discovery.event_policy import apply_event_policy
from quant_edge_lab.events.engine import evaluate_events
from quant_edge_lab.models.hypothesis import HypothesisSpec
from quant_edge_lab.universe.filters import apply_universe


def _map_num(df: pl.DataFrame, mapping: dict[str, float], out_col: str) -> pl.DataFrame:
    if not mapping:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias(out_col))
    m = pl.DataFrame({"instrument_id": list(mapping.keys()), out_col: [float(v) for v in mapping.values()]})
    return df.join(m, on="instrument_id", how="left")


def _stamp(hits: pl.DataFrame, exp_id: str, *, side: str = "long", entry_next: bool = True) -> pl.DataFrame:
    if hits.height == 0:
        return hits
    decision = pl.col("ts_utc") + pl.duration(minutes=1)
    entry = decision if entry_next else pl.col("ts_utc")
    extras = [
        pl.lit(exp_id).alias("experiment_id"),
        decision.alias("decision_ts"),
        entry.alias("entry_ts"),
        pl.col("ts_utc").alias("event_ts"),
        (pl.lit("evt-") + pl.col("instrument_id") + pl.lit("-") + decision.cast(pl.String)).alias("event_id"),
    ]
    if "side" not in hits.columns:
        extras.append(pl.lit(side).alias("side"))
    out = hits.with_columns(extras)
    return apply_event_policy(out, "first_ticker_day")


def eval_h001(
    featured: pl.DataFrame,
    instruments: pl.DataFrame,
    spec: HypothesisSpec,
    exp_id: str,
    gap: float = 0.15,
    rvol: float = 5.0,
) -> pl.DataFrame:
    spec = spec.model_copy(deep=True)
    for c in spec.trigger.conditions:
        if c.feature == "gap_pct":
            c.value = gap
        if c.feature == "rvol_5m":
            c.value = rvol
    ev = evaluate_events(featured, instruments, spec, experiment_id=exp_id)
    return apply_event_policy(ev, "first_ticker_day")


def eval_h040(
    featured: pl.DataFrame,
    instruments: pl.DataFrame,
    spec: HypothesisSpec,
    exp_id: str,
    *,
    monday_only: bool = True,
    midweek: bool = False,
    gap_up: bool = True,
    gap: float = 0.08,
) -> pl.DataFrame:
    spec = spec.model_copy(deep=True)
    for c in spec.trigger.conditions:
        if c.feature == "gap_pct":
            c.value = gap if gap_up else -abs(gap)
            c.op = ">=" if gap_up else "<="
    spec.side = "short" if gap_up else "long"
    ev = evaluate_events(featured, instruments, spec, experiment_id=exp_id)
    if ev.height == 0:
        return ev
    wd = ev["session_date"].dt.weekday()
    if monday_only:
        ev = ev.filter(wd == 1)
    elif midweek:
        ev = ev.filter(wd.is_in([2, 3, 4]))
    return apply_event_policy(ev, "first_ticker_day")


def eval_h002(sessioned: pl.DataFrame, instruments: pl.DataFrame, prev_close: dict[str, float], exp_id: str, near: float = 0.003, side: str = "long") -> pl.DataFrame:
    df = apply_universe(_map_num(sessioned, prev_close, "prev_close"), instruments, _univ())
    rth = df.filter(pl.col("is_rth"))
    if rth.height == 0:
        return rth.head(0)
    rth_open = (
        rth.filter(pl.col("time_et") == time(9, 30))
        .group_by("instrument_id")
        .agg(pl.col("open").first().alias("rth_open"))
    )
    w = rth.filter((pl.col("time_et") >= time(9, 30)) & (pl.col("time_et") <= time(9, 44)))
    stats = w.group_by("instrument_id").agg(
        pl.col("high").max().alias("H_w"),
        pl.col("low").min().alias("L_w"),
        pl.col("close").min().alias("min_c"),
        pl.col("prev_close").first(),
        pl.col("ticker").first(),
        pl.col("session_date").first(),
    ).join(rth_open, on="instrument_id")
    stats = stats.filter(
        (pl.col("rth_open") / pl.col("prev_close") - 1 >= 0.15)
        & (pl.col("H_w") > pl.col("rth_open"))
        & ((pl.col("H_w") - pl.col("L_w")) / pl.col("rth_open") >= 0.002)
        & (pl.col("min_c") >= pl.col("H_w") * (1 - near))
    )
    if stats.height == 0:
        return stats.head(0)
    later = rth.filter((pl.col("time_et") >= time(9, 45)) & (pl.col("time_et") <= time(10, 29)))
    later = later.join(stats.select(["instrument_id", "H_w", "ticker", "session_date"]), on="instrument_id")
    hits = later.filter(pl.col("high") >= pl.col("H_w")).sort(["instrument_id", "ts_utc"]).unique(
        subset=["instrument_id"], keep="first"
    )
    return _stamp(hits, exp_id, side=side)


def _univ():
    from quant_edge_lab.models.hypothesis import UniverseSpec

    return UniverseSpec(exchanges=["NASDAQ", "NYSE", "NYSE_AMERICAN"], security_type="COMMON_STOCK")


def eval_h003(
    sessioned: pl.DataFrame,
    instruments: pl.DataFrame,
    prev_close: dict[str, float],
    typ: dict[str, float],
    exp_id: str,
    *,
    ret_15m: float = 0.15,
    ret_5m: float = 0.08,
    side: str = "short",
) -> pl.DataFrame:
    df = sessioned.sort(["instrument_id", "ts_utc"]).with_columns(
        (pl.col("close") / pl.col("close").shift(5).over("instrument_id") - 1).alias("ret_5m"),
        (pl.col("close") / pl.col("close").shift(15).over("instrument_id") - 1).alias("ret_15m"),
        pl.col("volume").rolling_sum(5, min_samples=1).over("instrument_id").alias("roll_vol_5m"),
        pl.col("volume").rolling_sum(5, min_samples=1).shift(10).over("instrument_id").alias("roll_lag10"),
    )
    df = _map_num(_map_num(df, prev_close, "prev_close"), typ, "typ_1m_vol")
    df = apply_universe(df, instruments, _univ())
    df = df.with_columns((pl.col("roll_vol_5m") / (pl.col("typ_1m_vol") * 5)).alias("rvol_5m"))
    hits = df.filter(
        pl.col("is_rth")
        & (pl.col("time_et") >= time(9, 50))
        & (pl.col("time_et") <= time(15, 0))
        & (pl.col("ret_15m") >= ret_15m)
        & (pl.col("ret_5m") >= ret_5m)
        & (pl.col("ret_5m") >= 0.50 * pl.col("ret_15m"))
        & (pl.col("rvol_5m") >= 3.0)
        & (pl.col("roll_vol_5m") >= 1.25 * pl.col("roll_lag10"))
    )
    return _stamp(hits, exp_id, side=side)


def eval_h004(sessioned: pl.DataFrame, instruments: pl.DataFrame, prev_close: dict[str, float], typ: dict[str, float], exp_id: str, rvol: float = 5.0) -> pl.DataFrame:
    df = sessioned.sort(["instrument_id", "ts_utc"]).with_columns(
        (pl.col("close") / pl.col("close").shift(5).over("instrument_id") - 1).alias("ret_5m"),
        pl.col("volume").rolling_sum(5, min_samples=1).over("instrument_id").alias("roll_vol_5m"),
    )
    df = _map_num(_map_num(df, prev_close, "prev_close"), typ, "typ_1m_vol")
    df = apply_universe(df, instruments, _univ())
    df = df.with_columns((pl.col("roll_vol_5m") / (pl.col("typ_1m_vol") * 5)).alias("rvol_5m"))
    hits = df.filter(
        pl.col("is_rth")
        & (pl.col("time_et") >= time(9, 40))
        & (pl.col("time_et") <= time(14, 30))
        & (pl.col("rvol_5m") >= rvol)
        & (pl.col("ret_5m").abs() <= 0.012)
    )
    return _stamp(hits, exp_id, side="long")


def eval_h007(sessioned: pl.DataFrame, instruments: pl.DataFrame, prev_close: dict[str, float], yday_high: dict[str, float], exp_id: str, *, downside: bool = False) -> pl.DataFrame:
    df = _map_num(apply_universe(_map_num(sessioned, prev_close, "prev_close"), instruments, _univ()), yday_high, "lvl")
    rth = df.filter(pl.col("is_rth") & (pl.col("time_et") >= time(9, 35)) & (pl.col("time_et") <= time(14, 30)) & pl.col("lvl").is_not_null())
    if downside:
        brk = rth.filter(pl.col("low") <= pl.col("lvl")).sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
        trap_expr = pl.col("high") >= pl.col("lvl")
        side = "long"
    else:
        brk = rth.filter(pl.col("high") >= pl.col("lvl")).sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
        trap_expr = pl.col("low") <= pl.col("lvl")
        side = "short"
    if brk.height == 0:
        return brk
    brk = brk.select(["instrument_id", pl.col("ts_utc").alias("brk_ts"), "lvl"])
    joined = rth.join(brk, on="instrument_id")
    trap = joined.filter(
        (pl.col("ts_utc") > pl.col("brk_ts"))
        & (pl.col("ts_utc") <= pl.col("brk_ts") + pl.duration(minutes=15))
        & trap_expr
    )
    hits = trap.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
    return _stamp(hits, exp_id, side=side)


def eval_h014(
    sessioned: pl.DataFrame,
    instruments: pl.DataFrame,
    prev_close: dict[str, float],
    med20_range: dict[str, float],
    exp_id: str,
    compress_frac: float = 0.35,
) -> pl.DataFrame:
    df = sessioned.sort(["instrument_id", "ts_utc"]).with_columns(
        (
            pl.col("high").rolling_max(15, min_samples=15).over("instrument_id")
            - pl.col("low").rolling_min(15, min_samples=15).over("instrument_id")
        ).alias("range_15m"),
    )
    df = _map_num(_map_num(df, prev_close, "prev_close"), med20_range, "med20")
    df = apply_universe(df, instruments, _univ())
    df = df.filter(pl.col("is_rth") & (pl.col("time_et") >= time(9, 50)) & (pl.col("time_et") <= time(14, 30)))
    comp = df.filter(
        pl.col("med20").is_not_null()
        & (pl.col("range_15m") / pl.col("prev_close") <= compress_frac * pl.col("med20"))
    )
    if comp.height == 0:
        return comp.head(0)
    first_c = comp.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first").select(
        ["instrument_id", pl.col("ts_utc").alias("c_ts"), "range_15m", "prev_close"]
    )
    later = df.join(first_c, on="instrument_id")
    expn = later.filter(
        (pl.col("ts_utc") > pl.col("c_ts"))
        & (pl.col("ts_utc") <= pl.col("c_ts") + pl.duration(minutes=30))
        & ((pl.col("high") - pl.col("low")) / pl.col("prev_close") >= 1.5 * (pl.col("range_15m") / pl.col("prev_close")))
    )
    hits = expn.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
    hits = hits.with_columns(
        pl.when(pl.col("close") >= pl.col("open")).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
    )
    return _stamp(hits, exp_id)


def eval_h022(featured: pl.DataFrame, instruments: pl.DataFrame, spec: HypothesisSpec, n_gap10: int, exp_id: str, n_min: int = 80) -> pl.DataFrame:
    if n_gap10 < n_min:
        return featured.head(0)
    return eval_h001(featured, instruments, spec, exp_id)


def eval_h028(sessioned: pl.DataFrame, instruments: pl.DataFrame, prev_close: dict[str, float], ah_ret: dict[str, float], exp_id: str, *, fade: bool = False, thr: float = 0.05) -> pl.DataFrame:
    df = apply_universe(_map_num(sessioned, prev_close, "prev_close"), instruments, _univ())
    first = df.filter(pl.col("is_rth")).sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
    first = _map_num(first, ah_ret, "ah_ret")
    hits = first.filter(pl.col("ah_ret").abs() >= thr)
    hits = hits.with_columns(
        pl.when((pl.col("ah_ret") > 0) != fade).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
    )
    return _stamp(hits, exp_id, entry_next=False)


def eval_h037(sessioned: pl.DataFrame, instruments: pl.DataFrame, prev_close: dict[str, float], yday_high: dict[str, float], yday_low: dict[str, float], exp_id: str, *, lows: bool = False) -> pl.DataFrame:
    df = apply_universe(_map_num(sessioned, prev_close, "prev_close"), instruments, _univ())
    if lows:
        df = _map_num(df, yday_low, "lvl")
        hits = df.filter(pl.col("is_rth") & (pl.col("time_et") >= time(9, 35)) & (pl.col("time_et") <= time(15, 0)) & (pl.col("low") <= pl.col("lvl")))
        side = "long"
    else:
        df = _map_num(df, yday_high, "lvl")
        hits = df.filter(pl.col("is_rth") & (pl.col("time_et") >= time(9, 35)) & (pl.col("time_et") <= time(15, 0)) & (pl.col("high") >= pl.col("lvl")))
        side = "short"
    hits = hits.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
    return _stamp(hits, exp_id, side=side)


def typ_from_hist(vol_hist: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, hist in vol_hist.items():
        if hist:
            s = sorted(hist)
            out[k] = float(s[len(s) // 2])
    return out


def med_from_hist(hist_map: dict) -> dict[str, float]:
    return typ_from_hist(hist_map)


def count_n_gap10(sessioned: pl.DataFrame, instruments: pl.DataFrame, prev_close: dict[str, float], gap: float = 0.10) -> int:
    df = apply_universe(_map_num(sessioned, prev_close, "prev_close"), instruments, _univ())
    first = df.filter(pl.col("is_rth")).sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="first")
    if first.height == 0:
        return 0
    return first.filter((pl.col("open") / pl.col("prev_close") - 1) >= gap).height


BATCH10_PRIMARY = ["H001", "H002", "H003", "H004", "H007", "H014", "H022", "H028", "H037", "H040"]

# Frozen neighborhood variants (a priori). Do not add from results.
BATCH10_VARIANTS: dict[str, list[tuple[str, dict]]] = {
    "H001": [("H001.v1", {"gap": 0.20, "rvol": 5.0}), ("H001.v2", {"gap": 0.15, "rvol": 8.0})],
    "H002": [("H002.v1", {"near": 0.005})],
    "H003": [("H003.v1", {"ret_15m": 0.20, "ret_5m": 0.10})],
    "H004": [("H004.v1", {"rvol": 8.0})],
    "H007": [("H007.v1", {"downside": True})],
    "H014": [("H014.v1", {"compress_frac": 0.25})],
    "H022": [("H022.v1", {"n_min": 120})],
    "H028": [("H028.v1", {"fade": True, "thr": 0.05}), ("H028.v2", {"fade": False, "thr": 0.08})],
    "H037": [("H037.v1", {"lows": True})],
    "H040": [
        ("H040.v1", {"monday_only": True, "midweek": False, "gap_up": False, "gap": 0.08}),
        ("H040.v2", {"monday_only": False, "midweek": True, "gap_up": True, "gap": 0.08}),
    ],
}


def expand_batch10_jobs(family_ids: list[str], *, expand_variants: bool) -> list[tuple[str, str, dict]]:
    jobs: list[tuple[str, str, dict]] = []
    for fid in family_ids:
        jobs.append((fid, "default", {}))
        if expand_variants:
            for vid, params in BATCH10_VARIANTS.get(fid, []):
                jobs.append((fid, vid, params))
    return jobs


def job_key(fid: str, vid: str) -> str:
    return fid if vid == "default" else vid


def jobs_from_keys(keys: list[str]) -> list[tuple[str, str, dict]]:
    allj = expand_batch10_jobs(list(BATCH10_PRIMARY), expand_variants=True)
    mapping = {job_key(f, v): (f, v, p) for f, v, p in allj}
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise KeyError(f"unknown batch10 jobs: {missing}")
    return [mapping[k] for k in keys]


def eval_family_day(
    fid: str,
    *,
    featured: pl.DataFrame,
    sessioned: pl.DataFrame,
    instruments: pl.DataFrame,
    spec,
    exp_id: str,
    prev_close: dict[str, float],
    typ: dict[str, float],
    yday_high: dict[str, float],
    yday_low: dict[str, float],
    ah_ret: dict[str, float],
    med20_range: dict[str, float],
    n_gap10: int,
    params: dict,
) -> pl.DataFrame:
    if fid == "H001":
        return eval_h001(featured, instruments, spec, exp_id, gap=params.get("gap", 0.15), rvol=params.get("rvol", 5.0))
    if fid == "H040":
        return eval_h040(
            featured,
            instruments,
            spec,
            exp_id,
            monday_only=params.get("monday_only", True),
            midweek=params.get("midweek", False),
            gap_up=params.get("gap_up", True),
            gap=params.get("gap", 0.08),
        )
    if fid == "H002":
        return eval_h002(
            sessioned,
            instruments,
            prev_close,
            exp_id,
            near=params.get("near", 0.003),
            side=params.get("side", "long"),
        )
    if fid == "H003":
        return eval_h003(
            sessioned,
            instruments,
            prev_close,
            typ,
            exp_id,
            ret_15m=params.get("ret_15m", 0.15),
            ret_5m=params.get("ret_5m", 0.08),
            side=params.get("side", "short"),
        )
    if fid == "H004":
        return eval_h004(sessioned, instruments, prev_close, typ, exp_id, rvol=params.get("rvol", 5.0))
    if fid == "H007":
        lvl = yday_low if params.get("downside") else yday_high
        return eval_h007(sessioned, instruments, prev_close, lvl, exp_id, downside=bool(params.get("downside")))
    if fid == "H014":
        return eval_h014(
            sessioned, instruments, prev_close, med20_range, exp_id, compress_frac=params.get("compress_frac", 0.35)
        )
    if fid == "H022":
        return eval_h022(featured, instruments, spec, n_gap10, exp_id, n_min=params.get("n_min", 80))
    if fid == "H028":
        return eval_h028(
            sessioned,
            instruments,
            prev_close,
            ah_ret,
            exp_id,
            fade=bool(params.get("fade", False)),
            thr=params.get("thr", 0.05),
        )
    if fid == "H037":
        return eval_h037(
            sessioned, instruments, prev_close, yday_high, yday_low, exp_id, lows=bool(params.get("lows", False))
        )
    raise RuntimeError(f"no batch10 evaluator for {fid}")
