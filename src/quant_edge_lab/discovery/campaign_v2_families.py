"""Campaign v2 families A–J and baselines. Decision at confirmation close; side set then."""

from __future__ import annotations

from datetime import time

import numpy as np
import polars as pl

from quant_edge_lab.discovery.eval_batch10 import _stamp

THR = {
    "A_impulse_ret": 0.04,
    "A_impulse_ret_strict": 0.06,
    "A_rvol_10": 3.0,
    "A_pullback_lo": 0.15,
    "A_pullback_hi": 0.50,
    "A_observe_m": 15,
    "B_observe_m": 10,
    "B_reject_pct": 0.005,
    "B_accept_closes": 3,
    "C_tail": 0.05,
    "C_persist_band": 0.10,
    "C_decay_out": 0.15,
    "D_mkt_stress": -0.003,
    "D_breadth_stress": 0.40,
    "D_flat": 0.001,
    "D_rvol": 2.0,
    "D_stabilize": 0.001,
    "E_observe_m": 10,
    "F_runner": 0.15,
    "G_extreme": 0.05,
    "G_collapse": 0.20,
    "H_compress": 0.40,
    "I_sep_m": 5,
    "I_fail_pct": 0.003,
    "J_short": 0.008,
    "J_med": 0.015,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame()


def _from_idx(feat: pl.DataFrame, idx: list[int], side: str, exp_id: str) -> pl.DataFrame:
    if not idx:
        return _empty()
    hits = feat[idx]
    if "side" not in hits.columns:
        hits = hits.with_columns(pl.lit(side).alias("side"))
    return _stamp(hits, exp_id)


def eval_a(feat: pl.DataFrame, exp_id: str, *, strict: bool = False, impulse_only: bool = False) -> pl.DataFrame:
    thr = THR["A_impulse_ret_strict"] if strict else THR["A_impulse_ret"]
    if feat.height == 0:
        return feat
    f = feat.filter((pl.col("time_et") >= time(9, 50)) & (pl.col("time_et") <= time(14, 30)))
    if impulse_only:
        hits = f.filter((pl.col("ret_10m").abs() >= thr) & (pl.col("rvol_10m") >= THR["A_rvol_10"]))
        hits = hits.with_columns(
            pl.when(pl.col("ret_10m") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
        return _stamp(hits, exp_id)
    keep = []
    for g in f.partition_by("instrument_id", maintain_order=True):
        n = g.height
        if n < 30:
            continue
        ret = g["ret_10m"].to_numpy()
        rvol = g["rvol_10m"].to_numpy()
        close = g["close"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        vol = g["volume_1m"].to_numpy()
        hi3 = g["hi_3"].to_numpy() if "hi_3" in g.columns else high
        lo3 = g["lo_3"].to_numpy() if "lo_3" in g.columns else low
        used = False
        for i in range(10, n - 5):
            if used:
                break
            r = ret[i]
            if not np.isfinite(r) or abs(r) < thr or not np.isfinite(rvol[i]) or rvol[i] < THR["A_rvol_10"]:
                continue
            d = 1 if r > 0 else -1
            imp_abs = abs(r)
            imp_c = close[i]
            imp_vol = float(np.mean(vol[max(0, i - 9) : i + 1]))
            end = min(n, i + 1 + int(THR["A_observe_m"]))
            pb_j = None
            for j in range(i + 1, end):
                move = (close[j] - imp_c) * d
                # counter-move is negative in direction D
                if move >= 0:
                    continue
                frac = abs(close[j] - imp_c) / (imp_abs * imp_c + 1e-12) if imp_c else 0
                # impulse mag in price ≈ imp_abs * price; pullback as fraction of impulse
                frac = abs(close[j] - imp_c) / (abs(r) * close[i] + 1e-12)
                if THR["A_pullback_lo"] <= frac <= THR["A_pullback_hi"]:
                    pb_vol = float(np.mean(vol[i + 1 : j + 1])) if j > i else imp_vol
                    if pb_vol < imp_vol:
                        pb_j = j
                        break
            if pb_j is None:
                continue
            for k in range(pb_j + 1, min(n, pb_j + 12)):
                if d > 0 and close[k] > hi3[k - 1]:
                    keep.append((g, k, "long"))
                    used = True
                    break
                if d < 0 and close[k] < lo3[k - 1]:
                    keep.append((g, k, "short"))
                    used = True
                    break
    if not keep:
        return _empty()
    parts = []
    for g, k, side in keep:
        row = g.slice(k, 1).with_columns(pl.lit(side).alias("side"))
        parts.append(row)
    return _stamp(pl.concat(parts, how="diagonal_relaxed"), exp_id)


def eval_b(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    if feat.height == 0:
        return feat
    f = feat.filter((pl.col("time_et") >= time(9, 40)) & (pl.col("time_et") <= time(14, 30)))
    keep = []
    up = "up" in job_id
    accept = job_id.startswith("B.accept")
    for g in f.partition_by("instrument_id", maintain_order=True):
        lvl = g["yday_high"].to_numpy() if up else g["yday_low"].to_numpy()
        close = g["close"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        rvol = g["rvol_5m"].to_numpy()
        n = g.height
        broke = None
        for i in range(n):
            if not np.isfinite(lvl[i]):
                continue
            if up and high[i] > lvl[i]:
                broke = i
                break
            if (not up) and low[i] < lvl[i]:
                broke = i
                break
        if broke is None:
            continue
        window = range(broke, min(n, broke + int(THR["B_observe_m"])))
        closes_ok = 0
        rejected = False
        rej_i = None
        last_ok = None
        for j in window:
            if up:
                if close[j] >= lvl[j] * (1 - 0.0):
                    if close[j] > lvl[j]:
                        closes_ok += 1
                        last_ok = j
                if close[j] <= lvl[j] * (1 - THR["B_reject_pct"]):
                    rejected = True
                    rej_i = j
                    break
            else:
                if close[j] < lvl[j]:
                    closes_ok += 1
                    last_ok = j
                if close[j] >= lvl[j] * (1 + THR["B_reject_pct"]):
                    rejected = True
                    rej_i = j
                    break
        vol_ok = bool(np.nanmean(rvol[broke : min(n, broke + 10)]) >= 1.2)
        if accept and (not rejected) and closes_ok >= int(THR["B_accept_closes"]) and vol_ok and last_ok is not None:
            keep.append((g, last_ok, "long" if up else "short"))
        if (not accept) and rejected and rej_i is not None:
            keep.append((g, rej_i, "short" if up else "long"))
    if not keep:
        return _empty()
    parts = [g.slice(k, 1).with_columns(pl.lit(s).alias("side")) for g, k, s in keep]
    return _stamp(pl.concat(parts, how="diagonal_relaxed"), exp_id)


def eval_c(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    snaps = feat.filter((pl.col("time_et") >= time(9, 45)) & pl.col("xs_ret_rank").is_not_null() & pl.col("ret_15m").is_not_null())
    if "minutes_from_open" in snaps.columns:
        snaps = snaps.filter(pl.col("minutes_from_open") % 5 == 0)
    if snaps.height == 0:
        return snaps
    snaps = snaps.sort(["instrument_id", "ts_utc"]).with_columns(
        pl.col("xs_ret_rank").shift(1).over("instrument_id").alias("rank_5"),
        pl.col("xs_ret_rank").shift(2).over("instrument_id").alias("rank_10"),
    )
    tail = THR["C_tail"]
    persist = THR["C_persist_band"]
    decay = THR["C_decay_out"]
    if job_id == "C.persist_lead":
        hits = snaps.filter(
            (pl.col("xs_ret_rank") >= 1 - persist)
            & (pl.col("rank_5") >= 1 - tail)
            & (pl.col("xs_ret_rank") >= pl.col("rank_5") - 0.03)
        )
        return _stamp(hits, exp_id, side="long")
    if job_id == "C.decay_lead":
        hits = snaps.filter(
            (pl.col("rank_5") >= 1 - tail)
            & (pl.col("xs_ret_rank") < 1 - decay)
            & (pl.col("ret_15m") > 0)
        )
        return _stamp(hits, exp_id, side="short")
    if job_id == "C.persist_lag":
        hits = snaps.filter(
            (pl.col("xs_ret_rank") <= persist)
            & (pl.col("rank_5") <= tail)
            & (pl.col("xs_ret_rank") <= pl.col("rank_5") + 0.03)
        )
        return _stamp(hits, exp_id, side="short")
    if job_id == "C.recover_lag":
        hits = snaps.filter((pl.col("rank_5") <= tail) & (pl.col("xs_ret_rank") > decay) & (pl.col("ret_15m") < 0))
        return _stamp(hits, exp_id, side="long")
    if job_id == "C.raw_mom_long":
        return _stamp(snaps.filter(pl.col("xs_ret_rank") >= 1 - tail), exp_id, side="long")
    if job_id == "C.raw_mom_short":
        return _stamp(snaps.filter(pl.col("xs_ret_rank") <= tail), exp_id, side="short")
    raise RuntimeError(job_id)


def eval_d(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    f = feat.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    if f.height == 0:
        return f
    long = job_id.endswith("long")
    if long:
        setup = f.filter(
            (pl.col("mkt_ret_5m") <= THR["D_mkt_stress"])
            | (pl.col("breadth_pos") <= THR["D_breadth_stress"])
        ).filter(
            (pl.col("ret_5m") >= -THR["D_flat"])
            & (pl.col("xs_ret_rank") >= 0.90)
            & (pl.col("rvol_5m") >= THR["D_rvol"])
        )
        conf = setup.filter((pl.col("mkt_ret_5m").abs() <= THR["D_stabilize"]) | (pl.col("close") >= pl.col("hi_5")))
        return _stamp(conf, exp_id, side="long")
    setup = f.filter(
        (pl.col("mkt_ret_5m") >= -THR["D_mkt_stress"]) | (pl.col("breadth_neg") <= THR["D_breadth_stress"])
    ).filter((pl.col("ret_5m") <= THR["D_flat"]) & (pl.col("xs_ret_rank") <= 0.10) & (pl.col("rvol_5m") >= THR["D_rvol"]))
    conf = setup.filter((pl.col("mkt_ret_5m").abs() <= THR["D_stabilize"]) | (pl.col("close") <= pl.col("lo_5")))
    return _stamp(conf, exp_id, side="short")


def eval_e(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    f = feat.filter((pl.col("time_et") >= time(9, 40)) & (pl.col("time_et") <= time(14, 30)))
    if job_id == "E.h004_signed":
        hits = f.filter((pl.col("rvol_5m") >= 5.0) & (pl.col("ret_5m").abs() <= 0.012))
        hits = hits.with_columns(pl.lit("long").alias("side"))
        return _stamp(hits, exp_id)
    keep = []
    for g in f.partition_by("instrument_id", maintain_order=True):
        rvol = g["rvol_5m"].to_numpy()
        ret5 = g["ret_5m"].to_numpy()
        close = g["close"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        hi5 = g["hi_5"].to_numpy()
        lo5 = g["lo_5"].to_numpy()
        n = g.height
        shock = None
        for i in range(n):
            if np.isfinite(rvol[i]) and rvol[i] >= 5.0 and np.isfinite(ret5[i]) and abs(ret5[i]) <= 0.012:
                shock = i
                break
        if shock is None:
            continue
        pre_hi = hi5[shock]
        pre_lo = lo5[shock]
        for j in range(shock + 1, min(n, shock + 1 + int(THR["E_observe_m"]))):
            if job_id == "E.res_long" and high[j] > pre_hi and close[j] > pre_hi and rvol[j] >= 3:
                keep.append((g, j, "long"))
                break
            if job_id == "E.res_short" and low[j] < pre_lo and close[j] < pre_lo and rvol[j] >= 3:
                keep.append((g, j, "short"))
                break
            if job_id == "E.fail_up" and high[j] > pre_hi and close[j] < pre_hi and close[j] > pre_lo:
                keep.append((g, j, "short"))
                break
            if job_id == "E.fail_dn" and low[j] < pre_lo and close[j] > pre_lo and close[j] < pre_hi:
                keep.append((g, j, "long"))
                break
    if not keep:
        return _empty()
    parts = [g.slice(k, 1).with_columns(pl.lit(s).alias("side")) for g, k, s in keep]
    return _stamp(pl.concat(parts, how="diagonal_relaxed"), exp_id)


def eval_f(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    f = feat.filter((pl.col("time_et") >= time(9, 40)) & (pl.col("time_et") <= time(14, 0)))
    pos = job_id in {"F.cont_long", "F.fail_short"}
    runner = (pl.col("ret_3d") >= THR["F_runner"]) if pos else (pl.col("ret_3d") <= -THR["F_runner"])
    base = f.filter(runner)
    if job_id == "F.cont_long":
        hits = base.filter((pl.col("close") >= pl.col("rth_open") * 0.995) & (pl.col("new_high_15")) & (pl.col("rvol_5m") >= 2))
        return _stamp(hits, exp_id, side="long")
    if job_id == "F.cont_short":
        hits = base.filter((pl.col("close") <= pl.col("rth_open") * 1.005) & (pl.col("new_low_15")) & (pl.col("rvol_5m") >= 2))
        return _stamp(hits, exp_id, side="short")
    if job_id == "F.fail_short":
        hits = base.filter((pl.col("close") < pl.col("rth_open")) & (pl.col("ret_5m") > 0))
        # lose open after upward attempt: previous 5m positive then close below open
        hits = hits.filter(pl.col("close") < pl.col("rth_open"))
        return _stamp(hits, exp_id, side="short")
    if job_id == "F.fail_long":
        hits = base.filter((pl.col("close") > pl.col("rth_open")) & (pl.col("ret_5m") < 0))
        return _stamp(hits, exp_id, side="long")
    raise RuntimeError(job_id)


def eval_g(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    open_bar = feat.filter((pl.col("time_et") >= time(9, 30)) & (pl.col("time_et") <= time(9, 34)))
    if open_bar.height == 0:
        return open_bar
    snap = open_bar.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="last")
    n = snap.height
    if n < 20:
        return _empty()
    snap = snap.with_columns(
        (pl.col("gap_pct").rank() / n).alias("gap_rank"),
        (pl.col("ret_5m").rank() / n).alias("open_ret_rank"),
        (pl.col("rvol_5m").rank() / n).alias("open_rvol_rank"),
    )
    ext = THR["G_extreme"]
    leaders = snap.filter((pl.col("gap_rank") >= 1 - ext) | (pl.col("open_ret_rank") >= 1 - ext))
    later = feat.filter((pl.col("time_et") >= time(9, 35)) & (pl.col("time_et") <= time(9, 49)))
    later = later.join(
        leaders.select(["instrument_id", "gap_pct", "rth_open", "open_ret_rank", "xs_ret_rank"]),
        on="instrument_id",
        how="inner",
        suffix="_open",
    )
    if later.height == 0:
        return later
    last = later.sort(["instrument_id", "ts_utc"]).unique(subset=["instrument_id"], keep="last")
    if job_id == "G.cont":
        hits = last.filter((pl.col("xs_ret_rank") >= 1 - ext) & (pl.col("close") >= pl.col("rth_open")))
        return _stamp(hits, exp_id, side="long")
    hits = last.filter((pl.col("xs_ret_rank") < 1 - THR["G_collapse"]) & (pl.col("close") < pl.col("rth_open")))
    return _stamp(hits, exp_id, side="short")


def eval_h(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    f = feat.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    f = f.with_columns(
        (pl.col("range_15m") <= THR["H_compress"] * pl.col("med20")).alias("compress"),
    )
    f = f.with_columns(pl.col("compress").shift(5).over("instrument_id").alias("c5"))
    if job_id == "H.rel_long":
        hits = f.filter(pl.col("c5") & (pl.col("close") > pl.col("hi_15").shift(5).over("instrument_id")) & (pl.col("rvol_5m") >= 2))
        return _stamp(hits, exp_id, side="long")
    hits = f.filter(pl.col("c5") & (pl.col("close") < pl.col("lo_15").shift(5).over("instrument_id")) & (pl.col("rvol_5m") >= 2))
    return _stamp(hits, exp_id, side="short")


def eval_i(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    f = feat.filter((pl.col("time_et") >= time(9, 45)) & (pl.col("time_et") <= time(14, 30)))
    up = "up" in job_id
    success = "succ" in job_id
    keep = []
    sep = int(THR["I_sep_m"])
    for g in f.partition_by("instrument_id", maintain_order=True):
        lvl = g["yday_high"].to_numpy() if up else g["yday_low"].to_numpy()
        high = g["high"].to_numpy()
        low = g["low"].to_numpy()
        close = g["close"].to_numpy()
        n = g.height
        first = None
        for i in range(n):
            if not np.isfinite(lvl[i]):
                continue
            touch = high[i] >= lvl[i] if up else low[i] <= lvl[i]
            if not touch:
                continue
            if first is None:
                # rejection: close back
                if up and close[i] < lvl[i]:
                    first = i
                if (not up) and close[i] > lvl[i]:
                    first = i
                continue
            if i < first + sep:
                continue
            if up:
                if success and close[i] > lvl[i]:
                    keep.append((g, i, "long"))
                    break
                if (not success) and high[i] >= lvl[i] * (1 - THR["I_fail_pct"]) and close[i] < lvl[i] * (1 - THR["I_fail_pct"]):
                    keep.append((g, i, "short"))
                    break
            else:
                if success and close[i] < lvl[i]:
                    keep.append((g, i, "short"))
                    break
                if (not success) and low[i] <= lvl[i] * (1 + THR["I_fail_pct"]) and close[i] > lvl[i] * (1 + THR["I_fail_pct"]):
                    keep.append((g, i, "long"))
                    break
    if not keep:
        return _empty()
    parts = [g.slice(k, 1).with_columns(pl.lit(s).alias("side")) for g, k, s in keep]
    return _stamp(pl.concat(parts, how="diagonal_relaxed"), exp_id)


def eval_j(feat: pl.DataFrame, exp_id: str, job_id: str) -> pl.DataFrame:
    f = feat.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    s = THR["J_short"]
    m = THR["J_med"]
    if job_id == "J.agree":
        hits = f.filter(
            (pl.col("ret_5m").sign() == pl.col("ret_15m").sign())
            & (pl.col("ret_15m").sign() == pl.col("yday_ret").sign())
            & (pl.col("ret_5m").abs() >= s)
            & (pl.col("rvol_5m") >= 2)
            & (pl.col("close") >= pl.col("hi_3").shift(1).over("instrument_id"))
        )
        hits = hits.with_columns(
            pl.when(pl.col("ret_5m") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
        return _stamp(hits, exp_id)
    hits = f.filter(
        (pl.col("ret_5m").sign() != pl.col("yday_ret").sign())
        & (pl.col("ret_5m").abs() >= s)
        & (pl.col("ret_15m").abs() < m)
        & (pl.col("yday_ret").abs() >= 0.01)
    )
    hits = hits.with_columns(
        pl.when(pl.col("yday_ret") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
    )
    return _stamp(hits, exp_id)


MODEL_FEATS = [
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_60m",
    "acceleration_5_vs_15",
    "path_efficiency_15m",
    "range_position_15m",
    "close_location_value",
    "rvol_5m",
    "rvol_15m",
    "volume_acceleration",
    "range_5m",
    "range_vs_hist",
    "gap_pct",
    "minutes_from_open",
    "yday_ret",
    "ret_3d",
    "dist_yday_high",
    "xs_ret_rank",
    "xs_rvol_rank",
    "mkt_ret_5m",
    "breadth_pos",
    "px_disp_per_rvol",
]


def snapshot_matrix(feat: pl.DataFrame, times: list[time]) -> pl.DataFrame:
    return feat.filter(pl.col("time_et").is_in(times))


def eval_job(job_id: str, feat: pl.DataFrame, exp_id: str) -> pl.DataFrame:
    if feat.height == 0 or "rvol_5m" not in feat.columns:
        return _empty()
    if job_id == "A.impulse_pb":
        return eval_a(feat, exp_id, strict=False)
    if job_id == "A.impulse_pb_6":
        return eval_a(feat, exp_id, strict=True)
    if job_id == "A.impulse_only":
        return eval_a(feat, exp_id, impulse_only=True)
    if job_id.startswith("B."):
        return eval_b(feat, exp_id, job_id)
    if job_id.startswith("C."):
        return eval_c(feat, exp_id, job_id)
    if job_id.startswith("D."):
        return eval_d(feat, exp_id, job_id)
    if job_id.startswith("E."):
        return eval_e(feat, exp_id, job_id)
    if job_id.startswith("F."):
        return eval_f(feat, exp_id, job_id)
    if job_id.startswith("G."):
        return eval_g(feat, exp_id, job_id)
    if job_id.startswith("H."):
        return eval_h(feat, exp_id, job_id)
    if job_id.startswith("I."):
        return eval_i(feat, exp_id, job_id)
    if job_id.startswith("J."):
        return eval_j(feat, exp_id, job_id)
    if job_id == "K.mom_base":
        hits = snapshot_matrix(feat, [time(10, 0), time(11, 0), time(12, 0), time(13, 0), time(14, 0)])
        hits = hits.filter(pl.col("ret_15m").abs() >= 0.005).with_columns(
            pl.when(pl.col("ret_15m") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side")
        )
        return _stamp(hits, exp_id)
    return _empty()
