"""Build first-per-ticker-day market-residual shock events from RTH minutes."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.campaigns.close_features import bar_return
from quant_edge_lab.discovery.v5.session import classify_trading_day, rth_only, with_session
from quant_edge_lab.discovery.v6.clocks import (
    abs_z_bucket,
    entry_time,
    impulse_window_ok,
    outcome_close_time,
    p0_time,
    p1_time,
    p2_time,
    t1_bucket,
    wait_ok,
)
from quant_edge_lab.discovery.v6.history import SessionResidualHistory
from quant_edge_lab.discovery.v6.residual import (
    equal_weight_loo,
    impulse_sign,
    market_residual,
    retention_ratio,
    simple_return,
)

PRIMARY_HORIZON = 15
HistoryKey = tuple[str, str]


def _closes(g: pl.DataFrame) -> dict[time, float]:
    out: dict[time, float] = {}
    for row in g.select(["time_et", "close"]).iter_rows(named=True):
        out[row["time_et"]] = float(row["close"])
    return out


def _opens(g: pl.DataFrame) -> dict[time, float]:
    out: dict[time, float] = {}
    for row in g.select(["time_et", "open"]).iter_rows(named=True):
        out[row["time_et"]] = float(row["open"])
    return out


def candidate_impulse_starts(w_imp: int) -> list[time]:
    starts: list[time] = []
    for h in range(10, 15):
        for m in range(60):
            t = time(h, m)
            if impulse_window_ok(t, w_imp):
                starts.append(t)
    return starts


def candidate_starts(w_imp: int, w_wait: int) -> list[time]:
    return [t for t in candidate_impulse_starts(w_imp) if wait_ok(t, w_imp, w_wait)]


def impulse_clock_key(w_imp: int, t0: time) -> str:
    return f"{w_imp}:{t0.strftime('%H:%M')}"


def window_peer_ids(closes: dict[str, dict[time, float]], p_a: time, p_b: time) -> list[str]:
    return [iid for iid, c in closes.items() if p_a in c and p_b in c]


def window_residuals(
    closes: dict[str, dict[time, float]],
    peers: list[str],
    p_a: time,
    p_b: time,
    betas: dict[str, float],
) -> dict[str, float]:
    """LOO market residual on peers that have both window prices. Need >=2 names."""
    if len(peers) < 2:
        return {}
    r = np.array([simple_return(closes[i][p_b], closes[i][p_a]) for i in peers], dtype=float)
    mkt = equal_weight_loo(r)
    out: dict[str, float] = {}
    for j, iid in enumerate(peers):
        resid = market_residual(
            float(r[j]) if np.isfinite(r[j]) else None,
            betas.get(iid),
            float(mkt[j]) if np.isfinite(mkt[j]) else None,
        )
        if resid is not None:
            out[iid] = float(resid)
    return out


def signed_forward(fwd: float | None, sign: str | None, *, reverse: bool = False) -> float | None:
    if fwd is None or sign is None or not np.isfinite(float(fwd)):
        return None
    s = 1.0 if sign == "POSITIVE" else -1.0
    if reverse:
        s = -s
    return s * float(fwd)


def compute_impulse_map(
    closes: dict[str, dict[time, float]],
    *,
    w_imp: int,
    betas: dict[str, float],
) -> dict[HistoryKey, float]:
    """Impulse residuals vs P0→P1 eligible LOO. Independent of W_wait / P2."""
    day_map: dict[HistoryKey, float] = {}
    for t0 in candidate_impulse_starts(w_imp):
        p0t, p1t = p0_time(t0), p1_time(t0, w_imp)
        peers = window_peer_ids(closes, p0t, p1t)
        resid = window_residuals(closes, peers, p0t, p1t, betas)
        ck = impulse_clock_key(w_imp, t0)
        for iid, imp in resid.items():
            day_map[(iid, ck)] = imp
    return day_map


def _eligible_closes(
    day_bars: pl.DataFrame, eligible: set[str] | None, min_rth_minutes: int
) -> tuple | None:
    sess = with_session(day_bars)
    if classify_trading_day(sess, min_rth_minutes=min_rth_minutes) != "FULL_RTH":
        return None
    rth = rth_only(sess)
    groups = {str(g["instrument_id"][0]): g for g in rth.partition_by("instrument_id")}
    if eligible is not None:
        groups = {iid: g for iid, g in groups.items() if iid in eligible}
    closes = {iid: _closes(g) for iid, g in groups.items()}
    opens = {iid: _opens(g) for iid, g in groups.items()}
    return rth, groups, closes, opens


def process_day(
    day_bars: pl.DataFrame,
    cell: dict[str, Any],
    *,
    betas: dict[str, float],
    history: SessionResidualHistory,
    eligible: set[str] | None = None,
    min_rth_minutes: int = 380,
    impulse_map: dict[HistoryKey, float] | None = None,
) -> tuple[pl.DataFrame, dict[HistoryKey, float]]:
    """Events for this cell. Impulse map is P0→P1 only; remaining uses P0→P2 LOO."""
    parsed = _eligible_closes(day_bars, eligible, min_rth_minutes)
    if parsed is None:
        return pl.DataFrame(), {}
    rth, groups, closes, opens = parsed
    w_imp = int(cell["impulse_window_minutes"])
    w_wait = int(cell["wait_window_minutes"])
    z_min = float(cell["min_abs_shock_z"])
    high = float(cell["high_retention_min"])
    low = float(cell["low_retention_max"])
    if impulse_map is None:
        impulse_map = compute_impulse_map(closes, w_imp=w_imp, betas=betas)
    taken: set[str] = set()
    rows: list[dict[str, Any]] = []
    day = str(rth["session_date"][0])
    for t0 in candidate_starts(w_imp, w_wait):
        p0t, p1t, p2t = p0_time(t0), p1_time(t0, w_imp), p2_time(t0, w_imp, w_wait)
        rem_peers = window_peer_ids(closes, p0t, p2t)
        rem_by = window_residuals(closes, rem_peers, p0t, p2t, betas)
        ck = impulse_clock_key(w_imp, t0)
        t1 = p2t
        ent = entry_time(t1)
        oc15 = outcome_close_time(ent, PRIMARY_HORIZON)
        oc5 = outcome_close_time(ent, 5)
        oc30 = outcome_close_time(ent, 30)
        targets = [
            iid
            for iid, c in closes.items()
            if p0t in c and p1t in c and p2t in c and iid not in taken
        ]
        for iid in targets:
            imp = impulse_map.get((iid, ck))
            rem = rem_by.get(iid)
            z = history.shock_z_for(iid, ck, imp)
            if z is None or abs(z) < z_min:
                continue
            sgn = impulse_sign(imp)
            if sgn is None:
                continue
            z_b = abs_z_bucket(abs(z))
            tod = t1_bucket(t1)
            if z_b is None or tod is None:
                continue
            taken.add(iid)
            ret = retention_ratio(rem, imp)
            e_px = opens[iid].get(ent)
            fwd15 = bar_return(closes[iid].get(oc15), e_px) if e_px is not None else None
            fwd5 = bar_return(closes[iid].get(oc5), e_px) if e_px is not None else None
            fwd30 = bar_return(closes[iid].get(oc30), e_px) if e_px is not None else None
            is_h1 = ret is not None and ret >= high
            is_h2 = ret is not None and ret <= low
            g = groups[iid]
            rows.append(
                {
                    "instrument_id": iid,
                    "ticker": str(g["ticker"][0]) if "ticker" in g.columns else iid,
                    "trading_date": day,
                    "cell_id": cell["id"],
                    "impulse_start": t0.strftime("%H:%M"),
                    "t1": t1.strftime("%H:%M"),
                    "entry": ent.strftime("%H:%M"),
                    "impulse_residual": imp,
                    "remaining_residual": rem,
                    "retention": ret,
                    "shock_z": z,
                    "impulse_sign": sgn,
                    "abs_z_bucket": z_b,
                    "t1_bucket": tod,
                    "forward_5m": fwd5,
                    "forward_15m": fwd15,
                    "forward_30m": fwd30,
                    "primary_signed": signed_forward(fwd15, sgn, reverse=False),
                    "h2_signed": signed_forward(fwd15, sgn, reverse=True),
                    "is_h1": is_h1,
                    "is_h2": is_h2,
                    "is_h3": True,
                    "is_incremental_control": bool(ret is not None and ret < high),
                }
            )
    if not rows:
        return pl.DataFrame(), impulse_map
    return pl.DataFrame(rows), impulse_map


def filter_hypothesis(events: pl.DataFrame, hid: str) -> pl.DataFrame:
    if events.height == 0:
        return events
    if hid == "H1_HIGH_RETENTION_CONTINUATION":
        return events.filter(pl.col("is_h1"))
    if hid == "H2_LOW_RETENTION_REVERSAL":
        return events.filter(pl.col("is_h2")).with_columns(
            pl.col("h2_signed").alias("primary_signed")
        )
    if hid == "H3_IMPULSE_ONLY_CONTROL":
        return events.filter(pl.col("is_h3"))
    raise KeyError(hid)


def incremental_frames(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if events.height == 0:
        return events, events
    treated = events.filter(pl.col("is_h1"))
    control = events.filter(pl.col("is_incremental_control"))
    return treated, control
