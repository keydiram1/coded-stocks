"""CLOSE_DISLOCATION_REVERSAL_V1 event construction. Thresholds come from YAML/D1 freeze."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.campaigns.close_features import (
    BETA_WINDOW,
    bar_return,
    beta_from_history,
    close_volume_rvol,
    discrepancy,
    expected_final_5m,
    horizon_close_after_open,
    loo_cross_section,
    loo_daily_market,
    pick_bar_close,
    pick_close_px,
    pick_open,
    pick_window_volume,
    resolution_direction,
    signed_resolution,
)
from quant_edge_lab.discovery.v5.causality import assert_available_le_decision, assert_no_future_columns_in_events
from quant_edge_lab.discovery.v5.models import CandidateRule
from quant_edge_lab.discovery.v5.session import (
    CLOSE_ANCHOR,
    CLOSE_LAST,
    MIN_RTH_MINUTES,
    classify_trading_day,
    has_close_window,
    no_overnight_in_close_window,
    rth_only,
    with_session,
)

CAMPAIGN_ID = "CLOSE_DISLOCATION_REVERSAL_V1"
MECHANISM_ID = "forced_eod_auction_flow"


def _day_str(sessioned: pl.DataFrame) -> str:
    d = sessioned["session_date"][0]
    return str(d)


def _avail(ts) -> Any:
    if ts is None:
        return None
    return ts + timedelta(minutes=1)


def session_close_rows(
    day_bars: pl.DataFrame,
    *,
    min_rth_minutes: int = MIN_RTH_MINUTES,
) -> pl.DataFrame | None:
    sess = with_session(day_bars)
    kind = classify_trading_day(sess, min_rth_minutes=min_rth_minutes)
    if kind != "FULL_RTH":
        return None
    rth = rth_only(sess)
    rows: list[dict[str, Any]] = []
    for g in rth.partition_by("instrument_id", maintain_order=True):
        if not has_close_window(g) or not no_overnight_in_close_window(g):
            continue
        last = g.filter(pl.col("time_et") == CLOSE_LAST)
        if last.height == 0:
            continue
        last_ts = last["ts_utc"][-1]
        decision = _avail(last_ts)
        assert_available_le_decision(_avail(last_ts), decision, label="close_bar")
        obs = bar_return(pick_bar_close(g, CLOSE_LAST), pick_bar_close(g, CLOSE_ANCHOR))
        vol = pick_window_volume(g)
        daily = g.filter(pl.col("is_rth"))
        dret = None
        if daily.height:
            px = daily.sort("time_et")
            dret = bar_return(float(px["close"][-1]), float(px["close"][0]))
        rows.append(
            {
                "instrument_id": str(g["instrument_id"][0]),
                "ticker": str(g["ticker"][0]) if "ticker" in g.columns else None,
                "trading_date": _day_str(g),
                "decision_ts": decision,
                "first_available_at": decision,
                "observed_value": obs,
                "close_window_volume": vol,
                "daily_ret": dret,
                "close_px": pick_close_px(g),
            }
        )
    if not rows:
        return None
    out = pl.DataFrame(rows)
    assert_no_future_columns_in_events(out)
    rets = out["observed_value"].to_numpy().astype(float)
    mkt = loo_cross_section(rets)
    return out.with_columns(pl.Series("market_final_5m_loo", mkt))


def attach_expected(close_rows: pl.DataFrame, beta_map: dict[str, float]) -> pl.DataFrame:
    exp: list[float | None] = []
    disc: list[float | None] = []
    direc: list[str | None] = []
    betas: list[float | None] = []
    for row in close_rows.iter_rows(named=True):
        iid = str(row["instrument_id"])
        b = beta_map.get(iid)
        betas.append(float(b) if b is not None and np.isfinite(b) else None)
        e = expected_final_5m(b, row["market_final_5m_loo"])
        d = discrepancy(row["observed_value"], e)
        exp.append(e)
        disc.append(d)
        direc.append(resolution_direction(d))
    return close_rows.with_columns(
        pl.Series("beta_20d", betas, dtype=pl.Float64),
        pl.Series("expected_value", exp, dtype=pl.Float64),
        pl.Series("discrepancy", disc, dtype=pl.Float64),
        pl.Series("expected_resolution_direction", direc),
        pl.lit(MECHANISM_ID).alias("mechanism_id"),
    )


def attach_rvol(close_rows: pl.DataFrame, history: dict[str, list[tuple[str, float]]], lookback: int) -> pl.DataFrame:
    rvols: list[float | None] = []
    for row in close_rows.iter_rows(named=True):
        iid = str(row["instrument_id"])
        prior = [v for d, v in history.get(iid, []) if d < row["trading_date"]][-lookback:]
        rvols.append(close_volume_rvol(row["close_window_volume"], prior))
    return close_rows.with_columns(pl.Series("close_volume_rvol", rvols, dtype=pl.Float64))


def update_rvol_history(
    history: dict[str, list[tuple[str, float]]],
    close_rows: pl.DataFrame,
) -> dict[str, list[tuple[str, float]]]:
    for row in close_rows.iter_rows(named=True):
        vol = row["close_window_volume"]
        if vol is None:
            continue
        iid = str(row["instrument_id"])
        history.setdefault(iid, []).append((row["trading_date"], float(vol)))
    return history


def daily_panel_for_beta(close_rows: pl.DataFrame) -> pl.DataFrame:
    df = close_rows.rename({"observed_value": "ret_5m"}).select(
        ["instrument_id", "trading_date", "daily_ret"]
    )
    # beta.py daily_return_panel compounds ret_5m; we already have daily_ret.
    out = df.rename({"daily_ret": "daily_ret"}).with_columns(pl.lit(1.0).alias("weight_mkt"))
    return loo_daily_market(out)


def next_day_outcomes(today_rth: pl.DataFrame, next_rth: pl.DataFrame | None) -> dict[str, float | None]:
    close = pick_close_px(today_rth)
    if next_rth is None or next_rth.height == 0:
        return {
            "close_to_next_open": None,
            "next_open_to_5m": None,
            "next_open_to_15m": None,
            "next_open_to_30m": None,
        }
    nxt = rth_only(with_session(next_rth))
    op, _ = pick_open(nxt)
    cto = bar_return(op, close) if op is not None and close is not None else None
    c5 = horizon_close_after_open(nxt, 5)
    c15 = horizon_close_after_open(nxt, 15)
    c30 = horizon_close_after_open(nxt, 30)
    return {
        "close_to_next_open": cto,
        "next_open_to_5m": bar_return(c5, op) if c5 is not None and op is not None else None,
        "next_open_to_15m": bar_return(c15, op) if c15 is not None and op is not None else None,
        "next_open_to_30m": bar_return(c30, op) if c30 is not None and op is not None else None,
    }


def attach_outcomes_for_instrument(
    event_row: dict[str, Any],
    today_rth: pl.DataFrame,
    next_rth: pl.DataFrame | None,
) -> dict[str, Any]:
    outs = next_day_outcomes(today_rth, next_rth)
    direction = event_row.get("expected_resolution_direction")
    rec = dict(event_row)
    rec.update(outs)
    rec["signed_close_to_next_open"] = signed_resolution(outs["close_to_next_open"], direction)
    rec["signed_next_open_to_5m"] = signed_resolution(outs["next_open_to_5m"], direction)
    rec["signed_next_open_to_15m"] = signed_resolution(outs["next_open_to_15m"], direction)
    rec["signed_next_open_to_30m"] = signed_resolution(outs["next_open_to_30m"], direction)
    rec["primary_signed"] = rec["signed_next_open_to_15m"]
    return rec


def d1_quantile(values: list[float], q: float) -> float | None:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size < 20:
        return None
    return float(np.nanquantile(arr, q))


def freeze_rules_from_d1(events: pl.DataFrame, campaign_yaml: dict[str, Any]) -> list[CandidateRule]:
    abs_d = events.filter(pl.col("discrepancy").is_not_null())["discrepancy"].abs().to_list()
    rvol = events.filter(pl.col("close_volume_rvol").is_not_null())["close_volume_rvol"].to_list()
    rules: list[CandidateRule] = []
    for spec in campaign_yaml["hypotheses"]:
        dq = spec["dislocation_abs_quantile"]
        rq = spec.get("rvol_min_quantile")
        rules.append(
            CandidateRule(
                hypothesis_id=spec["hypothesis_id"],
                mechanism_id=MECHANISM_ID,
                role=spec["role"],
                dislocation_abs_min=d1_quantile(abs_d, float(dq)),
                rvol_min=d1_quantile(rvol, float(rq)) if rq is not None else None,
                require_rvol=bool(spec["require_rvol"]),
                primary_outcome=campaign_yaml["primary_outcome"],
                frozen_from_split="D1",
                frozen=True,
            )
        )
    return rules


def apply_rule(events: pl.DataFrame, rule: CandidateRule) -> pl.DataFrame:
    if rule.dislocation_abs_min is None:
        return events.head(0)
    df = events.filter(pl.col("discrepancy").abs() >= rule.dislocation_abs_min)
    if rule.require_rvol:
        if rule.rvol_min is None:
            return events.head(0)
        df = df.filter(pl.col("close_volume_rvol") >= rule.rvol_min)
    return df.with_columns(
        pl.lit(rule.hypothesis_id).alias("hypothesis_id"),
        pl.lit(rule.mechanism_id).alias("mechanism_id"),
    )


def eligible_for_event(row: dict[str, Any]) -> bool:
    if row.get("beta_20d") is None:
        return False
    if row.get("observed_value") is None or row.get("expected_value") is None:
        return False
    if row.get("discrepancy") is None or row.get("expected_resolution_direction") is None:
        return False
    return True


def build_events_from_days(
    days: list[str],
    load_day,
    load_next,
    *,
    rvol_lookback: int = 20,
    min_rth_minutes: int = MIN_RTH_MINUTES,
    progress=None,
) -> tuple[pl.DataFrame, dict[str, list[tuple[str, float]]]]:
    """load_day(day)->bars; load_next(day)->next day bars or None. Causal beta/rvol state persisted in return."""
    history_daily: list[pl.DataFrame] = []
    rvol_hist: dict[str, list[tuple[str, float]]] = {}
    parts: list[pl.DataFrame] = []
    for i, day in enumerate(days):
        bars = load_day(day)
        if bars is None or (hasattr(bars, "height") and bars.height == 0):
            if progress:
                progress(i + 1, len(days), day, 0)
            continue
        close_rows = session_close_rows(bars, min_rth_minutes=min_rth_minutes)
        if close_rows is None:
            if progress:
                progress(i + 1, len(days), day, 0)
            continue
        close_rows = attach_rvol(close_rows, rvol_hist, rvol_lookback)
        beta_map = beta_from_history(history_daily, close_rows["instrument_id"].to_list())
        close_rows = attach_expected(close_rows, beta_map)
        close_rows = close_rows.filter(pl.col("beta_20d").is_not_null())
        nxt = load_next(day)
        sess = rth_only(with_session(bars))
        nxt_by = {}
        if nxt is not None and nxt.height:
            ns = rth_only(with_session(nxt))
            for g in ns.partition_by("instrument_id"):
                nxt_by[str(g["instrument_id"][0])] = g
        enriched = []
        for g in sess.partition_by("instrument_id"):
            iid = str(g["instrument_id"][0])
            hit = close_rows.filter(pl.col("instrument_id") == iid)
            if hit.height == 0:
                continue
            rec = hit.row(0, named=True)
            if not eligible_for_event(rec):
                continue
            enriched.append(attach_outcomes_for_instrument(rec, g, nxt_by.get(iid)))
        if enriched:
            parts.append(pl.DataFrame(enriched))
        panel = daily_panel_for_beta(close_rows)
        if panel.height:
            history_daily.append(panel)
            if len(history_daily) > BETA_WINDOW:
                history_daily = history_daily[-BETA_WINDOW:]
        update_rvol_history(rvol_hist, close_rows)
        if progress:
            n_ev = parts[-1].height if parts else 0
            progress(i + 1, len(days), day, n_ev)
    if not parts:
        return pl.DataFrame(), rvol_hist
    return pl.concat(parts, how="diagonal_relaxed"), rvol_hist
