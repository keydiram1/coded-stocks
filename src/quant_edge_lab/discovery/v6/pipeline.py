"""D1-only day walk: causal eligibility, V5 beta, four frozen cells. No D2/D3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import daily_panel_from_session
from quant_edge_lab.discovery.v5.eligibility import (
    EligibilityConfig,
    causal_ok,
    is_listed,
    listed_ids,
    rth_dollar_volume,
    update_elig_hist,
)
from quant_edge_lab.discovery.v5.session import (
    CLOSE_LAST,
    classify_trading_day,
    rth_only,
    with_session,
)
from quant_edge_lab.discovery.v6.calendar import assert_no_future_splits, d1_loadable_days
from quant_edge_lab.discovery.v6.design import campaign_block, frozen_cells
from quant_edge_lab.discovery.v6.evaluate import evaluate_hypotheses
from quant_edge_lab.discovery.v6.events import process_day
from quant_edge_lab.discovery.v6.history import HistoryKey, SessionResidualHistory
from quant_edge_lab.discovery.v6.selection import select_d1_cell
from quant_edge_lab.features.v4r.beta import BETA_WINDOW, beta_from_history

LoadDay = Callable[[str], pl.DataFrame | None]


def eligibility_cfg(camp: dict[str, Any]) -> EligibilityConfig:
    el = camp["eligibility"]
    return EligibilityConfig(
        min_prev_close=float(el["price_prev_close_min"]),
        min_median_rth_dvol=float(el["trailing_20d_median_rth_dollar_volume_min"]),
        min_dvol_days=BETA_WINDOW,
        lookback=BETA_WINDOW,
    )


def _close_1559(g: pl.DataFrame) -> float | None:
    sub = g.filter(pl.col("time_et") == CLOSE_LAST) if "time_et" in g.columns else g.head(0)
    if sub.height == 0:
        return None
    v = float(sub["close"][-1])
    return v if np.isfinite(v) else None


def eligible_today(
    iids: list[str],
    day: str,
    *,
    listed: set[str] | None,
    elig_hist: dict[str, dict[str, list[tuple[str, float]]]],
    cfg: EligibilityConfig,
) -> set[str]:
    out: set[str] = set()
    for iid in iids:
        if not is_listed(iid, listed):
            continue
        if not causal_ok(iid, day, elig_hist, cfg):
            continue
        out.add(iid)
    return out


def walk_d1(
    days: list[str],
    load_day: LoadDay,
    *,
    calendar: list[str],
    cal_man: dict[str, Any],
    des: dict[str, Any],
    instruments: pl.DataFrame | None,
    emit_splits: frozenset[str] = frozenset({"D1"}),
    min_rth_minutes: int = 380,
) -> dict[str, Any]:
    """Process warmup+D1 only. Emits events only on emit_splits (D1)."""
    from quant_edge_lab.discovery.v5.partitions import split_for_day

    assert_no_future_splits(days, cal_man)
    camp = campaign_block(des)
    cells = frozen_cells(des)
    cfg = eligibility_cfg(camp)
    listed = listed_ids(instruments, cfg)
    elig_hist: dict[str, dict[str, list[tuple[str, float]]]] = {}
    beta_panels: list[pl.DataFrame] = []
    w_imps = sorted({int(c["impulse_window_minutes"]) for c in cells})
    histories = {w: SessionResidualHistory() for w in w_imps}
    events_by_cell: dict[str, list[pl.DataFrame]] = {c["id"]: [] for c in cells}
    for day in days:
        impulse_today: dict[int, dict[HistoryKey, float]] = {}
        if day not in calendar:
            raise RuntimeError(f"{day} is not on the pinned research calendar")
        bars = load_day(day)
        if bars is None or bars.height == 0:
            for hist in histories.values():
                hist.push({})
            continue
        sess = with_session(bars)
        if classify_trading_day(sess, min_rth_minutes=min_rth_minutes) != "FULL_RTH":
            for hist in histories.values():
                hist.push({})
            continue
        rth = rth_only(sess)
        iids = [str(x) for x in rth["instrument_id"].unique().to_list()]
        elig = eligible_today(iids, day, listed=listed, elig_hist=elig_hist, cfg=cfg)
        beta_map = beta_from_history(beta_panels, sorted(elig)) if elig else {}
        split = split_for_day(day, cal_man)
        for cell in cells:
            ev, day_map = process_day(
                bars,
                cell,
                betas=beta_map,
                history=histories[int(cell["impulse_window_minutes"])],
                eligible=elig,
                min_rth_minutes=min_rth_minutes,
                impulse_map=impulse_today.get(int(cell["impulse_window_minutes"])),
            )
            w_imp = int(cell["impulse_window_minutes"])
            impulse_today[w_imp] = day_map
            if split in emit_splits and ev.height:
                events_by_cell[cell["id"]].append(ev)
        for w_imp in histories:
            histories[w_imp].push(impulse_today.get(w_imp, {}))
        panel = daily_panel_from_session(sess, eligible=listed)
        if panel.height:
            beta_panels.append(panel)
            if len(beta_panels) > BETA_WINDOW:
                beta_panels = beta_panels[-BETA_WINDOW:]
        for iid in iids:
            if listed is not None and iid not in listed:
                continue
            g = rth.filter(pl.col("instrument_id") == iid)
            update_elig_hist(
                elig_hist,
                iid=iid,
                day=day,
                close_1559=_close_1559(g),
                rth_dvol=rth_dollar_volume(g),
                lookback=cfg.lookback,
            )
    sample = camp["sample"]
    inf = camp["inference"]
    n_boot = int(inf["bootstrap_daily_means"])
    seed = int(inf["seed"])
    bh_q = float(inf["bh_q"])
    cell_rows: list[dict[str, Any]] = []
    evaluated: dict[str, Any] = {}
    for cell in cells:
        parts = events_by_cell[cell["id"]]
        ev = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
        evl = evaluate_hypotheses(ev, n_boot=n_boot, seed=seed, bh_q=bh_q)
        evaluated[cell["id"]] = evl
        cell_rows.append(
            {
                "cell": cell,
                "h1_stats": evl["h1_stats"],
                "incremental": evl["incremental"],
            }
        )
    selection = select_d1_cell(cell_rows, sample)
    return {
        "split": "D1",
        "stopped_after": "D1",
        "d2_opened": False,
        "d3_opened": False,
        "cells": evaluated,
        "selection": selection,
        "primary_estimand": "equal_weight_trading_day_mean",
    }


def d1_days_from_calendar(calendar: list[str], cal_man: dict[str, Any]) -> list[str]:
    days = d1_loadable_days(calendar, cal_man)
    assert_no_future_splits(days, cal_man)
    return days
