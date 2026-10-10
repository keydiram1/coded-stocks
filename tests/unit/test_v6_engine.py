from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_edge_lab.discovery.v5.continuation_stats import equal_weight_day_stats
from quant_edge_lab.discovery.v5.evaluation import day_block_stats
from quant_edge_lab.discovery.v6.calendar import (
    EXPECTED_CALENDAR_HASH,
    EXPECTED_CALENDAR_N_DAYS,
    calendar_manifest,
    d1_loadable_days,
)
from quant_edge_lab.discovery.v6.clocks import decision_ts_entry, entry_time, t1_bucket
from quant_edge_lab.discovery.v6.design import frozen_cells, load_v6
from quant_edge_lab.discovery.v6.evaluate import evaluate_hypotheses
from quant_edge_lab.discovery.v6.events import (
    process_day,
    signed_forward,
    window_peer_ids,
    window_residuals,
)
from quant_edge_lab.discovery.v6.history import SessionResidualHistory
from quant_edge_lab.discovery.v6.incremental import (
    incremental_confirmatory_ok,
    incremental_d1_eligible,
    incremental_stats,
    matched_day_deltas,
)
from quant_edge_lab.discovery.v6.promotion import (
    LABEL_NOT_INCREMENTAL,
    d3_campaign_decision,
    h1_survives_d2,
    refuse_open_d3,
)
from quant_edge_lab.discovery.v6.residual import (
    equal_weight_loo,
    market_residual,
    retention_ratio,
    shock_z,
    simple_return,
)
from quant_edge_lab.discovery.v6.runner import run_campaign_v6, run_d1_execute
from quant_edge_lab.discovery.v6.selection import select_d1_cell
from quant_edge_lab.features.v4r.beta import beta_from_history

SAMPLE = {
    "min_ticker_days": 750,
    "min_trading_days": 100,
    "min_tickers": 75,
    "max_top_ticker_share": 0.50,
    "min_win_rate": 0.48,
}

CELL = {
    "id": "W5_W5",
    "impulse_window_minutes": 5,
    "wait_window_minutes": 5,
    "min_abs_shock_z": 2.0,
    "high_retention_min": 0.50,
    "low_retention_max": 0.00,
}


def _hist_filled(iids: tuple[str, ...], value: float = 0.001) -> SessionResidualHistory:
    hist = SessionResidualHistory()
    clocks = [f"5:{h:02d}:{m:02d}" for h in range(10, 14) for m in range(60)]
    for i in range(20):
        val = value * ((-1) ** i)
        hist.push({(iid, ck): val for iid in iids for ck in clocks})
    return hist


def _rth_day(
    day: str, names: dict[str, float], *, drop: dict[str, time] | None = None
) -> pl.DataFrame:
    rows = []
    y, m, d = (int(x) for x in day.split("-"))
    skip = drop or {}
    for iid, px0 in names.items():
        for h in range(9, 16):
            for mm in range(60):
                t = time(h, mm)
                if skip.get(iid) == t:
                    continue
                shock = 20.0 if iid == "AAA" and time(10, 0) <= t <= time(10, 4) else 0.0
                px = px0 + shock
                rows.append(
                    {
                        "instrument_id": iid,
                        "ticker": iid,
                        "session_date": day,
                        "time_et": t,
                        "is_rth": True,
                        "open": px,
                        "close": px,
                        "high": px,
                        "low": px,
                        "volume": 1_000_000.0,
                        "ts_utc": datetime(y, m, d, h, mm),
                    }
                )
    return pl.DataFrame(rows)


def test_residual_retention_shock_z_unclipped():
    assert simple_return(110.0, 100.0) == pytest.approx(0.10)
    assert market_residual(0.10, 1.0, 0.02) == pytest.approx(0.08)
    assert retention_ratio(0.12, 0.08) == pytest.approx(1.5)
    assert retention_ratio(-0.04, 0.08) == pytest.approx(-0.5)
    assert retention_ratio(0.1, 0.0) is None
    prior = [0.01 * ((-1) ** i) for i in range(20)]
    z = shock_z(0.08, prior)
    assert z is not None and z > 2.0
    assert shock_z(0.08, prior[:19]) is None
    loo = equal_weight_loo(np.array([0.1, 0.0, 0.0]))
    assert loo[0] == pytest.approx(0.0)
    assert signed_forward(0.01, "NEGATIVE") == pytest.approx(-0.01)


def test_v6_mean_is_equal_day_not_event_weighted():
    ev = pl.DataFrame(
        {
            "trading_date": ["d1"] * 10 + ["d2"],
            "primary_signed": [0.01] * 10 + [-0.01],
            "instrument_id": [f"t{i}" for i in range(11)],
        }
    )
    v6 = equal_weight_day_stats(ev, n_boot=50, seed=1)
    evt = day_block_stats(ev, n_boot=50, seed=1)
    assert v6["mean"] == pytest.approx(0.0)
    assert evt["mean"] != pytest.approx(v6["mean"])
    assert v6["primary_estimand"] == "equal_weight_trading_day_mean"
    out = evaluate_hypotheses(
        ev.with_columns(
            pl.lit(True).alias("is_h1"),
            pl.lit(False).alias("is_h2"),
            pl.lit(True).alias("is_h3"),
            pl.lit(0.0).alias("h2_signed"),
            pl.lit(False).alias("is_incremental_control"),
            pl.lit("POSITIVE").alias("impulse_sign"),
            pl.lit("[2.0,2.5)").alias("abs_z_bucket"),
            pl.lit("11:00").alias("t1_bucket"),
        )
    )
    assert out["h1_stats"]["mean"] == pytest.approx(0.0)


def test_exact_20_session_history_no_reachback():
    hist = SessionResidualHistory()
    key = ("AAA", "5:10:00")
    for i in range(20):
        hist.push({key: 0.01 * ((-1) ** i)})
    assert hist.lookup("AAA", "5:10:00") is not None
    assert hist.shock_z_for("AAA", "5:10:00", 0.08) is not None
    hole = SessionResidualHistory()
    for i in range(20):
        hole.push({} if i == 10 else {key: 0.01 * ((-1) ** i)})
    assert hole.lookup("AAA", "5:10:00") is None
    hole.push({key: 0.02})
    assert hole.lookup("AAA", "5:10:00") is None
    cur = SessionResidualHistory()
    for i in range(19):
        cur.push({key: 0.01})
    z_before = cur.shock_z_for("AAA", "5:10:00", 0.08)
    assert z_before is None
    today = 0.99
    z_with_today_not_pushed = cur.shock_z_for("AAA", "5:10:00", 0.08)
    assert z_with_today_not_pushed is None
    cur.push({key: today})
    after = cur.lookup("AAA", "5:10:00")
    assert after is not None and today not in after[:19]


def test_entry_skips_t1_plus_1():
    t1 = time(11, 10)
    assert entry_time(t1) == time(11, 12)
    ts = datetime(2022, 1, 3, 16, 10)
    assert decision_ts_entry(ts) == ts + timedelta(minutes=2)
    assert t1_bucket(time(11, 10)) == "11:00"


def test_incremental_does_not_cross_sign():
    treated = pl.DataFrame(
        {
            "trading_date": ["d1", "d1"],
            "impulse_sign": ["POSITIVE", "POSITIVE"],
            "abs_z_bucket": ["[2.0,2.5)", "[2.0,2.5)"],
            "t1_bucket": ["11:00", "11:00"],
            "primary_signed": [0.01, 0.02],
        }
    )
    control_wrong = pl.DataFrame(
        {
            "trading_date": ["d1", "d1"],
            "impulse_sign": ["NEGATIVE", "NEGATIVE"],
            "abs_z_bucket": ["[2.0,2.5)", "[2.0,2.5)"],
            "t1_bucket": ["11:00", "11:00"],
            "primary_signed": [0.00, 0.00],
        }
    )
    assert matched_day_deltas(treated, control_wrong) == []
    control_ok = control_wrong.with_columns(pl.lit("POSITIVE").alias("impulse_sign"))
    deltas = matched_day_deltas(treated, control_ok)
    assert len(deltas) == 1
    assert deltas[0] == pytest.approx(0.015)


def test_incremental_inference_and_d1_vs_d2_gates():
    rng = np.random.default_rng(42)
    strong = list(0.002 + 0.0004 * rng.standard_normal(120))
    st = incremental_stats(strong, n_boot=200, seed=42)
    assert st["n_matched_days"] == 120
    assert st["mean_day_delta"] > 0
    assert st["se"] is not None
    assert 0 <= st["p_one_sided"] < 0.05
    assert incremental_d1_eligible(st)
    assert incremental_confirmatory_ok(st)[0] is True
    weak = incremental_stats([0.001] * 20)
    assert incremental_d1_eligible(weak)
    assert incremental_confirmatory_ok(weak)[1] == "incremental_n_matched_days"


def test_d1_selection_requires_incremental_mean():
    cell_a = {"id": "W5_W5"}
    cell_b = {"id": "W5_W15"}
    h1_ok = {
        "mean": 0.002,
        "ticker_days": 800,
        "trading_days": 120,
        "tickers": 80,
        "top_ticker_share": 0.1,
        "win_rate": 0.55,
    }
    h1_better = {**h1_ok, "mean": 0.005}
    inc_pos = {"mean_day_delta": 0.001, "n_matched_days": 10, "p_one_sided": 0.4}
    inc_neg = {"mean_day_delta": -0.001, "n_matched_days": 10, "p_one_sided": 0.9}
    rows = [
        {"cell": cell_b, "h1_stats": h1_better, "incremental": inc_neg},
        {"cell": cell_a, "h1_stats": h1_ok, "incremental": inc_pos},
    ]
    out = select_d1_cell(rows, SAMPLE)
    assert out["decision"] == "FREEZE"
    assert out["selected_id"] == "W5_W5"
    kill = select_d1_cell(
        [{"cell": cell_b, "h1_stats": h1_better, "incremental": inc_neg}], SAMPLE
    )
    assert kill["decision"] == "KILL_AT_D1"


def test_d2_d3_incremental_inference_gates():
    h1 = {
        "mean": 0.002,
        "ticker_days": 800,
        "trading_days": 120,
        "tickers": 80,
        "top_ticker_share": 0.1,
        "win_rate": 0.55,
    }
    inc_ok = {"mean_day_delta": 0.0004, "n_matched_days": 120, "p_one_sided": 0.01}
    inc_p = {"mean_day_delta": 0.0004, "n_matched_days": 120, "p_one_sided": 0.20}
    ok, _ = h1_survives_d2(h1, inc_ok, sample=SAMPLE, bh_survivor=True)
    assert ok is True
    bad, why = h1_survives_d2(h1, inc_p, sample=SAMPLE, bh_survivor=True)
    assert bad is False and why == "incremental_p"
    h1_big = {**h1, "mean": 0.0020}
    d3 = d3_campaign_decision(
        h1_big, inc_p, sample=SAMPLE, bh_survivor=True, d2_mean=0.001
    )
    assert d3["label"] == LABEL_NOT_INCREMENTAL
    pass_row = d3_campaign_decision(
        h1_big, inc_ok, sample=SAMPLE, bh_survivor=True, d2_mean=0.001
    )
    assert pass_row["label"] == "RESEARCH_PASS"
    with pytest.raises(RuntimeError, match="D3 refused"):
        refuse_open_d3(False)


def test_readiness_does_not_execute_or_read_payloads(monkeypatch):
    called = []

    def boom(*_a, **_k):
        called.append("d1")
        raise AssertionError("run_d1_execute must not run during readiness")

    monkeypatch.setattr("quant_edge_lab.discovery.v6.runner.run_d1_execute", boom)
    out = run_campaign_v6(Path("."), execute=False)
    assert out["mode"] == "READINESS"
    assert called == []
    assert out["calendar_n_days"] == EXPECTED_CALENDAR_N_DAYS
    assert out["calendar_hash_expected"] == EXPECTED_CALENDAR_HASH
    assert out["calendar_identity_match"] is True
    assert out["primary_estimand"] == "equal_weight_trading_day_mean"
    assert out["execution_status"] == "NOT_APPROVED"
    with pytest.raises(RuntimeError, match="execute_v6 refused"):
        run_campaign_v6(Path("."), execute=True)
    assert called == []


def test_d1_loadable_days_stop_before_d2():
    des = load_v6(Path("."))
    man = calendar_manifest(des)
    days = d1_loadable_days(
        [
            "2021-10-01",
            "2021-10-29",
            man["splits"]["D1"]["end"],
            man["splits"]["D2"]["start"],
            man["splits"]["D3"]["start"],
        ],
        man,
    )
    assert man["splits"]["D2"]["start"] not in days
    assert man["splits"]["D3"]["start"] not in days
    assert days[-1] <= man["splits"]["D1"]["end"]
    assert run_d1_execute.__doc__ is not None and "D1-only" in run_d1_execute.__doc__


def test_frozen_cells_match_yaml():
    des = load_v6(Path("."))
    cells = frozen_cells(des)
    assert len(cells) == 4
    assert all(c["min_abs_shock_z"] == 2.0 for c in cells)


def test_process_day_entry_is_t1_plus_2():
    bars = _rth_day("2022-01-03", {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0})
    hist = _hist_filled(("AAA", "BBB", "CCC"))
    ev, _ = process_day(
        bars, CELL, betas={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}, history=hist
    )
    if ev.height:
        t1 = ev["t1"][0]
        entry = ev["entry"][0]
        th, tm = map(int, t1.split(":"))
        eh, em = map(int, entry.split(":"))
        assert eh * 60 + em == th * 60 + tm + 2


def test_first_event_reserved_before_future_outcome():
    bars = _rth_day(
        "2022-01-03",
        {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0},
        drop={"AAA": time(10, 25)},
    )
    hist = _hist_filled(("AAA", "BBB", "CCC"))
    ev, _ = process_day(
        bars, CELL, betas={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}, history=hist
    )
    aaa = ev.filter(pl.col("instrument_id") == "AAA")
    assert aaa.height == 1
    assert aaa["impulse_start"][0] == "10:00"
    val = aaa["primary_signed"][0]
    assert val is None or (isinstance(val, float) and val != val)


def test_ineligible_excluded_from_loo_and_missing_beta_blocks_signal():
    bars = _rth_day(
        "2022-01-03",
        {"AAA": 100.0, "BBB": 100.0, "CCC": 200.0},
    )
    hist = _hist_filled(("AAA", "BBB", "CCC"))
    ev_all, _ = process_day(
        bars,
        CELL,
        betas={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0},
        history=hist,
        eligible={"AAA", "BBB", "CCC"},
    )
    hist2 = _hist_filled(("AAA", "BBB", "CCC"))
    ev_ex, day_map = process_day(
        bars,
        CELL,
        betas={"AAA": 1.0, "BBB": 1.0},
        history=hist2,
        eligible={"AAA", "BBB"},
    )
    assert all(r[0] != "CCC" for r in day_map)
    if ev_ex.height:
        assert "CCC" not in ev_ex["instrument_id"].to_list()
    hist3 = _hist_filled(("AAA", "BBB", "CCC"))
    ev_nb, dm_nb = process_day(
        bars, CELL, betas={}, history=hist3, eligible={"AAA", "BBB", "CCC"}
    )
    assert ev_nb.height == 0
    assert dm_nb == {}
    _ = ev_all


def test_current_day_not_in_beta():
    def panel(stock: float, mkt: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "instrument_id": ["AAA"],
                "daily_ret": [stock],
                "daily_mkt_loo": [mkt],
            }
        )

    prior = [panel(0.02 * (i + 1), 0.01 * (i + 1)) for i in range(20)]
    today = panel(100.0, 0.01 * 21)
    b_prior = beta_from_history(prior, ["AAA"])
    b_leak = beta_from_history(prior + [today], ["AAA"])
    assert "AAA" in b_prior
    assert b_prior["AAA"] != b_leak["AAA"]
    assert beta_from_history(prior[:19], ["AAA"]) == {}


def test_newly_eligible_gets_beta_from_listed_not_event_universe():
    def panel(i: int, names: list[str], *, new_ret: float | None = None) -> pl.DataFrame:
        rets = []
        for n in names:
            if n == "NEW" and new_ret is not None:
                rets.append(new_ret)
            elif n == "NEW":
                rets.append(0.02 * (i + 1))
            else:
                rets.append(0.01 * (i + 1))
        return pl.DataFrame(
            {
                "instrument_id": names,
                "daily_ret": rets,
                "daily_mkt_loo": [0.008 * (i + 1)] * len(names),
            }
        )

    listed = [panel(i, ["OLD", "NEW"]) for i in range(20)]
    event_only = [p.filter(pl.col("instrument_id") == "OLD") for p in listed]
    b_ok = beta_from_history(listed, ["NEW"])
    b_wrong = beta_from_history(event_only, ["NEW"])
    assert "NEW" in b_ok
    assert "NEW" not in b_wrong
    leaked = beta_from_history(listed + [panel(20, ["OLD", "NEW"], new_ret=50.0)], ["NEW"])
    assert b_ok["NEW"] != leaked["NEW"]


def test_separate_impulse_and_remaining_loo_universes():
    p0, p1, p2 = time(9, 59), time(10, 4), time(10, 9)
    closes = {
        "AAA": {p0: 100.0, p1: 120.0, p2: 120.0},
        "BBB": {p0: 100.0, p1: 100.0, p2: 100.0},
        "NO_P2": {p0: 100.0, p1: 150.0},
        "NO_P1": {p0: 100.0, p2: 70.0},
    }
    betas = {k: 1.0 for k in closes}
    imp_ids = window_peer_ids(closes, p0, p1)
    rem_ids = window_peer_ids(closes, p0, p2)
    assert "NO_P2" in imp_ids and "NO_P1" not in imp_ids
    assert "NO_P1" in rem_ids and "NO_P2" not in rem_ids
    imp = window_residuals(closes, imp_ids, p0, p1, betas)
    rem = window_residuals(closes, rem_ids, p0, p2, betas)
    imp_no = window_residuals(closes, ["AAA", "BBB"], p0, p1, betas)
    rem_no = window_residuals(closes, ["AAA", "BBB"], p0, p2, betas)
    assert "NO_P2" in imp and "NO_P1" not in imp
    assert "NO_P1" in rem and "NO_P2" not in rem
    assert imp["AAA"] != imp_no["AAA"]
    assert rem["AAA"] != rem_no["AAA"]


def test_impulse_shock_independent_of_wait_window():
    cell_w5 = dict(CELL)
    cell_w15 = {
        **CELL,
        "id": "W5_W15",
        "wait_window_minutes": 15,
    }
    names = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "NO_P2_LATE": 100.0}
    bars = _rth_day("2022-01-03", names, drop={"NO_P2_LATE": time(10, 19)})
    hist = _hist_filled(tuple(names))
    betas = {k: 1.0 for k in names}
    ev5, imap = process_day(bars, cell_w5, betas=betas, history=hist)
    assert ("NO_P2_LATE", "5:10:00") in imap
    ev15, _ = process_day(
        bars, cell_w15, betas=betas, history=hist, impulse_map=imap
    )
    a5 = ev5.filter(pl.col("instrument_id") == "AAA")
    a15 = ev15.filter(pl.col("instrument_id") == "AAA")
    assert a5.height == 1 and a15.height == 1
    assert a5["impulse_start"][0] == a15["impulse_start"][0] == "10:00"
    assert a5["impulse_residual"][0] == a15["impulse_residual"][0]
    assert a5["shock_z"][0] == a15["shock_z"][0]
    assert "forward_5m" in ev5.columns and "forward_30m" in ev5.columns


def test_target_missing_p0_p1_or_p2_is_not_an_event():
    hist = _hist_filled(("AAA", "BBB", "CCC"))
    betas = {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}
    for drop_t in (time(9, 59), time(10, 4), time(10, 9)):
        bars = _rth_day(
            "2022-01-03",
            {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0},
            drop={"AAA": drop_t},
        )
        ev, _ = process_day(bars, CELL, betas=betas, history=hist)
        aaa = ev.filter(pl.col("instrument_id") == "AAA")
        if aaa.height:
            assert aaa["impulse_start"][0] != "10:00"
