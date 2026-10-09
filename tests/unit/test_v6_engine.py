from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_edge_lab.discovery.v6.clocks import decision_ts_entry, entry_time, t1_bucket
from quant_edge_lab.discovery.v6.design import frozen_cells, load_v6
from quant_edge_lab.discovery.v6.events import process_day, signed_forward
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
from quant_edge_lab.discovery.v6.runner import run_campaign_v6
from quant_edge_lab.discovery.v6.selection import select_d1_cell

SAMPLE = {
    "min_ticker_days": 750,
    "min_trading_days": 100,
    "min_tickers": 75,
    "max_top_ticker_share": 0.50,
    "min_win_rate": 0.48,
}


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


def test_entry_skips_t1_plus_1():
    t1 = time(11, 10)
    assert entry_time(t1) == time(11, 12)
    ts = datetime(2022, 1, 3, 16, 10)  # T1 bar start UTC placeholder
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


def test_signed_forward_and_readiness_does_not_execute():
    assert signed_forward(0.01, "POSITIVE") == pytest.approx(0.01)
    assert signed_forward(0.01, "NEGATIVE") == pytest.approx(-0.01)
    out = run_campaign_v6(Path("."), execute=False)
    assert out["mode"] == "READINESS"
    assert out["d1_parameter_cells"] == 4
    assert out["confirmatory_signal_hypotheses"] == 3
    assert out["incremental_mechanism_gate"] == 1
    assert [c["id"] for c in out["frozen_cells"]] == ["W5_W5", "W5_W15", "W15_W5", "W15_W15"]
    assert out["primary_entry"] == "T1_plus_2_open"
    assert out["execution_status"] == "NOT_APPROVED"
    assert "No D1 returns" in out["note"]
    with pytest.raises(RuntimeError, match="execute_v6 refused"):
        run_campaign_v6(Path("."), execute=True)


def test_frozen_cells_match_yaml():
    des = load_v6(Path("."))
    cells = frozen_cells(des)
    assert len(cells) == 4
    assert all(c["min_abs_shock_z"] == 2.0 for c in cells)


def test_process_day_entry_is_t1_plus_2():
    rows = []
    day = "2022-01-03"
    for iid, px0 in (("AAA", 100.0), ("BBB", 100.0), ("CCC", 100.0)):
        for h in range(9, 16):
            for m in range(60):
                if h == 15 and m > 59:
                    continue
                t = time(h, m)
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
                        "ts_utc": datetime(2022, 1, 3, h, m),
                    }
                )
    bars = pl.DataFrame(rows)
    cell = {
        "id": "W5_W5",
        "impulse_window_minutes": 5,
        "wait_window_minutes": 5,
        "min_abs_shock_z": 2.0,
        "high_retention_min": 0.50,
        "low_retention_max": 0.00,
    }
    prior = [0.001 * ((-1) ** i) for i in range(20)]
    hist = {
        "AAA": {f"5:{h:02d}:{m:02d}": prior for h in range(10, 14) for m in range(60)},
        "BBB": {f"5:{h:02d}:{m:02d}": prior for h in range(10, 14) for m in range(60)},
        "CCC": {f"5:{h:02d}:{m:02d}": prior for h in range(10, 14) for m in range(60)},
    }
    ev, _ = process_day(bars, cell, betas={"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}, prior_clock=hist)
    if ev.height:
        t1 = ev["t1"][0]
        entry = ev["entry"][0]
        th, tm = map(int, t1.split(":"))
        eh, em = map(int, entry.split(":"))
        assert eh * 60 + em == th * 60 + tm + 2
