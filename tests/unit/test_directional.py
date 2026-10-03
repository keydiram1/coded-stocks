from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from quant_edge_lab.discovery.dir_gates import decide_directional, freeze_hashes, load_gates
from quant_edge_lab.discovery.eval_directional import eval_d004, prepare_rth
from quant_edge_lab.discovery.stages import select_falsify_days, select_stage_days
from quant_edge_lab.universe.filters import add_session_columns


def test_tiny_signed_mean_cannot_pass():
    gates = load_gates(Path("."))
    g = gates.stages["fast"]
    d, reason = decide_directional(
        mean_primary=-0.0000025,
        median_primary=0.0,
        win_rate=0.47,
        ticker_days=40000,
        trading_days=199,
        top_ticker_share=0.05,
        frac_days_agree=0.5,
        ci=[-0.00001, 0.00001],
        gate=g,
        stage_complete=True,
    )
    assert d == "KILL"
    assert "sign disagrees" in reason or "economic floor" in reason


def test_positive_but_too_small_killed():
    g = load_gates(Path(".")).stages["fast"]
    d, reason = decide_directional(
        mean_primary=0.0002,
        median_primary=0.0001,
        win_rate=0.51,
        ticker_days=100,
        trading_days=20,
        top_ticker_share=0.1,
        frac_days_agree=0.6,
        ci=[0.0001, 0.0003],
        gate=g,
        stage_complete=True,
    )
    assert d == "KILL" and "economic floor" in reason


def test_fast_gate_allows_modest_positive():
    g = load_gates(Path(".")).stages["fast"]
    d, _ = decide_directional(
        mean_primary=0.002,
        median_primary=0.001,
        win_rate=0.52,
        ticker_days=40,
        trading_days=10,
        top_ticker_share=0.2,
        frac_days_agree=0.55,
        ci=[-0.001, 0.005],
        gate=g,
        stage_complete=True,
    )
    assert d == "PASS"


def test_full_requires_ci_exclude_zero():
    g = load_gates(Path(".")).stages["full"]
    d, reason = decide_directional(
        mean_primary=0.003,
        median_primary=0.001,
        win_rate=0.52,
        ticker_days=250,
        trading_days=90,
        top_ticker_share=0.2,
        frac_days_agree=0.55,
        ci=[-0.001, 0.006],
        gate=g,
        stage_complete=True,
    )
    assert d == "KILL" and "CI" in reason


def test_freeze_hashes_stable():
    a = freeze_hashes(Path("."))
    b = freeze_hashes(Path("."))
    assert a == b
    assert len(a["gates"]) == 64


def test_falsify_disjoint_from_fast_broad():
    days = [f"2022-01-{d:02d}" for d in range(1, 32)] + [f"2022-06-{d:02d}" for d in range(1, 31)]
    days += [f"2023-03-{d:02d}" for d in range(1, 32)]
    fast = set(select_stage_days(days, "fast", n_fast=20))
    broad = set(select_stage_days(days, "broad", n_broad=40))
    fal = set(select_falsify_days(days, n=15))
    assert fal.isdisjoint(fast)
    assert fal.isdisjoint(broad)


def test_d004_mom5_assigns_side_from_ret5():
    rows = []
    t0 = datetime(2022, 1, 4, 14, 30)  # 09:30 ET ≈ 14:30 UTC Jan
    for i in range(40):
        ts = t0 + timedelta(minutes=i)
        close = 10.0 + i * 0.001
        rows.append(
            {
                "instrument_id": "ticker:TEST",
                "ticker": "TEST",
                "ts_utc": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1_000_000.0,
                "exchange": "NASDAQ",
                "security_type": "COMMON_STOCK",
            }
        )
    bars = pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
    sessioned = add_session_columns(bars)
    inst = pl.DataFrame(
        {"instrument_id": ["ticker:TEST"], "exchange": ["NASDAQ"], "security_type": ["COMMON_STOCK"]}
    )
    prev = {"ticker:TEST": 10.0}
    typ = {"ticker:TEST": 1000.0}
    rth = prepare_rth(sessioned, inst, prev, typ, {}, {}, { "ticker:TEST": 0.02})
    ev = eval_d004(rth, inst, prev, typ, "exp", "D004.mom5")
    assert ev.height >= 1
    assert set(ev["side"].unique()) <= {"long", "short"}
