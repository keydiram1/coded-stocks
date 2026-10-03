from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from quant_edge_lab.discovery.campaign_v2_families import eval_a
from quant_edge_lab.discovery.campaign_v2_runner import freeze_campaign
from quant_edge_lab.discovery.dir_gates import decide_directional, load_gates_from
from quant_edge_lab.features.causal_store import assert_inputs_causal, compute_day_features
from quant_edge_lab.universe.filters import add_session_columns


def _bars(n: int = 80, start: datetime | None = None, close0: float = 10.0) -> pl.DataFrame:
    t0 = start or datetime(2022, 6, 8, 13, 30)
    rows = []
    for i in range(n):
        ts = t0 + timedelta(minutes=i)
        close = close0 * (1.0 + 0.0002 * i)
        rows.append(
            {
                "instrument_id": "ticker:TEST",
                "ticker": "TEST",
                "ts_utc": ts,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 50_000.0,
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))


def _feat():
    bars = _bars()
    sessioned = add_session_columns(bars)
    inst = pl.DataFrame(
        {"instrument_id": ["ticker:TEST"], "exchange": ["NASDAQ"], "security_type": ["COMMON_STOCK"]}
    )
    hist = {
        "prev_close": {"ticker:TEST": 10.0},
        "typ": {"ticker:TEST": 1000.0},
        "yday_ret": {"ticker:TEST": 0.02},
        "yday_high": {"ticker:TEST": 10.2},
        "yday_low": {"ticker:TEST": 9.8},
        "yday_range": {"ticker:TEST": 0.04},
        "yday_clv": {"ticker:TEST": 0.5},
        "ret_2d": {"ticker:TEST": 0.03},
        "ret_3d": {"ticker:TEST": 0.05},
        "ret_5d": {"ticker:TEST": 0.08},
        "ah_ret": {"ticker:TEST": 0.01},
        "med20": {"ticker:TEST": 0.02},
        "pm_ret": {"ticker:TEST": 0.0},
    }
    return compute_day_features(sessioned, inst, hist), sessioned


def test_available_at_not_after_decision():
    feat, _ = _feat()
    assert feat.height > 0
    assert_inputs_causal(feat, ["ret_5m", "rvol_5m"])
    assert (feat["available_at"] <= feat["decision_ts"]).all()


def test_future_poison_does_not_change_past_features():
    feat, sessioned = _feat()
    cutoff = feat["ts_utc"][20]
    early = sessioned.filter(pl.col("ts_utc") <= cutoff)
    inst = pl.DataFrame(
        {"instrument_id": ["ticker:TEST"], "exchange": ["NASDAQ"], "security_type": ["COMMON_STOCK"]}
    )
    hist = {
        "prev_close": {"ticker:TEST": 10.0},
        "typ": {"ticker:TEST": 1000.0},
        "yday_high": {"ticker:TEST": 10.2},
        "yday_low": {"ticker:TEST": 9.8},
        "med20": {"ticker:TEST": 0.02},
    }
    feat_early = compute_day_features(early, inst, hist)
    key = feat.filter(pl.col("ts_utc") == cutoff).select(["ret_5m", "rvol_5m", "gap_pct"])
    key_e = feat_early.filter(pl.col("ts_utc") == cutoff).select(["ret_5m", "rvol_5m", "gap_pct"])
    assert key.height == 1 and key_e.height == 1
    assert abs(float(key["ret_5m"][0] or 0) - float(key_e["ret_5m"][0] or 0)) < 1e-12


def test_one_bar_shift_ret1m_uses_prior_close():
    feat, _ = _feat()
    i = 25
    r = float(feat["ret_1m"][i])
    c0 = float(feat["close"][i - 1])
    c1 = float(feat["close"][i])
    assert abs(r - (c1 / c0 - 1)) < 1e-9


def test_v2_gates_kill_tiny_and_concentration():
    g = load_gates_from(Path("."), Path("knowledge/campaigns/directional_v2_gates.yaml")).stages["fast"]
    d, _reason = decide_directional(
        mean_primary=0.0000025,
        median_primary=0.0,
        win_rate=0.51,
        ticker_days=100,
        trading_days=20,
        top_ticker_share=0.05,
        frac_days_agree=0.6,
        ci=[0.0, 0.001],
        gate=g,
        stage_complete=True,
    )
    assert d == "KILL"
    d2, r2 = decide_directional(
        mean_primary=0.002,
        median_primary=0.001,
        win_rate=0.52,
        ticker_days=40,
        trading_days=10,
        top_ticker_share=0.35,
        frac_days_agree=0.6,
        ci=[0.001, 0.003],
        gate=g,
        stage_complete=True,
    )
    assert d2 == "KILL" and "concentration" in r2


def test_freeze_campaign_stable():
    a = freeze_campaign(Path("."))
    b = freeze_campaign(Path("."))
    assert a["manifest"] == b["manifest"]
    assert len(a["manifest"]) == 64


def test_impulse_only_fires_on_large_move():
    t0 = datetime(2022, 6, 8, 13, 30)
    rows = []
    px = 10.0
    for i in range(50):
        if 15 <= i < 25:
            px *= 1.006
        ts = t0 + timedelta(minutes=i)
        rows.append(
            {
                "instrument_id": "ticker:TEST",
                "ticker": "TEST",
                "ts_utc": ts,
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": 1_000_000.0,
            }
        )
    bars = pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
    sessioned = add_session_columns(bars)
    inst = pl.DataFrame(
        {"instrument_id": ["ticker:TEST"], "exchange": ["NASDAQ"], "security_type": ["COMMON_STOCK"]}
    )
    hist = {
        "prev_close": {"ticker:TEST": 10.0},
        "typ": {"ticker:TEST": 100.0},
        "yday_high": {"ticker:TEST": 11.0},
        "yday_low": {"ticker:TEST": 9.0},
        "med20": {"ticker:TEST": 0.03},
    }
    feat = compute_day_features(sessioned, inst, hist)
    ev = eval_a(feat, "exp", impulse_only=True)
    assert ev.height >= 1
    assert "side" in ev.columns
