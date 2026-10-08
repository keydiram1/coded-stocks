from __future__ import annotations

import polars as pl
import pytest

from quant_edge_lab.discovery.v5.evaluation import day_block_stats, decide, evaluate_frozen, signed_floor
from quant_edge_lab.discovery.v5.manifest import load_v5
from quant_edge_lab.discovery.v5.models import CandidateRule
from pathlib import Path


def _gates(**sample):
    return {
        "sample": {
            "min_ticker_days": 1,
            "min_trading_days": 1,
            "min_tickers": 1,
            "max_top_ticker_share": 1.0,
            "min_win_rate": 0.0,
            **sample,
        },
        "d2": {"signed_mean_floor": 0.0003},
        "d3": {"signed_mean_floor": 0.0012, "require_sign_match_d2": True},
        "inference": {"bootstrap_draws": 20, "bootstrap_seed": 1, "bh_q": 0.10},
    }


def _rule():
    return CandidateRule(
        hypothesis_id="H1_PRIMARY",
        mechanism_id="forced_eod_auction_flow",
        role="primary",
        dislocation_abs_min=0.0,
        require_rvol=False,
        primary_outcome="next_open_to_15m",
        frozen=True,
    )


def _events(mean: float, n: int = 40) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": [f"i{i%5}" for i in range(n)],
            "trading_date": [f"2024-01-{(i % 20) + 1:02d}" for i in range(n)],
            "discrepancy": [0.05] * n,
            "close_volume_rvol": [2.0] * n,
            "primary_signed": [mean] * n,
        }
    )


def test_yaml_floors_are_signed_not_abs():
    _man, gates = load_v5(Path("."))
    assert signed_floor(gates, "D2") == pytest.approx(0.0003)
    assert signed_floor(gates, "D3") == pytest.approx(0.0012)


def test_negative_twenty_bp_never_passes():
    st = {"ticker_days": 10, "trading_days": 10, "tickers": 5, "top_ticker_share": 0.2, "mean": -0.002, "win_rate": 0.2}
    d2 = decide(split="D2", hypothesis_id="H1", stats=st, gates=_gates(), bh_survivor=True)
    d3 = decide(split="D3", hypothesis_id="H1", stats=st, gates=_gates(), d2_mean=-0.002)
    assert d2.label == "KILL" and d2.reason == "direction_failed"
    assert d3.label == "KILL"
    assert d3.label != "RESEARCH_PASS"


def test_positive_below_d2_floor_kills():
    st = {"ticker_days": 10, "trading_days": 10, "tickers": 5, "top_ticker_share": 0.2, "mean": 0.0001, "win_rate": 0.6}
    d2 = decide(split="D2", hypothesis_id="H1", stats=st, gates=_gates(), bh_survivor=True)
    assert d2.label == "KILL" and d2.reason == "below_d2_floor"


def test_d3_subthreshold_only_if_positive_below_floor():
    st = {"ticker_days": 10, "trading_days": 10, "tickers": 5, "top_ticker_share": 0.2, "mean": 0.0008, "win_rate": 0.6}
    d3 = decide(split="D3", hypothesis_id="H1", stats=st, gates=_gates(), d2_mean=0.0008)
    assert d3.label == "VALIDATED_SUBTHRESHOLD_PHENOMENON"


def test_sample_counts_use_finite_primary_only():
    ev = pl.DataFrame(
        {
            "instrument_id": ["a", "a", "b"],
            "trading_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "primary_signed": [0.01, None, float("nan")],
        }
    )
    st = day_block_stats(ev, n_boot=10, seed=1)
    assert st["events"] == 1
    assert st["ticker_days"] == 1
    assert st["trading_days"] == 1
    assert st["tickers"] == 1


def test_d3_does_not_inspect_dead_d2():
    ev = _events(-0.002)
    with pytest.raises(AssertionError, match="dead D2"):
        evaluate_frozen(ev, [_rule()], split="D3", gates=_gates(), d2_survivors=None)
    out = evaluate_frozen(ev, [_rule()], split="D3", gates=_gates(), d2_survivors=set())
    assert out == []


def test_d2_runs_bh_across_hypotheses():
    ev = _events(0.002)
    rules = [
        _rule(),
        CandidateRule(
            hypothesis_id="H2_ABS_Q95",
            mechanism_id="forced_eod_auction_flow",
            role="robustness",
            dislocation_abs_min=0.0,
            require_rvol=False,
            primary_outcome="next_open_to_15m",
            frozen=True,
        ),
        CandidateRule(
            hypothesis_id="H3_DISLOCATION_ONLY",
            mechanism_id="forced_eod_auction_flow",
            role="robustness",
            dislocation_abs_min=0.0,
            require_rvol=False,
            primary_outcome="next_open_to_15m",
            frozen=True,
        ),
    ]
    rows = evaluate_frozen(ev, rules, split="D2", gates=_gates())
    assert len(rows) == 3
    assert all("p_one_sided" in r["stats"] for r in rows)
    assert all("bh_rejected" in r["stats"] for r in rows)
