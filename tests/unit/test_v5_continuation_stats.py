from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from quant_edge_lab.discovery.v5.continuation import GATES_REL
from quant_edge_lab.discovery.v5.continuation_stats import equal_weight_day_stats
from quant_edge_lab.discovery.v5.evaluation import decide, onesided_greater_p


def _unbalanced() -> pl.DataFrame:
    day1 = pl.DataFrame(
        {
            "instrument_id": [f"a{i}" for i in range(100)],
            "trading_date": ["2024-11-01"] * 100,
            "primary_signed": [0.0] * 100,
        }
    )
    day2 = pl.DataFrame(
        {
            "instrument_id": ["z"],
            "trading_date": ["2024-11-02"],
            "primary_signed": [0.0024],
        }
    )
    return pl.concat([day1, day2])


def _loose_gates() -> dict:
    return yaml.safe_load((Path(".") / GATES_REL).read_text(encoding="utf-8")) | {
        "sample": {
            "min_ticker_days": 1,
            "min_trading_days": 1,
            "min_tickers": 1,
            "max_top_ticker_share": 1.0,
            "min_win_rate": 0.0,
        }
    }


def test_primary_mean_equals_mean_of_daily_means_not_pooled_events():
    ev = _unbalanced()
    st = equal_weight_day_stats(ev, n_boot=400, seed=42)
    day_means = [0.0, 0.0024]
    equal = float(np.mean(day_means))
    pooled = float(np.mean([0.0] * 100 + [0.0024]))
    assert st["mean"] == pytest.approx(equal)
    assert st["mean"] == pytest.approx(0.0012)
    assert st["mean_bp"] == pytest.approx(12.0)
    assert st["mean"] != pytest.approx(pooled)
    assert pooled < 0.0003
    assert sorted(st["daily_means"]) == pytest.approx(sorted(day_means))
    dm = np.asarray(st["daily_means"], dtype=float)
    expect_se = float(np.std(dm, ddof=1) / np.sqrt(dm.size))
    assert st["se"] == pytest.approx(expect_se)
    assert st["p_one_sided"] == pytest.approx(onesided_greater_p(dm))
    lo, hi = st["ci"]
    assert lo <= equal <= hi
    assert st["event_win_rate"] == pytest.approx(1.0 / 101.0)
    assert st["primary_estimand"] == "equal_weight_trading_day_mean"


def test_twelve_bp_floor_uses_equal_day_mean():
    st = equal_weight_day_stats(_unbalanced(), n_boot=50, seed=1)
    d3 = decide(
        split="D3",
        hypothesis_id="H1_PRIMARY",
        stats={
            "ticker_days": 10,
            "trading_days": 10,
            "tickers": 5,
            "top_ticker_share": 0.2,
            "mean": st["mean"],
            "win_rate": 0.6,
        },
        gates=_loose_gates(),
        bh_survivor=True,
    )
    assert d3.label == "RESEARCH_PASS"
    pooled = 0.0024 / 101.0
    wrong = decide(
        split="D3",
        hypothesis_id="H1_PRIMARY",
        stats={
            "ticker_days": 10,
            "trading_days": 10,
            "tickers": 5,
            "top_ticker_share": 0.2,
            "mean": pooled,
            "win_rate": 0.6,
        },
        gates=_loose_gates(),
        bh_survivor=True,
    )
    assert wrong.label == "VALIDATED_SUBTHRESHOLD_PHENOMENON"
