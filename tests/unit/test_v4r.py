from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_edge_lab.discovery.v4r.association import _daily_spearman
from quant_edge_lab.discovery.v4r.overlay import join_overlay
from quant_edge_lab.discovery.v4r.search import ridge_train_scaler
from quant_edge_lab.discovery.v4r.stability import direction_impossible, max_possible, stability_impossible
from quant_edge_lab.features.v4r.beta import BETA_WINDOW, attach_beta, ols_beta
from quant_edge_lab.features.v4r.target import attach_forward_holding_residuals
from quant_edge_lab.hashing import sha256_file
from quant_edge_lab.peers import GRAPH_FEATURE_COLS, PeerGraph, attach_peer_features, topk_finite


def test_v4_yaml_untouched():
    root = Path(".")
    assert sha256_file(root / "knowledge/campaigns/directional_v4_manifest.yaml") == "ee59be114e76f3ecd78bd3ad2590e60c94f0754a42096818d9fbf7ee8a7b4a3e"
    assert sha256_file(root / "knowledge/campaigns/directional_v4_gates.yaml") == "02ada62546ec70681c7d884f4cb820923e6d1417e21a6d31586fc8c955b8ac38"


def test_ols_beta_null_if_short():
    assert not np.isfinite(ols_beta(np.ones(10), np.ones(10)))
    x = np.arange(20, dtype=float)
    y = 2 * x + 1
    assert abs(ols_beta(y, x) - 2.0) < 1e-9


def test_attach_beta_never_fills_one():
    g = pl.DataFrame({"instrument_id": ["A", "B"], "ret_5m": [0.01, 0.02]})
    out = attach_beta(g, {"A": 0.5})
    assert out["beta_20d"].to_list()[0] == 0.5
    assert out["beta_20d"].to_list()[1] is None


def _grid(prices: list[float], day: str, ticker: str = "A") -> pl.DataFrame:
    ts0 = datetime.fromisoformat(f"{day}T14:40:00+00:00")
    rows = []
    prev = prices[0]
    for i, px in enumerate(prices):
        ret = px / prev - 1 if i else 0.0
        prev = px
        rows.append(
            {
                "instrument_id": ticker,
                "trading_date": day,
                "decision_ts": ts0 + timedelta(minutes=5 * i),
                "ret_5m": ret,
                "market_ret_5m": 0.0,
                "beta_20d": 1.0,
            }
        )
    return pl.DataFrame(rows)


def test_target_compound_three_legs():
    # prices 100,101,102,103 → rets 0, 0.01, 101/102? wait
    # bar i return is close_i/close_{i-1}-1. Forward 15m at i=0 uses rets of bars 1,2,3
    g = _grid([100, 101, 102, 103], "2023-05-17")
    # ret[1]=1%, ret[2]=102/101-1, ret[3]=103/102-1
    out = attach_forward_holding_residuals(g)
    r = out["ret_5m"].to_list()
    expect = (1 + r[1]) * (1 + r[2]) * (1 + r[3]) - 1
    got = out["future_raw_15m"][0]
    assert abs(got - expect) < 1e-12
    assert abs(out["future_residual_15m"][0] - expect) < 1e-12  # mkt 0, beta 1


def test_target_fourth_bar_irrelevant():
    a = attach_forward_holding_residuals(_grid([100, 101, 102, 103, 200], "2023-05-17"))
    b = attach_forward_holding_residuals(_grid([100, 101, 102, 103, 50], "2023-05-17"))
    assert a["future_raw_15m"][0] == b["future_raw_15m"][0]


def test_target_no_next_day_wrap():
    g = pl.concat([_grid([100, 101, 102, 103], "2023-05-17"), _grid([100, 110, 120, 130], "2023-05-18")])
    out = attach_forward_holding_residuals(g)
    last = out.filter(pl.col("trading_date") == "2023-05-17")[-1]
    assert last["future_raw_15m"][0] is None


def test_target_ticker_isolation():
    a = _grid([100, 101, 102, 103], "2023-05-17", "A")
    b = _grid([100, 200, 300, 400], "2023-05-17", "B")
    out = attach_forward_holding_residuals(pl.concat([a, b]))
    fa = attach_forward_holding_residuals(a)["future_raw_15m"][0]
    assert abs(out.filter(pl.col("instrument_id") == "A")["future_raw_15m"][0] - fa) < 1e-12


def test_beta_not_affected_by_future_market_in_label():
    # beta is an input column at t; changing future market legs changes residual label not beta
    g = _grid([100, 101, 102, 103], "2023-05-17")
    g = g.with_columns(pl.Series("market_ret_5m", [0.0, 0.01, 0.01, 0.01]))
    out = attach_forward_holding_residuals(g)
    assert out["beta_20d"][0] == 1.0
    g2 = g.with_columns(pl.Series("market_ret_5m", [0.0, 0.5, 0.5, 0.5]))
    out2 = attach_forward_holding_residuals(g2)
    assert out2["beta_20d"][0] == out["beta_20d"][0]
    assert out2["future_residual_15m"][0] != out["future_residual_15m"][0]


def test_topk_finite_excludes_nan():
    s = np.array([np.nan, 0.1, np.nan, 0.9, 0.2])
    idx = topk_finite(s, 2)
    assert set(idx.tolist()) == {3, 4} or list(idx)[0] == 3


def test_empty_graph_null_not_neg_resid():
    g = pl.DataFrame(
        {
            "decision_ts": [datetime(2023, 5, 17, 14, 40, tzinfo=timezone.utc)],
            "instrument_id": ["A"],
            "resid_ret_5m": [0.02],
            "resid_rank_5m": [0.5],
        }
    )
    out = attach_peer_features(g, PeerGraph("2023-05-10", "", "", (), {}))
    assert out["leader_response_gap"][0] is None
    assert out["graph_available"][0] is False


def test_join_fanout_fails():
    keys = {
        "trading_date": ["d", "d"],
        "instrument_id": ["A", "B"],
        "decision_ts": [datetime(2023, 1, 1, tzinfo=timezone.utc)] * 2,
    }
    s1 = pl.DataFrame({**keys, "x": [1, 2]})
    ov = pl.DataFrame({**keys, **{c: [0.0, 0.0] for c in GRAPH_FEATURE_COLS}})
    ov = pl.concat([ov, ov.head(1)])
    with pytest.raises(Exception, match="duplicate"):
        join_overlay(s1, ov)


def test_day_spearman_not_row_n():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(12):
        for i in range(500):
            rows.append({"trading_date": f"2022-01-{d+1:02d}", "feat": float(rng.normal()), "future_residual_15m": float(rng.normal())})
    df = pl.DataFrame(rows)
    p, meta = _daily_spearman(df, "feat", "future_residual_15m")
    assert meta["n_days"] == 12
    assert meta["n_rows"] == 12 * 500
    assert p > 1e-20  # noise; must not be 1e-300 from 6000 iid rows


def test_stability_early_stop_arithmetic():
    assert max_possible(10, 50, 100) == 60
    assert stability_impossible(10, 50, 100, 0.60) is False  # 10+50=60
    assert stability_impossible(9, 50, 100, 0.60) is True  # 9+50=59 < 60
    assert direction_impossible(agrees=5, path_hits=10, remaining=0, need_frac=0.80) is True
    assert direction_impossible(agrees=5, path_hits=10, remaining=90, need_frac=0.80) is False


def test_ridge_scaler_train_only():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = x[:, 0] * 0.1 + rng.normal(size=200) * 0.01
    art = ridge_train_scaler(x, y, [0.01, 0.1])
    assert "mu" in art and "sd" in art
    cut = int(200 * 0.7)
    assert np.allclose(art["mu"], x[:cut].mean(axis=0), atol=1e-9)


def test_future_poison_decision_features_unchanged():
    g = _grid([100, 101, 102, 103, 104], "2023-05-17")
    g = g.with_columns(pl.lit(0.5).alias("same_clock_surprise"), pl.lit(0.1).alias("xs_breadth_5m"))
    a = attach_forward_holding_residuals(g)
    poisoned = g.with_columns(pl.when(pl.int_range(0, pl.len()) >= 3).then(pl.col("ret_5m") + 9.0).otherwise(pl.col("ret_5m")).alias("ret_5m"))
    b = attach_forward_holding_residuals(poisoned)
    assert a["same_clock_surprise"][0] == b["same_clock_surprise"][0]
    assert a["xs_breadth_5m"][0] == b["xs_breadth_5m"][0]
    assert a["beta_20d"][0] == b["beta_20d"][0]
    assert a["future_raw_15m"][0] != b["future_raw_15m"][0]


def test_checkpoint_resume_identity(tmp_path: Path, monkeypatch):
    from quant_edge_lab.discovery.v4r import checkpoint as ck
    from quant_edge_lab.discovery.v4r.stage1 import v4r_ckpt_dir

    monkeypatch.setattr(ck, "v4r_ckpt_dir", lambda root: tmp_path)
    ident = {"cfg_hash": "a", "git": "g", "manifest": "m", "gates": "g2", "science": "s"}
    ck.write_ckpt(tmp_path, "stage1", {"done": ["2022-01-03"], "last": "2022-01-03"}, ident)
    rec = ck.load_ckpt(tmp_path, "stage1", ident)
    assert rec["done"] == ["2022-01-03"]
    with pytest.raises(RuntimeError, match="incompatible"):
        ck.load_ckpt(tmp_path, "stage1", {**ident, "science": "other"})

    g = _grid([100, 101, 102, 103], "2023-05-17")
    g = g.with_columns(pl.lit(1.2).alias("beta_20d"))
    out = attach_forward_holding_residuals(g)
    shifted = g.with_columns(pl.col("ret_5m").shift(1).over("instrument_id").alias("ret_5m"))
    out2 = attach_forward_holding_residuals(shifted)
    assert out["beta_20d"][1] == out2["beta_20d"][1]
    assert out["ret_5m"][1] != out2["ret_5m"][1]
