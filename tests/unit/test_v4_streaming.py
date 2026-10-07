from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_edge_lab.discovery.campaign_v4 import run_pipeline_on_frame, synthetic_panel
from quant_edge_lab.discovery.campaign_v4_runner import scan_search_days
from quant_edge_lab.discovery.v4_search import run_pipeline_streaming
from quant_edge_lab.hashing import sha256_file


def test_v4_yaml_hashes_unchanged():
    root = Path(".")
    assert sha256_file(root / "knowledge/campaigns/directional_v4_manifest.yaml") == "ee59be114e76f3ecd78bd3ad2590e60c94f0754a42096818d9fbf7ee8a7b4a3e"
    assert sha256_file(root / "knowledge/campaigns/directional_v4_gates.yaml") == "02ada62546ec70681c7d884f4cb820923e6d1417e21a6d31586fc8c955b8ac38"


def test_eager_scan_blocked_for_large_day_lists(monkeypatch):
    monkeypatch.delenv("QUANT_EDGE_V4_ALLOW_EAGER", raising=False)
    with pytest.raises(Exception, match="streaming"):
        scan_search_days(Path("."), ["d"] * 41)


def test_eager_vs_streaming_planted_equivalent(tmp_path: Path):
    tiny = {
        "economic_floor_abs_mean": 0.0005,
        "max_top_ticker_share": 0.95,
        "leaf": {"min_ticker_days": 30, "min_trading_days": 5, "min_tickers": 5, "min_parent_frac": 0.1},
        "tree": {"max_depth": 3, "node_alpha": 0.5, "quantile_grid": [0.2, 0.5, 0.8]},
        "stability": {"n_subsamples": 4, "min_selection_freq": 0.0, "min_sign_agree": 0.5},
        "d3": {"abs_mean_floor": 0.0003, "require_sign_match_d2": True},
        "linear": {"lambda_grid": [0.01, 0.1]},
    }
    df = synthetic_panel(n_days=24, n_names=20, seed=7, planted=True)
    days_all = sorted(df["trading_date"].unique().to_list())
    n = len(days_all)
    i1, i2 = int(n * 0.40), int(n * 0.60)
    man = {
        "targets": {"primary": "future_residual_15m"},
        "search": {"seed": 4},
        "search_features": ["leader_response_gap", "activity_acceleration", "xs_dispersion_5m", "H004", "irrelevant_E"],
        "splits": {
            "D1": {"start": days_all[0], "end": days_all[i1 - 1], "n_days": i1},
            "D2": {"start": days_all[i1], "end": days_all[i2 - 1], "n_days": i2 - i1},
            "D3": {"start": days_all[i2], "end": days_all[-1], "n_days": n - i2},
        },
    }
    eager = run_pipeline_on_frame(df, man, tiny, features=man["search_features"])
    store = tmp_path / "data" / "derived" / "features" / "cross_sectional_v4"
    for d in days_all:
        dest = store / f"date={d}" / "part.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("trading_date") == d).write_parquet(dest)
    stream = run_pipeline_streaming(tmp_path, man, tiny, days_all[:i1], days_all[i1:i2], days_all[i2:], features=man["search_features"])
    assert eager["n_candidates"] == stream["n_candidates"]
    assert sorted(x["id"] for x in eager["d3"]) == sorted(x["id"] for x in stream["d3"])
    for a, b in zip(sorted(eager["d3"], key=lambda r: r["id"]), sorted(stream["d3"], key=lambda r: r["id"])):
        if a["d2"]["mean"] is not None and b["d2"]["mean"] is not None:
            assert abs(a["d2"]["mean"] - b["d2"]["mean"]) < 1e-10


def test_overlay_join_loads_graph_cols_when_ridge_omits_keys(tmp_path: Path):
    from quant_edge_lab.discovery.v4_search import _part_cols
    from quant_edge_lab.peers import GRAPH_FEATURE_COLS

    day = "2022-01-03"
    n = 8
    base = pl.DataFrame(
        {
            "trading_date": [day] * n,
            "instrument_id": [f"S{i}" for i in range(n)],
            "decision_ts": ["2022-01-03T14:35:00.000000+00:00"] * n,
            "resid_ret_5m": [0.01 * i for i in range(n)],
            "future_residual_15m": [0.001] * n,
        }
    )
    ov = pl.DataFrame(
        {
            "trading_date": [day] * n,
            "instrument_id": [f"S{i}" for i in range(n)],
            "decision_ts": ["2022-01-03T14:35:00.000000+00:00"] * n,
            **{c: [float(i) if c == "leader_response_gap" else 0.1 * i for i in range(n)] for c in GRAPH_FEATURE_COLS},
        }
    )
    bp = tmp_path / "base.parquet"
    op = tmp_path / "ov.parquet"
    base.write_parquet(bp)
    ov.write_parquet(op)
    got = _part_cols(bp, ["resid_ret_5m", "leader_response_gap", "future_residual_15m"], overlay=op)
    assert "leader_response_gap" in got.columns
    assert got["leader_response_gap"].to_list() == [float(i) for i in range(n)]

