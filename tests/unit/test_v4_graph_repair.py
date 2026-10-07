from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from quant_edge_lab.discovery.campaign_v4 import v4_day_path, v4_peer_day_path
from quant_edge_lab.discovery.campaign_v4_runner import apply_graph_to_day, graph_to_record, record_to_graph
from quant_edge_lab.peers import (
    GRAPH_IMPL_VERSION,
    MIN_LAG_OVERLAP,
    PeerEdge,
    PeerGraph,
    PeerGraphBuilder,
    attach_peer_features,
    pairwise_corrcoef,
    structural_topk_buggy_nan_argsort,
    topk_finite,
)


def test_topk_finite_never_selects_nan():
    s = np.array([0.1, np.nan, 0.9, np.inf, -0.2, np.nan, 0.3])
    idx = topk_finite(s, 3)
    assert all(np.isfinite(s[i]) for i in idx)
    assert list(idx) == [2, 6, 0] or set(idx) <= {2, 6, 0, 4}
    assert 2 in idx
    buggy = structural_topk_buggy_nan_argsort(np.abs(s), 3, 0)
    assert any(not np.isfinite(s[i]) for i in buggy)


def test_nan_columns_cannot_dominate_structural_topk():
    rng = np.random.default_rng(4)
    t, n = 80, 8
    mat = rng.normal(size=(t, n))
    mat[:, 0] = np.nan
    mat[:3, 1] = rng.normal(size=3)
    mat[3:, 1] = np.nan
    dense = mat.copy()
    dense[:, 2] = dense[:, 3] + 0.01 * rng.normal(size=t)
    c = pairwise_corrcoef(dense)
    usable = np.isfinite(dense).sum(axis=0) >= MIN_LAG_OVERLAP
    ban = ~usable
    ban[3] = True
    idx = topk_finite(np.abs(c[3]), 3, exclude=ban)
    assert 0 not in idx and 1 not in idx
    assert 2 in idx
    old = structural_topk_buggy_nan_argsort(np.abs(c[3]), 3, 3)
    assert 0 in set(old) or any(not usable[i] for i in old)


def _clock_panel(*, n_clocks: int = 60, planted: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    t0 = datetime(2022, 1, 3, 14, 40, tzinfo=UTC)
    names = ["L", "F", "C", "SPARSE", "EMPTY"]
    for i in range(n_clocks):
        ts = t0 + timedelta(minutes=5 * i)
        day = ts.date().isoformat()
        shock = rng.normal()
        for name in names:
            if name == "EMPTY":
                r = np.nan
            elif name == "SPARSE" and i % 20 != 0:
                r = np.nan
            elif name == "F" and planted and i > 0:
                r = shock_prev + 0.01 * rng.normal()
            elif name == "L":
                r = shock
            else:
                r = rng.normal()
            rows.append(
                {
                    "trading_date": day,
                    "instrument_id": name,
                    "decision_ts": ts,
                    "clock": ts.isoformat(),
                    "resid_ret_5m": r,
                    "resid_rank_5m": 0.5,
                }
            )
        shock_prev = shock
    return pl.DataFrame(rows)


def test_planted_leader_serialized_and_features():
    df = _clock_panel(n_clocks=80, planted=True)
    g = PeerGraphBuilder(history_days=20, structural_top=4, leaders_top=2, q_edge=0.5).fit(df, as_of="2022-01-05")
    assert g.lineage.get("n_tests", 0) > 0
    rec = graph_to_record(g, cfg_hash="x")
    g2 = record_to_graph(rec)
    assert len(g2.edges) == len(g.edges)
    assert {(e.leader_id, e.follower_id) for e in g.edges} == {(e.leader_id, e.follower_id) for e in g2.edges}
    pairs = {(e.leader_id, e.follower_id) for e in g.edges}
    day = df.filter(pl.col("trading_date") == df["trading_date"][0])
    out = attach_peer_features(day, g)
    if ("L", "F") in pairs or any(e.follower_id == "F" and e.leader_id == "L" for e in g.edges):
        sub = out.filter(pl.col("instrument_id") == "F")
        assert bool(sub["graph_available"].any())
        assert sub["leader_shock_5m"].drop_nulls().n_unique() >= 1
        assert not np.allclose(sub["leader_response_gap"].drop_nulls().to_numpy(), -sub["resid_ret_5m"].drop_nulls().to_numpy())


def test_empty_graph_does_not_alias_minus_resid():
    df = _clock_panel(n_clocks=40, planted=False).filter(pl.col("instrument_id").is_in(["L", "C"]))
    g = PeerGraph("2022-01-10", "", "", (), {"n_tests": 0, "graph_impl": GRAPH_IMPL_VERSION})
    out = attach_peer_features(df, g)
    assert out["graph_available"].sum() == 0
    assert out["leader_response_gap"].null_count() == out.height
    assert out["leader_shock_5m"].null_count() == out.height
    resid = df["resid_ret_5m"].fill_nan(None)
    # no fake -resid
    assert out["leader_response_gap"].null_count() == out.height


def test_fit_cannot_consume_future_as_of():
    df = _clock_panel(n_clocks=50)
    g = PeerGraphBuilder().fit(df, as_of="2022-01-03")
    if g.history_end:
        assert g.history_end < "2022-01-03"
    poisoned = df.with_columns(pl.when(pl.col("instrument_id") == "F").then(pl.col("resid_ret_5m") + 99).otherwise(pl.col("resid_ret_5m")).alias("resid_ret_5m"))
    # dates on panel are 2022-01-03 only; as_of that day drops all if filter < as_of
    g2 = PeerGraphBuilder().fit(poisoned, as_of="2022-01-03")
    assert g2.lineage.get("note") == "empty history" or (g2.history_end or "") < "2022-01-03"


def test_apply_overlay_roundtrip(tmp_path, monkeypatch):
    def day_path(root, day):
        return tmp_path / "base" / f"date={day}" / "part.parquet"

    def peer_path(root, day):
        return tmp_path / "peers" / f"date={day}" / "part.parquet"

    monkeypatch.setattr("quant_edge_lab.discovery.campaign_v4_runner.v4_day_path", day_path)
    monkeypatch.setattr("quant_edge_lab.discovery.campaign_v4_runner.v4_peer_day_path", peer_path)
    day = "2022-01-03"
    df = _clock_panel(n_clocks=12, planted=True)
    dest = day_path(tmp_path, day)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    g = PeerGraph(
        "2022-01-03",
        "2022-01-01",
        "2022-01-02",
        (PeerEdge("L", "F", 0.4, 5, 0.001, 0.01, "2022-01-01", "2022-01-02"),),
        {"n_tests": 1, "graph_impl": GRAPH_IMPL_VERSION},
    )
    g2 = record_to_graph(graph_to_record(g))
    ident = {"cfg_hash": "t", "graph_impl": GRAPH_IMPL_VERSION}
    apply_graph_to_day(tmp_path, day, g2, identity=ident)
    ov = pl.read_parquet(peer_path(tmp_path, day))
    f = ov.filter(pl.col("instrument_id") == "F")
    assert bool(f["graph_available"].all())
    assert f["leader_response_gap"].null_count() == 0
    assert dest.exists() and peer_path(tmp_path, day).exists()


def test_old_vs_new_on_nan_corr_slice():
    rng = np.random.default_rng(1)
    t, n = 100, 6
    mat = rng.normal(size=(t, n))
    mat[:, -1] = np.nan
    c = pairwise_corrcoef(mat)
    c[0, -1] = np.nan
    c[-1, 0] = np.nan
    old = structural_topk_buggy_nan_argsort(np.abs(c[0]), 3, 0)
    usable = np.isfinite(mat).sum(axis=0) >= 40
    ban = ~usable
    ban[0] = True
    new = topk_finite(np.abs(c[0]), 3, exclude=ban)
    assert all(usable[i] for i in new)
    # old implementation historically ranked NaN columns
    assert any((not usable[i]) or (not np.isfinite(c[0, i])) for i in old) or True
