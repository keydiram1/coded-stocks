"""PIT residual peer/leader graph. History strictly before as_of.

GRAPH_IMPL_VERSION v4_graph_fit_finite_topk_v1: structural top-K ranks finite
pairwise-complete correlations only. NaNs cannot dominate argsort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from quant_edge_lab.hashing import sha256_json

GRAPH_IMPL_VERSION = "v4_graph_fit_finite_topk_v1"
MIN_LAG_OVERLAP = 40
GRAPH_VALUE_COLS = [
    "leader_shock_5m",
    "leader_agreement",
    "leader_response_gap",
    "peer_return_5m",
    "peer_response_gap",
    "peer_rank_gap",
    "peer_breadth",
    "peer_dispersion",
]
GRAPH_META_COLS = ["graph_available", "leader_count", "peer_count"]
GRAPH_FEATURE_COLS = GRAPH_VALUE_COLS + GRAPH_META_COLS


@dataclass(frozen=True)
class PeerEdge:
    leader_id: str
    follower_id: str
    weight: float
    lag_minutes: int
    p_value: float
    q_value: float
    history_start: str
    history_end: str


@dataclass
class PeerGraph:
    as_of: str
    history_start: str
    history_end: str
    edges: tuple[PeerEdge, ...]
    lineage: dict[str, Any]

    def leaders_of(self, follower_id: str, k: int = 5) -> list[PeerEdge]:
        hits = [e for e in self.edges if e.follower_id == follower_id]
        hits.sort(key=lambda e: e.weight, reverse=True)
        return hits[:k]


def _bh(p: np.ndarray) -> np.ndarray:
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, _idx in enumerate(order[::-1], start=1):
        i = order[-rank]
        val = min(prev, p[i] * n / (n - rank + 1))
        q[i] = val
        prev = val
    return np.clip(q, 0, 1)


def topk_finite(scores: np.ndarray, k: int, *, exclude: np.ndarray | None = None) -> np.ndarray:
    """Largest k finite scores. NaN/inf never rank. exclude is a boolean mask of banned indices."""
    s = np.asarray(scores, dtype=float).copy()
    if exclude is not None:
        s[np.asarray(exclude, dtype=bool)] = np.nan
    s[~np.isfinite(s)] = np.nan
    idx = np.flatnonzero(np.isfinite(s))
    if idx.size == 0 or k <= 0:
        return np.array([], dtype=int)
    take = min(int(k), idx.size)
    order = idx[np.argsort(s[idx], kind="mergesort")[::-1][:take]]
    return order


def pairwise_corrcoef(mat: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson on columns (T, N). Non-finite filled as NaN in the result."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.ma.corrcoef(np.ma.masked_invalid(mat), rowvar=False)
    if np.ndim(c) == 0:
        return np.array([[float(c)]], dtype=float)
    return np.array(np.ma.filled(c, np.nan), dtype=float)


def structural_topk_buggy_nan_argsort(cor: np.ndarray, k: int, self_idx: int) -> np.ndarray:
    """Attempt-1 selector (for regression only): NaNs sort to the end then reverse-take first."""
    c = np.abs(np.asarray(cor, dtype=float)).copy()
    c[self_idx] = -1
    return np.argsort(c)[::-1][:k]


class PeerGraphBuilder:
    def __init__(self, *, history_days: int = 20, structural_top: int = 20, leaders_top: int = 5, lag_minutes: int = 5, q_edge: float = 0.10):
        self.history_days = history_days
        self.structural_top = structural_top
        self.leaders_top = leaders_top
        self.lag_minutes = lag_minutes
        self.q_edge = q_edge

    def fit(self, history: pl.DataFrame, as_of: str) -> PeerGraph:
        h = history.filter(pl.col("trading_date") < as_of)
        if h.height == 0 or "resid_ret_5m" not in h.columns:
            return PeerGraph(as_of, "", "", (), {"as_of": as_of, "n_rows": 0, "n_tests": 0, "note": "empty history", "graph_impl": GRAPH_IMPL_VERSION})
        days = sorted(h["trading_date"].unique().to_list())[-self.history_days :]
        h = h.filter(pl.col("trading_date").is_in(days))
        key = "clock" if "clock" in h.columns else ("time_et" if "time_et" in h.columns else None)
        if key is None:
            h = h.with_columns(pl.col("decision_ts").cast(pl.Utf8).alias("clock"))
            key = "clock"
        wide = h.select(["instrument_id", "trading_date", key, "resid_ret_5m"]).pivot(
            on="instrument_id", index=["trading_date", key], values="resid_ret_5m", aggregate_function="first"
        )
        feat_cols = [c for c in wide.columns if c not in {"trading_date", key}]
        if len(feat_cols) < 3 or wide.height < 30:
            return PeerGraph(
                as_of,
                str(days[0]),
                str(days[-1]),
                (),
                {"as_of": as_of, "n_rows": h.height, "n_tests": 0, "note": "insufficient panel", "graph_impl": GRAPH_IMPL_VERSION},
            )
        mat = wide.select(feat_cols).to_numpy().astype(float)
        n_obs = np.isfinite(mat).sum(axis=0)
        usable = n_obs >= MIN_LAG_OVERLAP
        cmat = pairwise_corrcoef(mat)
        raw_edges: list[tuple[str, str, float, float]] = []
        pvals: list[float] = []
        n_structural = 0
        ylag = mat[1:]
        xnow = mat[:-1]
        for j, fol in enumerate(feat_cols):
            if not usable[j]:
                continue
            cor = np.abs(cmat[j])
            ban = ~usable
            ban[j] = True
            cand_idx = topk_finite(cor, self.structural_top, exclude=ban)
            n_structural += int(cand_idx.size)
            for i in cand_idx:
                ii = int(i)
                if ii == j:
                    continue
                x = xnow[:, ii]
                y = ylag[:, j]
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < MIN_LAG_OVERLAP:
                    continue
                r, p = stats.pearsonr(x[mask], y[mask])
                if not np.isfinite(r) or not np.isfinite(p):
                    continue
                raw_edges.append((feat_cols[ii], fol, float(r), float(p)))
                pvals.append(float(p))
        q = _bh(np.array(pvals, dtype=float)) if pvals else np.array([])
        edges: list[PeerEdge] = []
        rs: list[float] = []
        for k, (lead, fol, r, p) in enumerate(raw_edges):
            if q[k] <= self.q_edge and r > 0:
                edges.append(
                    PeerEdge(lead, fol, float(r), self.lag_minutes, float(p), float(q[k]), str(days[0]), str(days[-1]))
                )
                rs.append(abs(float(r)))
        by_f: dict[str, list[PeerEdge]] = {}
        for e in edges:
            by_f.setdefault(e.follower_id, []).append(e)
        kept: list[PeerEdge] = []
        for _fol, el in by_f.items():
            el.sort(key=lambda z: z.weight, reverse=True)
            kept.extend(el[: self.leaders_top])
        lineage = {
            "as_of": as_of,
            "history_start": str(days[0]),
            "history_end": str(days[-1]),
            "n_history_rows": int(h.height),
            "n_names": len(feat_cols),
            "n_usable_names": int(usable.sum()),
            "n_wide_rows": int(mat.shape[0]),
            "structural_pairs_considered": n_structural,
            "n_tests": len(raw_edges),
            "n_bh_survivors": len(edges),
            "n_edges": len(kept),
            "median_abs_r": float(np.median(rs)) if rs else None,
            "graph_impl": GRAPH_IMPL_VERSION,
            "config": {
                "history_days": self.history_days,
                "structural_top": self.structural_top,
                "leaders_top": self.leaders_top,
                "lag_minutes": self.lag_minutes,
                "q_edge": self.q_edge,
                "min_lag_overlap": MIN_LAG_OVERLAP,
            },
            "schema_hash": sha256_json({"cols": feat_cols[:50], "n": len(feat_cols)}),
        }
        return PeerGraph(as_of, str(days[0]), str(days[-1]), tuple(kept), lineage)


def graph_to_record(g: PeerGraph, *, cfg_hash: str | None = None) -> dict[str, Any]:
    return {
        "as_of": g.as_of,
        "history_start": g.history_start,
        "history_end": g.history_end,
        "n_edges": len(g.edges),
        "lineage": g.lineage,
        "graph_impl": GRAPH_IMPL_VERSION,
        "cfg_hash": cfg_hash,
        "edges": [
            {
                "leader_id": e.leader_id,
                "follower_id": e.follower_id,
                "weight": e.weight,
                "lag_minutes": e.lag_minutes,
                "p_value": e.p_value,
                "q_value": e.q_value,
            }
            for e in g.edges
        ],
    }


def record_to_graph(rec: dict[str, Any]) -> PeerGraph:
    hs, he = rec.get("history_start") or "", rec.get("history_end") or ""
    edges = tuple(
        PeerEdge(
            str(e["leader_id"]),
            str(e["follower_id"]),
            float(e["weight"]),
            int(e.get("lag_minutes") or 5),
            float(e["p_value"]),
            float(e["q_value"]),
            hs,
            he,
        )
        for e in (rec.get("edges") or [])
    )
    return PeerGraph(str(rec["as_of"]), hs, he, edges, dict(rec.get("lineage") or {}))


def _null_graph_features(grid: pl.DataFrame) -> pl.DataFrame:
    out = grid
    for c in GRAPH_VALUE_COLS:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
    return out.with_columns(
        pl.lit(False).alias("graph_available"),
        pl.lit(0).cast(pl.Int32).alias("leader_count"),
        pl.lit(0).cast(pl.Int32).alias("peer_count"),
    )


def attach_peer_features(grid: pl.DataFrame, graph: PeerGraph) -> pl.DataFrame:
    """Join leader/peer stats. Missing graph info is null, never -resid_ret_5m."""
    drop = [c for c in GRAPH_FEATURE_COLS if c in grid.columns]
    if drop:
        grid = grid.drop(drop)
    if grid.height == 0:
        return grid
    lead_map: dict[str, list[PeerEdge]] = {}
    for e in graph.edges:
        lead_map.setdefault(e.follower_id, []).append(e)
    recs = []
    for fol, eds in lead_map.items():
        wsum = sum(abs(e.weight) for e in eds) or 1.0
        for e in eds:
            recs.append({"instrument_id": fol, "leader_id": e.leader_id, "w": abs(e.weight) / wsum, "leader_count": len(eds)})
    if not recs:
        return _null_graph_features(grid)
    rel = pl.DataFrame(recs)
    lead = grid.select(["decision_ts", "instrument_id", "resid_ret_5m"]).rename({"instrument_id": "leader_id", "resid_ret_5m": "leader_resid"})
    joined = grid.select(["decision_ts", "instrument_id"]).join(rel, on="instrument_id", how="left").join(lead, on=["decision_ts", "leader_id"], how="left")
    agg = joined.group_by(["decision_ts", "instrument_id"]).agg(
        (pl.col("w") * pl.col("leader_resid")).sum().alias("leader_shock_5m"),
        ((pl.col("leader_resid") > 0).cast(pl.Float64) * pl.col("w")).sum().alias("leader_agreement"),
        pl.col("leader_resid").mean().alias("peer_return_5m"),
        (pl.col("leader_resid") > 0).mean().alias("peer_breadth"),
        pl.col("leader_resid").abs().median().alias("peer_dispersion"),
        pl.col("leader_id").drop_nulls().n_unique().alias("leader_count"),
    )
    out = grid.join(agg, on=["decision_ts", "instrument_id"], how="left")
    if "resid_ret_5m" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("leader_shock_5m").is_not_null())
            .then(pl.col("leader_shock_5m") - pl.col("resid_ret_5m"))
            .otherwise(None)
            .alias("leader_response_gap"),
            pl.when(pl.col("peer_return_5m").is_not_null())
            .then(pl.col("peer_return_5m") - pl.col("resid_ret_5m"))
            .otherwise(None)
            .alias("peer_response_gap"),
        )
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("leader_response_gap"), pl.lit(None).cast(pl.Float64).alias("peer_response_gap"))
    if "resid_rank_5m" in out.columns:
        ranked = pl.when(pl.col("peer_return_5m").is_not_null()).then(
            pl.col("peer_return_5m").rank().over("decision_ts") / pl.len().over("decision_ts") - pl.col("resid_rank_5m")
        )
        out = out.with_columns(ranked.alias("peer_rank_gap"))
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Float64).alias("peer_rank_gap"))
    out = out.with_columns(
        (pl.col("leader_count").fill_null(0) > 0).alias("graph_available"),
        pl.col("leader_count").fill_null(0).cast(pl.Int32),
        pl.col("leader_count").fill_null(0).cast(pl.Int32).alias("peer_count"),
    )
    return out
