"""PIT residual peer/leader graph. History strictly before as_of."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from quant_edge_lab.hashing import sha256_json


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
    for rank, idx in enumerate(order[::-1], start=1):
        i = order[-rank]
        val = min(prev, p[i] * n / (n - rank + 1))
        q[i] = val
        prev = val
    return np.clip(q, 0, 1)


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
            return PeerGraph(as_of, "", "", (), {"as_of": as_of, "n_rows": 0, "note": "empty history"})
        days = sorted(h["trading_date"].unique().to_list())[-self.history_days :]
        h = h.filter(pl.col("trading_date").is_in(days))
        ids = sorted(h["instrument_id"].unique().to_list())
        # Wide residual matrix at a coarse clock (use time_et if present else row order).
        key = "clock" if "clock" in h.columns else ("time_et" if "time_et" in h.columns else None)
        if key is None:
            h = h.with_columns(pl.col("decision_ts").cast(pl.Utf8).alias("clock"))
            key = "clock"
        wide = h.select(["instrument_id", "trading_date", key, "resid_ret_5m"]).pivot(
            on="instrument_id", index=["trading_date", key], values="resid_ret_5m", aggregate_function="first"
        )
        feat_cols = [c for c in wide.columns if c not in {"trading_date", key}]
        if len(feat_cols) < 3 or wide.height < 30:
            return PeerGraph(as_of, str(days[0]), str(days[-1]), (), {"as_of": as_of, "n_rows": h.height, "note": "insufficient panel"})
        mat = wide.select(feat_cols).to_numpy().astype(float)
        # Correlation prescreen then lagged OLS p-values on top-k.
        with np.errstate(invalid="ignore"):
            cmat = np.corrcoef(np.nan_to_num(mat, nan=0.0), rowvar=False)
        edges: list[PeerEdge] = []
        pvals: list[float] = []
        raw_edges: list[tuple[str, str, float, float]] = []
        n = mat.shape[0]
        ylag = mat[1:]
        xnow = mat[:-1]
        for j, fol in enumerate(feat_cols):
            cor = np.abs(cmat[j])
            cor[j] = -1
            cand_idx = np.argsort(cor)[::-1][: self.structural_top]
            for i in cand_idx:
                if i == j:
                    continue
                x = xnow[:, i]
                y = ylag[:, j]
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 40:
                    continue
                r, p = stats.pearsonr(x[mask], y[mask])
                raw_edges.append((feat_cols[i], fol, float(r), float(p)))
                pvals.append(float(p))
        q = _bh(np.array(pvals, dtype=float)) if pvals else np.array([])
        for k, (lead, fol, r, p) in enumerate(raw_edges):
            if q[k] <= self.q_edge and r > 0:
                edges.append(
                    PeerEdge(lead, fol, float(r), self.lag_minutes, float(p), float(q[k]), str(days[0]), str(days[-1]))
                )
        # Keep top leaders per follower
        by_f: dict[str, list[PeerEdge]] = {}
        for e in edges:
            by_f.setdefault(e.follower_id, []).append(e)
        kept: list[PeerEdge] = []
        for fol, el in by_f.items():
            el.sort(key=lambda z: z.weight, reverse=True)
            kept.extend(el[: self.leaders_top])
        lineage = {
            "as_of": as_of,
            "history_start": str(days[0]),
            "history_end": str(days[-1]),
            "n_history_rows": int(h.height),
            "n_names": len(feat_cols),
            "n_tests": len(raw_edges),
            "n_edges": len(kept),
            "config": {
                "history_days": self.history_days,
                "structural_top": self.structural_top,
                "leaders_top": self.leaders_top,
                "lag_minutes": self.lag_minutes,
                "q_edge": self.q_edge,
            },
            "schema_hash": sha256_json({"cols": feat_cols[:50], "n": len(feat_cols)}),
        }
        return PeerGraph(as_of, str(days[0]), str(days[-1]), tuple(kept), lineage)


def attach_peer_features(grid: pl.DataFrame, graph: PeerGraph) -> pl.DataFrame:
    if grid.height == 0:
        return grid
    lead_map: dict[str, list[PeerEdge]] = {}
    for e in graph.edges:
        lead_map.setdefault(e.follower_id, []).append(e)
    # Map instrument -> resid at this timestamp via join of leaders.
    xs = grid.select(["decision_ts", "instrument_id", "resid_ret_5m", "resid_rank_5m"] if "resid_rank_5m" in grid.columns else ["decision_ts", "instrument_id", "resid_ret_5m"])
    recs = []
    for fol, eds in lead_map.items():
        wsum = sum(abs(e.weight) for e in eds) or 1.0
        for e in eds:
            recs.append({"instrument_id": fol, "leader_id": e.leader_id, "w": abs(e.weight) / wsum})
    if not recs:
        return grid.with_columns(
            pl.lit(0.0).alias("leader_shock_5m"),
            pl.lit(0.0).alias("leader_agreement"),
            (pl.lit(0.0) - pl.col("resid_ret_5m")).alias("leader_response_gap") if "resid_ret_5m" in grid.columns else pl.lit(0.0).alias("leader_response_gap"),
            pl.lit(0.0).alias("peer_return_5m"),
            pl.lit(0.0).alias("peer_response_gap"),
            pl.lit(0.0).alias("peer_rank_gap"),
            pl.lit(0.0).alias("peer_breadth"),
            pl.lit(0.0).alias("peer_dispersion"),
        )
    rel = pl.DataFrame(recs)
    lead = grid.select(["decision_ts", "instrument_id", "resid_ret_5m"]).rename({"instrument_id": "leader_id", "resid_ret_5m": "leader_resid"})
    joined = grid.select(["decision_ts", "instrument_id"]).join(rel, on="instrument_id", how="left").join(lead, on=["decision_ts", "leader_id"], how="left")
    agg = joined.group_by(["decision_ts", "instrument_id"]).agg(
        (pl.col("w") * pl.col("leader_resid")).sum().alias("leader_shock_5m"),
        ((pl.col("leader_resid") > 0).cast(pl.Float64) * pl.col("w")).sum().alias("leader_agreement"),
        pl.col("leader_resid").mean().alias("peer_return_5m"),
        (pl.col("leader_resid") > 0).mean().alias("peer_breadth"),
        pl.col("leader_resid").abs().median().alias("peer_dispersion"),
    )
    out = grid.join(agg, on=["decision_ts", "instrument_id"], how="left")
    out = out.with_columns(
        pl.col("leader_shock_5m").fill_null(0.0),
        pl.col("leader_agreement").fill_null(0.0),
        pl.col("peer_return_5m").fill_null(0.0),
        pl.col("peer_breadth").fill_null(0.0),
        pl.col("peer_dispersion").fill_null(0.0),
    )
    if "resid_ret_5m" in out.columns:
        out = out.with_columns(
            (pl.col("leader_shock_5m") - pl.col("resid_ret_5m")).alias("leader_response_gap"),
            (pl.col("peer_return_5m") - pl.col("resid_ret_5m")).alias("peer_response_gap"),
        )
    if "resid_rank_5m" in out.columns:
        out = out.with_columns((pl.col("peer_return_5m").rank().over("decision_ts") / pl.len().over("decision_ts") - pl.col("resid_rank_5m")).alias("peer_rank_gap"))
    else:
        out = out.with_columns(pl.lit(0.0).alias("peer_rank_gap"))
    return out
