"""Day-aware association: one statistic per trading day, inference across days."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from quant_edge_lab.discovery.conditional.association import bh_qvalues


def _daily_spearman(df: pl.DataFrame, feat: str, target: str) -> tuple[float, dict]:
    rhos: list[float] = []
    n_rows = 0
    n_tickers = 0
    for day, g in df.group_by("trading_date", maintain_order=True):
        xy = g.select([feat, target]).drop_nulls()
        if feat in xy.columns:
            xy = xy.filter(pl.col(feat).is_finite())
        if target in xy.columns:
            xy = xy.filter(pl.col(target).is_finite())
        n_rows += xy.height
        if "instrument_id" in g.columns:
            n_tickers += int(g["instrument_id"].n_unique())
        if xy.height < 10:
            continue
        r, _p = stats.spearmanr(xy[feat].to_numpy().astype(float), xy[target].to_numpy().astype(float))
        if np.isfinite(r):
            rhos.append(float(r))
    arr = np.array(rhos, dtype=float)
    meta = {
        "n_rows": n_rows,
        "n_days": int(arr.size),
        "n_tickers": n_tickers,
        "daily_effect_mean": float(arr.mean()) if arr.size else None,
        "daily_effect_median": float(np.median(arr)) if arr.size else None,
        "aggregate_effect": float(arr.mean()) if arr.size else None,
    }
    if arr.size < 8:
        return 1.0, {**meta, "p_raw": 1.0}
    t_p = stats.ttest_1samp(arr, 0.0).pvalue
    p = float(t_p) if np.isfinite(t_p) else 1.0
    meta["p_raw"] = p
    return p, meta


def _daily_binary_diff(df: pl.DataFrame, feat: str, target: str) -> tuple[float, dict]:
    diffs: list[float] = []
    n_rows = 0
    for _day, g in df.group_by("trading_date", maintain_order=True):
        xy = g.select([feat, target]).drop_nulls()
        n_rows += xy.height
        if xy.height < 20:
            continue
        x = xy[feat].to_numpy()
        y = xy[target].to_numpy().astype(float)
        # coerce to 0/1
        xb = x.astype(float)
        vals = np.unique(xb[np.isfinite(xb)])
        if vals.size < 2:
            continue
        a = y[xb == vals[0]]
        b = y[xb == vals[-1]]
        if a.size < 5 or b.size < 5:
            continue
        diffs.append(float(np.nanmean(b) - np.nanmean(a)))
    arr = np.array(diffs, dtype=float)
    meta = {
        "n_rows": n_rows,
        "n_days": int(arr.size),
        "daily_effect_mean": float(arr.mean()) if arr.size else None,
        "aggregate_effect": float(arr.mean()) if arr.size else None,
    }
    if arr.size < 8:
        return 1.0, {**meta, "p_raw": 1.0}
    t_p = stats.ttest_1samp(arr, 0.0).pvalue
    p = float(t_p) if np.isfinite(t_p) else 1.0
    meta["p_raw"] = p
    return p, meta


def is_binary_series(s: pl.Series) -> bool:
    u = s.drop_nulls().unique()
    if u.len() > 2:
        return False
    return set(u.to_list()) <= {True, False, 0, 1, 0.0, 1.0}


def node_association(df: pl.DataFrame, features: list[str], target: str) -> tuple[list[tuple[str, float]], list[dict], np.ndarray]:
    tests: list[tuple[str, float]] = []
    lineage: list[dict] = []
    for feat in features:
        if feat not in df.columns:
            continue
        if is_binary_series(df[feat]):
            p, meta = _daily_binary_diff(df, feat, target)
        else:
            p, meta = _daily_spearman(df, feat, target)
        tests.append((feat, p))
        lineage.append({"feature": feat, "decision": "considered", "p_raw": p, **meta})
    q = bh_qvalues(np.array([p for _, p in tests], dtype=float)) if tests else np.array([])
    for i, rec in enumerate(lineage):
        rec["q_adjusted"] = float(q[i]) if i < q.size else None
    return tests, lineage, q
