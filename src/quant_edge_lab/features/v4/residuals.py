"""Leave-one-out market residualization. Weights use lagged info only."""

from __future__ import annotations

import numpy as np
import polars as pl


def winsor_weights(w: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    if w.size == 0:
        return w
    a, b = np.nanquantile(w, lo), np.nanquantile(w, hi)
    return np.clip(w, a, b)


def loo_market_returns(ret: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Vectorized LOO weighted mean. ret/weight same length (one timestamp cross-section)."""
    w = np.where(np.isfinite(weight) & (weight > 0) & np.isfinite(ret), weight, 0.0)
    r = np.where(np.isfinite(ret), ret, 0.0)
    num = float(np.dot(w, r))
    den = float(w.sum())
    loo_num = num - w * r
    loo_den = den - w
    out = np.divide(loo_num, loo_den, out=np.full_like(r, np.nan), where=loo_den > 1e-12)
    return out


def attach_loo_residuals(grid: pl.DataFrame, *, ret_col: str, weight_col: str, out_mkt: str, out_resid: str, beta_col: str | None = None) -> pl.DataFrame:
    if grid.height == 0:
        return grid
    parts = []
    for g in grid.partition_by("decision_ts", maintain_order=True):
        r = g[ret_col].to_numpy().astype(float)
        w = g[weight_col].to_numpy().astype(float)
        w = winsor_weights(w)
        m = loo_market_returns(r, w)
        beta = g[beta_col].to_numpy().astype(float) if beta_col and beta_col in g.columns else np.ones_like(r)
        resid = r - beta * m
        parts.append(g.with_columns(pl.Series(out_mkt, m), pl.Series(out_resid, resid)))
    return pl.concat(parts, how="diagonal_relaxed") if parts else grid
