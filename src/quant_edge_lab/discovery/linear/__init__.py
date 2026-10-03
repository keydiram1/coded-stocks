"""Frozen ridge linear benchmark. Small lambda grid on D1 only."""

from __future__ import annotations

import numpy as np


def _design(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (x - mu) / sd
    return np.hstack([np.ones((z.shape[0], 1)), z]), mu, sd


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    y = np.nan_to_num(y, nan=0.0)
    z, _, _ = _design(x)
    p = z.shape[1]
    a = z.T @ z + lam * np.eye(p)
    a[0, 0] -= lam  # do not shrink intercept
    b = z.T @ y
    return np.linalg.solve(a, b)


def predict_ridge(x: np.ndarray, coef: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    z = (x - mu) / np.where(sd < 1e-12, 1.0, sd)
    z = np.hstack([np.ones((z.shape[0], 1)), z])
    return z @ coef


def pick_lambda(x: np.ndarray, y: np.ndarray, grid: list[float]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    y = np.nan_to_num(y, nan=0.0)
    z, mu, sd = _design(x)
    best_l, best_mse, best_c = grid[0], 1e18, None
    n = z.shape[0]
    cut = int(n * 0.7) if n > 50 else n
    for lam in grid:
        c = fit_ridge(x[:cut], y[:cut], lam)
        pred = predict_ridge(x[cut:] if cut < n else x, c, mu, sd)
        yy = y[cut:] if cut < n else y
        mse = float(np.mean((pred - yy) ** 2))
        if mse < best_mse:
            best_l, best_mse, best_c = lam, mse, c
    assert best_c is not None
    return best_l, best_c, mu, sd


def decile_spread(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 50:
        return 0.0
    p, yy = pred[mask], y[mask]
    q = np.quantile(p, [0.1, 0.9])
    lo, hi = yy[p <= q[0]], yy[p >= q[1]]
    if lo.size < 5 or hi.size < 5:
        return 0.0
    return float(np.mean(hi) - np.mean(lo))
