"""Day-block association tests and BH adjustment for V4 CIT."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        val = min(prev, p[i] * n / rank)
        q[i] = val
        prev = val
    return np.clip(q, 0, 1)


def spearman_dayblock(x: np.ndarray, y: np.ndarray, days: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 1.0
    r, p = stats.spearmanr(x[mask], y[mask])
    if not np.isfinite(p):
        return 1.0
    # Inflate p if few independent days
    nd = len(set(days[mask].tolist()))
    if nd < 8:
        return 1.0
    return float(p)


def binary_ttest(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 1.0
    xb = x[mask].astype(float)
    uniq = np.unique(xb)
    if uniq.size < 2:
        return 1.0
    a = y[mask][xb == uniq[0]]
    b = y[mask][xb == uniq[-1]]
    if a.size < 10 or b.size < 10:
        return 1.0
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return float(p) if np.isfinite(p) else 1.0


def heterogeneity_score(left_y: np.ndarray, right_y: np.ndarray) -> float:
    if left_y.size < 5 or right_y.size < 5:
        return 0.0
    return abs(float(np.nanmean(left_y) - np.nanmean(right_y)))
