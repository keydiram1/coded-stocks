"""Day-block White Reality Check / Hansen SPA-style family inference.

Candidates are compared to a zero-benchmark (no residual edge). Day is the
block. This is not permission to reopen D1 search.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def white_reality_check(day_means: np.ndarray, *, n_boot: int = 500, seed: int = 4) -> dict[str, Any]:
    """day_means: (n_days, n_candidates) signed mean residual on D3 (or D2).

    Statistic: max over candidates of the studentized mean (vs 0).
    Bootstrap resamples days with replacement. Null is all means <= 0 after
    recentering at the observed mean (Hansen SPA consistent-null recentering).
    """
    x = np.asarray(day_means, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_days, k = x.shape
    if n_days < 10 or k < 1:
        return {"p": None, "n_days": int(n_days), "n_candidates": int(k), "note": "insufficient_blocks"}
    mu = np.nanmean(x, axis=0)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(n_days)
    se = np.where(se < 1e-12, 1e-12, se)
    t_obs = float(np.nanmax(mu / se))
    # Hansen recentering: subtract max(mu, 0) so models with negative mean stay negative
    center = np.maximum(mu, 0.0)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_days, size=n_days)
        xb = x[idx] - center
        mub = np.nanmean(xb, axis=0)
        seb = np.nanstd(xb, axis=0, ddof=1) / np.sqrt(n_days)
        seb = np.where(seb < 1e-12, 1e-12, seb)
        t_b = float(np.nanmax(mub / seb))
        if t_b >= t_obs - 1e-15:
            exceed += 1
    p = (exceed + 1) / (n_boot + 1)
    return {
        "p": float(p),
        "t_obs": t_obs,
        "n_days": int(n_days),
        "n_candidates": int(k),
        "n_boot": n_boot,
        "procedure": "hansen_spa_dayblock_vs_zero",
    }
