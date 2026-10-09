"""Matched incremental day-delta. Hierarchical H1 gate, not a fourth hypothesis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.discovery.v5.evaluation import onesided_greater_p
from quant_edge_lab.statistics.summarize import _block_mean_distribution


def stratum_key(sign: str, z_bucket: str, t1_bucket: str) -> tuple[str, str, str]:
    return (sign, z_bucket, t1_bucket)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    s, z, t = row.get("impulse_sign"), row.get("abs_z_bucket"), row.get("t1_bucket")
    if not s or not z or not t:
        return None
    if s not in {"POSITIVE", "NEGATIVE"}:
        return None
    return stratum_key(str(s), str(z), str(t))


def matched_day_deltas(
    treated: pl.DataFrame,
    control: pl.DataFrame,
    *,
    col: str = "primary_signed",
) -> list[float]:
    """One delta per trading day that has at least one two-sided stratum. No imputation."""
    if treated.height == 0 or control.height == 0:
        return []
    t_by: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    c_by: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in treated.iter_rows(named=True):
        key = _row_key(row)
        y = row.get(col)
        if key is None or y is None or not np.isfinite(float(y)):
            continue
        t_by[str(row["trading_date"])][key].append(float(y))
    for row in control.iter_rows(named=True):
        key = _row_key(row)
        y = row.get(col)
        if key is None or y is None or not np.isfinite(float(y)):
            continue
        c_by[str(row["trading_date"])][key].append(float(y))
    days = sorted(set(t_by) | set(c_by))
    out: list[float] = []
    for day in days:
        deltas: list[float] = []
        keys = set(t_by[day]) & set(c_by[day])
        for key in keys:
            deltas.append(float(np.mean(t_by[day][key]) - np.mean(c_by[day][key])))
        if deltas:
            out.append(float(np.mean(deltas)))
    return out


def incremental_stats(
    day_deltas: list[float], *, n_boot: int = 1000, seed: int = 42
) -> dict[str, Any]:
    arr = np.asarray(day_deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n_matched_days": 0,
            "mean_day_delta": None,
            "se": None,
            "p_one_sided": 1.0,
            "ci": None,
            "mean_bp": None,
        }
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else None
    ci = None
    if n > 1:
        blocks = np.array([str(i) for i in range(n)])
        dist = _block_mean_distribution(arr, blocks, n_boot, seed)
        ci = [float(np.nanquantile(dist, 0.025)), float(np.nanquantile(dist, 0.975))]
    return {
        "n_matched_days": n,
        "mean_day_delta": mean,
        "se": se,
        "p_one_sided": onesided_greater_p(arr),
        "ci": ci,
        "mean_bp": mean * 10_000.0,
    }


def incremental_d1_eligible(stats: dict[str, Any]) -> bool:
    m = stats.get("mean_day_delta")
    return m is not None and float(m) > 0


def incremental_confirmatory_ok(stats: dict[str, Any]) -> tuple[bool, str | None]:
    n = int(stats.get("n_matched_days") or 0)
    m = stats.get("mean_day_delta")
    p = stats.get("p_one_sided")
    if n < 100:
        return False, "incremental_n_matched_days"
    if m is None or float(m) <= 0:
        return False, "incremental_mean"
    if p is None or float(p) >= 0.05:
        return False, "incremental_p"
    return True, None
