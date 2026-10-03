"""Frozen K/L models: ridge, L2 logistic, shallow GBM. Chronological only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    z = (x - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, mu, sd


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    z, mu, sd = _std(x)
    yv = np.nan_to_num(y, nan=0.0)
    n, p = z.shape
    xtx = z.T @ z + alpha * np.eye(p)
    xty = z.T @ yv
    w = np.linalg.solve(xtx, xty)
    return {"kind": "ridge", "w": w, "mu": mu, "sd": sd}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((x - model["mu"]) / model["sd"], nan=0.0)
    return z @ model["w"]


def fit_logit(x: np.ndarray, y_sign: np.ndarray, l2: float = 1.0) -> dict:
    z, mu, sd = _std(x)
    y = (y_sign > 0).astype(float)
    p = z.shape[1]
    w = np.zeros(p)

    def sig(u):
        return 1 / (1 + np.exp(-np.clip(u, -30, 30)))

    for _ in range(80):
        pr = sig(z @ w)
        grad = z.T @ (pr - y) / max(len(y), 1) + l2 * w
        w = w - 0.15 * grad
    return {"kind": "logit", "w": w, "mu": mu, "sd": sd}


def predict_logit_proba(model: dict, x: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((x - model["mu"]) / model["sd"], nan=0.0)
    u = np.clip(z @ model["w"], -30, 30)
    return 1 / (1 + np.exp(-u))


@dataclass
class _Stump:
    feat: int
    thr: float
    left: float
    right: float


def _fit_stump(x: np.ndarray, resid: np.ndarray, rng: np.random.Generator) -> _Stump:
    n, p = x.shape
    feats = rng.choice(p, size=min(8, p), replace=False)
    best = _Stump(0, 0.0, 0.0, 0.0)
    best_loss = np.inf
    for j in feats:
        col = x[:, j]
        qs = np.quantile(col, [0.3, 0.5, 0.7])
        for t in qs:
            left = resid[col <= t]
            right = resid[col > t]
            if left.size < 10 or right.size < 10:
                continue
            lv, rv = float(np.mean(left)), float(np.mean(right))
            pred = np.where(col <= t, lv, rv)
            loss = float(np.mean((resid - pred) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = _Stump(int(j), float(t), lv, rv)
    return best


def fit_gbm(x: np.ndarray, y: np.ndarray, *, n_estimators: int = 40, lr: float = 0.1, seed: int = 42) -> dict:
    z, mu, sd = _std(x)
    yv = np.nan_to_num(y, nan=0.0)
    rng = np.random.default_rng(seed)
    trees: list[_Stump] = []
    pred = np.zeros(len(yv))
    for _ in range(n_estimators):
        resid = yv - pred
        st = _fit_stump(z, resid, rng)
        trees.append(st)
        col = z[:, st.feat]
        pred = pred + lr * np.where(col <= st.thr, st.left, st.right)
    return {"kind": "gbm", "trees": trees, "mu": mu, "sd": sd, "lr": lr}


def predict_gbm(model: dict, x: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((x - model["mu"]) / model["sd"], nan=0.0)
    pred = np.zeros(z.shape[0])
    lr = model["lr"]
    for st in model["trees"]:
        col = z[:, st.feat]
        pred = pred + lr * np.where(col <= st.thr, st.left, st.right)
    return pred


def score_to_side(score: np.ndarray) -> np.ndarray:
    return np.where(score >= 0, "long", "short")


def select_coverage(score: np.ndarray, coverage: float) -> np.ndarray:
    """True for tail by |score| (confidence). coverage=1 keeps all."""
    if coverage >= 0.999:
        return np.ones(len(score), dtype=bool)
    k = max(1, int(round(len(score) * coverage)))
    order = np.argsort(-np.abs(score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:k]] = True
    return mask


def rank_tails(score: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    """Within a simultaneous snapshot: top coverage LONG, bottom SHORT."""
    n = len(score)
    k = max(1, int(round(n * coverage)))
    order = np.argsort(-score)
    long_m = np.zeros(n, dtype=bool)
    short_m = np.zeros(n, dtype=bool)
    long_m[order[:k]] = True
    short_m[order[-k:]] = True
    return long_m, short_m
