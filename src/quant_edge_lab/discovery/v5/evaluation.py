"""Day-aware evaluation and frozen D1→D2→D3 decisions. primary_signed is already signed."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from quant_edge_lab.discovery.v5.models import CandidateRule, ScientificDecision
from quant_edge_lab.statistics.summarize import _block_mean_distribution, benjamini_hochberg


def trial_count(campaign_yaml: dict[str, Any]) -> int:
    return len(campaign_yaml.get("hypotheses") or [])


def finite_primary(events: pl.DataFrame, col: str = "primary_signed") -> pl.DataFrame:
    if events.height == 0 or col not in events.columns:
        return events.head(0) if events.height else events
    return events.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())


def concentration(events: pl.DataFrame) -> dict[str, Any]:
    if events.height == 0:
        return {
            "events": 0,
            "ticker_days": 0,
            "trading_days": 0,
            "tickers": 0,
            "top_ticker_share": None,
        }
    td = events.select(["instrument_id", "trading_date"]).unique().height
    days = events["trading_date"].n_unique()
    tickers = events["instrument_id"].n_unique()
    vc = events.group_by("instrument_id").len().sort("len", descending=True)
    top = float(vc["len"][0] / events.height)
    return {
        "events": events.height,
        "ticker_days": int(td),
        "trading_days": int(days),
        "tickers": int(tickers),
        "top_ticker_share": top,
    }


def onesided_greater_p(daily_means: np.ndarray) -> float:
    arr = np.asarray(daily_means, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return 1.0
    res = stats.ttest_1samp(arr, 0.0, alternative="greater")
    p = float(res.pvalue)
    return p if np.isfinite(p) else 1.0


def signed_floor(gates: dict[str, Any], split: str) -> float:
    key = split.lower()
    if key not in gates or "signed_mean_floor" not in gates[key]:
        raise KeyError(f"gates.{key}.signed_mean_floor is required; no Python default floor")
    return float(gates[key]["signed_mean_floor"])


def day_block_stats(events: pl.DataFrame, col: str = "primary_signed", *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    ydf = finite_primary(events, col)
    conc = concentration(ydf)
    if ydf.height == 0:
        return {
            **conc,
            "mean": None,
            "median": None,
            "win_rate": None,
            "se": None,
            "ci": None,
            "mean_bp": None,
            "p_one_sided": 1.0,
            "daily_means": [],
        }
    y = ydf[col].to_numpy().astype(float)
    blocks = ydf["trading_date"].cast(pl.Utf8).to_numpy()
    if y.size != blocks.size:
        raise AssertionError("bootstrap values and block labels length mismatch")
    mean = float(np.mean(y))
    by = ydf.group_by("trading_date").agg(pl.col(col).mean().alias("m"))
    dm = by["m"].drop_nulls().to_numpy().astype(float)
    se = float(np.std(dm, ddof=1) / np.sqrt(dm.size)) if dm.size > 1 else None
    ci = None
    if y.size > 1:
        dist = _block_mean_distribution(y, blocks, n_boot, seed)
        ci = [float(np.nanquantile(dist, 0.025)), float(np.nanquantile(dist, 0.975))]
    return {
        **conc,
        "mean": mean,
        "median": float(np.median(y)),
        "win_rate": float(np.mean(y > 0)),
        "se": se,
        "ci": ci,
        "mean_bp": mean * 10_000.0,
        "direction_agreement": float(np.mean(dm > 0)) if dm.size else None,
        "p_one_sided": onesided_greater_p(dm),
        "daily_means": [float(x) for x in dm],
    }


def _sample_kill(stats: dict[str, Any], sample: dict[str, Any]) -> str | None:
    if stats["ticker_days"] < sample["min_ticker_days"] or stats["trading_days"] < sample["min_trading_days"] or stats["tickers"] < sample["min_tickers"]:
        return "sample"
    if stats.get("top_ticker_share") is not None and stats["top_ticker_share"] > sample["max_top_ticker_share"]:
        return "concentration"
    wr = stats.get("win_rate")
    if wr is not None and wr < sample["min_win_rate"]:
        return "win_rate"
    return None


def decide(
    *,
    split: str,
    hypothesis_id: str,
    stats: dict[str, Any],
    gates: dict[str, Any],
    d2_mean: float | None = None,
    trial_index: int | None = None,
    bh_survivor: bool | None = None,
) -> ScientificDecision:
    sample = gates["sample"]
    mean = stats.get("mean")
    kw = {"hypothesis_id": hypothesis_id, "split": split, "mean_primary": mean, "trial_index": trial_index}

    if split == "D1":
        return ScientificDecision(label="ACTIVE", reason="d1_descriptive_no_significance_tuning", **kw)

    sk = _sample_kill(stats, sample)
    if sk:
        return ScientificDecision(label="KILL", reason=sk, **kw)
    if mean is None:
        return ScientificDecision(label="KILL", reason="no_mean", **kw)

    if split == "D2":
        if mean <= 0:
            return ScientificDecision(label="KILL", reason="direction_failed", **kw)
        floor = signed_floor(gates, "D2")
        if mean < floor:
            return ScientificDecision(label="KILL", reason="below_d2_floor", **kw)
        if bh_survivor is not True:
            return ScientificDecision(label="KILL", reason="bh_not_survivor", **kw)
        return ScientificDecision(label="ACTIVE", reason="d2_bh_survivor", **kw)

    if split == "D3":
        if mean <= 0:
            return ScientificDecision(label="KILL", reason="direction_failed", **kw)
        if gates["d3"].get("require_sign_match_d2") and (d2_mean is None or d2_mean <= 0):
            return ScientificDecision(label="KILL", reason="sign_mismatch_d2", **kw)
        floor = signed_floor(gates, "D3")
        if mean < floor:
            return ScientificDecision(label="VALIDATED_SUBTHRESHOLD_PHENOMENON", reason="below_d3_floor", **kw)
        return ScientificDecision(label="RESEARCH_PASS", reason="d3_signed_floor", **kw)

    return ScientificDecision(label="ACTIVE", reason="measured", **kw)


def evaluate_frozen(
    events: pl.DataFrame,
    rules: list[CandidateRule],
    *,
    split: str,
    gates: dict[str, Any],
    d2_means: dict[str, float] | None = None,
    d2_survivors: set[str] | None = None,
) -> list[dict[str, Any]]:
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule

    if split == "D3":
        if d2_survivors is None:
            raise AssertionError("D3 must not inspect dead D2 hypotheses; pass d2_survivors")
        rules = [r for r in rules if r.hypothesis_id in d2_survivors]

    q = float(gates["inference"]["bh_q"])
    rows: list[dict[str, Any]] = []
    pvals: list[float] = []
    for i, rule in enumerate(rules, start=1):
        sub = apply_rule(events, rule)
        st = day_block_stats(
            sub,
            n_boot=int(gates["inference"]["bootstrap_draws"]),
            seed=int(gates["inference"]["bootstrap_seed"]),
        )
        pvals.append(float(st.get("p_one_sided") or 1.0))
        rows.append({"rule": rule, "stats": st, "trial_index": i})

    bh_map: dict[str, bool] = {}
    if split == "D2" and rows:
        bh = benjamini_hochberg(pvals, q=q)
        for rec, b in zip(rows, bh, strict=True):
            bh_map[rec["rule"].hypothesis_id] = bool(b["bh_rejected"])
            rec["stats"]["bh_rejected"] = bool(b["bh_rejected"])
            rec["stats"]["bh_q"] = q
            rec["stats"]["p_one_sided"] = b["p"]

    out = []
    for rec in rows:
        rule: CandidateRule = rec["rule"]
        hid = rule.hypothesis_id
        dec = decide(
            split=split,
            hypothesis_id=hid,
            stats=rec["stats"],
            gates=gates,
            d2_mean=(d2_means or {}).get(hid),
            trial_index=rec["trial_index"],
            bh_survivor=bh_map.get(hid) if split == "D2" else None,
        )
        out.append(
            {
                "rule": rule.model_dump(),
                "stats": rec["stats"],
                "decision": dec.model_dump(),
                "trial_count": trial_count({"hypotheses": [{"hypothesis_id": r.hypothesis_id} for r in rules]}) if split != "D2" else len(rows),
            }
        )
    return out


def d2_survivor_ids(d2_rows: list[dict[str, Any]]) -> set[str]:
    out = set()
    for row in d2_rows:
        dec = row.get("decision") or {}
        if dec.get("label") == "ACTIVE" and dec.get("reason") == "d2_bh_survivor":
            out.add(row["rule"]["hypothesis_id"])
    return out
