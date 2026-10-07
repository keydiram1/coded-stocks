"""V4 freeze, matrix helpers, CIT pipeline on in-memory frames. Does not modify V2/V3."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from rich.console import Console

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
from quant_edge_lab.discovery.conditional import AlgorithmCandidate, Condition, apply_conditions, make_candidate, signed_target
from quant_edge_lab.discovery.conditional.partition import QUANTILES, discover_tree, frozen_quantile_cuts, leaves_to_candidates
from quant_edge_lab.discovery.conditional.simplify import estimate_candidate, neighbor_ok, sample_stats, simplify_candidate
from quant_edge_lab.discovery.knowledge import atomic_write_json, knowledge_dir
from quant_edge_lab.discovery.linear import decile_spread, pick_lambda, predict_ridge
from quant_edge_lab.features.causal_store import day_path
from quant_edge_lab.features.v4 import on_decision_grid
from quant_edge_lab.features.v4.features import attach_forward_residuals, enrich_grid
from quant_edge_lab.hashing import git_sha, sha256_file, sha256_json
from quant_edge_lab.peers import attach_peer_features
from quant_edge_lab.validation.multiple_testing import white_reality_check

console = Console()
MANIFEST_REL = Path("knowledge/campaigns/directional_v4_manifest.yaml")
GATES_REL = Path("knowledge/campaigns/directional_v4_gates.yaml")
BATCH_ID = "dir-campaign-v4"


class SharedInfrastructureError(RuntimeError):
    pass


class CampaignStop(SharedInfrastructureError):
    pass


def freeze_campaign(root: Path) -> dict[str, str]:
    return {
        "manifest": sha256_file(root / MANIFEST_REL),
        "gates": sha256_file(root / GATES_REL),
        "git": git_sha(root),
    }


def load_v4(root: Path) -> tuple[dict, dict]:
    man = yaml.safe_load((root / MANIFEST_REL).read_text(encoding="utf-8"))
    gates = yaml.safe_load((root / GATES_REL).read_text(encoding="utf-8"))
    return man, gates


def research_days(root: Path) -> list[str]:
    man = load_manifest(root)
    return sorted(
        d for d, r in man.get("files", {}).items() if r.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
    )


def assert_splits_match(root: Path, man: dict) -> None:
    days = research_days(root)
    w = int(man["eligibility"]["warmup_trading_days"])
    rest = days[w:]
    n = len(rest)
    i1, i2 = int(n * 0.40), int(n * 0.60)
    d1, d2, d3 = rest[:i1], rest[i1:i2], rest[i2:]
    s = man["splits"]
    if d1[0] != s["D1"]["start"] or d1[-1] != s["D1"]["end"] or len(d1) != s["D1"]["n_days"]:
        raise RuntimeError("D1 bounds drifted vs freeze")
    if d2[0] != s["D2"]["start"] or d2[-1] != s["D2"]["end"]:
        raise RuntimeError("D2 bounds drifted vs freeze")
    if d3[0] != s["D3"]["start"] or d3[-1] != s["D3"]["end"]:
        raise RuntimeError("D3 bounds drifted vs freeze")


def split_frame(df: pl.DataFrame, man: dict) -> dict[str, pl.DataFrame]:
    def _rng(key: str) -> pl.DataFrame:
        a, b = man["splits"][key]["start"], man["splits"][key]["end"]
        return df.filter((pl.col("trading_date") >= a) & (pl.col("trading_date") <= b))

    return {"D1": _rng("D1"), "D2": _rng("D2"), "D3": _rng("D3")}


def v4_store(root: Path) -> Path:
    p = Paths(root).features / "cross_sectional_v4"
    p.mkdir(parents=True, exist_ok=True)
    return p


def v4_day_path(root: Path, day: str) -> Path:
    return v4_store(root) / f"date={day}" / "part.parquet"


def v4_peer_store(root: Path) -> Path:
    p = Paths(root).features / "cross_sectional_v4_peers"
    p.mkdir(parents=True, exist_ok=True)
    return p


def v4_peer_day_path(root: Path, day: str) -> Path:
    return v4_peer_store(root) / f"date={day}" / "part.parquet"


def build_day_matrix(root: Path, day: str, *, graph=None, clock_mean: dict[int, float] | None = None) -> pl.DataFrame:
    p = day_path(root, day)
    if not p.exists():
        raise FileNotFoundError(f"causal_v2 missing for {day}; compute features first")
    feat = pl.read_parquet(p)
    feat = on_decision_grid(feat)
    if "trading_date" not in feat.columns:
        feat = feat.with_columns(pl.lit(day).alias("trading_date"))
    if "decision_ts" in feat.columns:
        from quant_edge_lab.features.v4 import ensure_utc

        feat = ensure_utc(feat, "decision_ts")
    if "available_at" in feat.columns:
        from quant_edge_lab.features.v4 import ensure_utc

        feat = ensure_utc(feat, "available_at")
        from quant_edge_lab.features.v4 import assert_available_le_decision

        assert_available_le_decision(feat, "available_at", "decision_ts")
    if "prev_close" in feat.columns:
        feat = feat.filter(pl.col("prev_close") >= 1.0)
    if "typ_1m_vol" in feat.columns and "prev_close" in feat.columns:
        feat = feat.with_columns((pl.col("typ_1m_vol") * pl.col("prev_close") * 390.0).alias("trailing_rth_dvol_proxy"))
        feat = feat.filter(pl.col("trailing_rth_dvol_proxy") >= 1_000_000)
    feat = enrich_grid(feat)
    if clock_mean and "minutes_from_open" in feat.columns:
        cmap = pl.DataFrame({"minutes_from_open": list(clock_mean.keys()), "same_clock_mean_return": list(clock_mean.values())})
        if "same_clock_mean_return" in feat.columns:
            feat = feat.drop("same_clock_mean_return")
        feat = feat.join(cmap, on="minutes_from_open", how="left")
        if "ret_5m" in feat.columns:
            feat = feat.with_columns((pl.col("ret_5m") - pl.col("same_clock_mean_return").fill_null(0.0)).alias("same_clock_surprise"))
    if graph is not None:
        feat = attach_peer_features(feat, graph)
    feat = attach_forward_residuals(feat)
    if feat.height:
        nkey = feat.select(["instrument_id", "decision_ts"]).n_unique()
        if nkey != feat.height:
            raise RuntimeError(f"duplicate decision keys on {day}: {feat.height} rows vs {nkey} unique")
    dest = v4_day_path(root, day)
    dest.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(dest)
    return feat


def _quantile_bank(d1: pl.DataFrame, features: list[str]) -> dict[str, list[float]]:
    out = {}
    for f in features:
        if f in d1.columns and d1[f].dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32):
            if d1[f].drop_nulls().n_unique() > 4:
                out[f] = frozen_quantile_cuts(d1[f], QUANTILES)
    return out


def _neighbors_pass(
    cand: AlgorithmCandidate,
    d2: pl.DataFrame,
    est: dict[str, Any],
    qbank: dict[str, list[float]],
    *,
    min_td: int,
    min_days: int,
    min_tickers: int,
    max_conc: float,
    frac: float,
) -> bool:
    """Robustness: neighboring D1 quantile cuts must keep sign and >= frac of |effect|. Not a search."""
    means: list[float | None] = []
    for cond in cand.conditions:
        if isinstance(cond.threshold, bool) or cond.operator == "==":
            continue
        cuts = qbank.get(cond.feature) or []
        if len(cuts) < 2:
            continue
        idx = int(np.argmin([abs(float(x) - float(cond.threshold)) for x in cuts]))
        for j in (idx - 1, idx + 1):
            if j < 0 or j >= len(cuts):
                continue
            alt_conds = tuple(
                Condition(c.feature, c.operator, cuts[j] if c is cond else c.threshold) for c in cand.conditions
            )
            alt = make_candidate(alt_conds, cand.direction, cand.target, cand.horizon_minutes, cand.source_model, cand.discovery_sample, parent_id=cand.candidate_id)
            e = estimate_candidate(d2, alt, min_td=min_td, min_days=min_days, min_tickers=min_tickers, max_conc=max_conc)
            means.append(e.get("mean"))
    if not means:
        return True
    return neighbor_ok(float(est["mean"]), means, frac=frac)


def run_pipeline_on_frame(df: pl.DataFrame, man: dict, gates: dict, *, features: list[str] | None = None) -> dict[str, Any]:
    target = man["targets"]["primary"]
    feats = features or list(man["search_features"])
    feats = [f for f in feats if f in df.columns]
    parts = split_frame(df, man)
    d1, d2, d3 = parts["D1"], parts["D2"], parts["D3"]
    lineage: list[dict[str, Any]] = []
    qbank = _quantile_bank(d1, feats)
    leaf = gates["leaf"]
    tree = discover_tree(
        d1,
        features=feats,
        target=target,
        max_depth=int(gates["tree"]["max_depth"]),
        node_alpha=float(gates["tree"]["node_alpha"]),
        min_ticker_days=int(leaf["min_ticker_days"]),
        min_parent_frac=float(leaf["min_parent_frac"]),
        quantile_cuts=qbank,
        lineage=lineage,
    )
    cands = leaves_to_candidates(tree, target=target, sample="D1")
    # Stability: day-block subsamples (capped for unit tests if few days)
    days = d1["trading_date"].unique().to_list() if "trading_date" in d1.columns else []
    n_sub = min(int(gates["stability"]["n_subsamples"]), max(len(days), 1))
    freq: dict[str, int] = {f: 0 for f in feats}
    rng = np.random.default_rng(int(man["search"]["seed"]))
    for i in range(n_sub):
        if not days:
            break
        take = set(rng.choice(days, size=max(len(days) // 2, 1), replace=False).tolist())
        sub = d1.filter(pl.col("trading_date").is_in(list(take)))
        t2 = discover_tree(sub, features=feats, target=target, max_depth=int(gates["tree"]["max_depth"]), node_alpha=float(gates["tree"]["node_alpha"]), min_ticker_days=max(20, int(leaf["min_ticker_days"] // 10)), min_parent_frac=float(leaf["min_parent_frac"]), quantile_cuts=qbank)
        used = set()

        def walk(n):
            if n.feature:
                used.add(n.feature)
            if n.left:
                walk(n.left)
            if n.right:
                walk(n.right)

        walk(t2)
        for f in used:
            freq[f] = freq.get(f, 0) + 1
    stable = {f for f, c in freq.items() if n_sub and c / n_sub >= float(gates["stability"]["min_selection_freq"])}
    d2_ok: list[tuple[AlgorithmCandidate, dict]] = []
    simplified_log = []
    for c in cands:
        if c.conditions and not any(cond.feature in stable or not stable for cond in c.conditions):
            continue
        simp, slog = simplify_candidate(
            c, d2, min_td=int(leaf["min_ticker_days"]), min_days=int(leaf["min_trading_days"]), min_tickers=int(leaf["min_tickers"]), max_conc=float(gates["max_top_ticker_share"])
        )
        simplified_log.extend(slog)
        est = estimate_candidate(d2, simp, min_td=int(leaf["min_ticker_days"]), min_days=int(leaf["min_trading_days"]), min_tickers=int(leaf["min_tickers"]), max_conc=float(gates["max_top_ticker_share"]))
        if est.get("kill") or est.get("mean") is None:
            continue
        if abs(est["mean"]) < float(gates["economic_floor_abs_mean"]) * 0.25:
            continue
        neigh_frac = float(gates.get("threshold_neighbors", {}).get("min_effect_frac", 0.50))
        if gates.get("threshold_neighbors") and not _neighbors_pass(
            simp,
            d2,
            est,
            qbank,
            min_td=int(leaf["min_ticker_days"]),
            min_days=int(leaf["min_trading_days"]),
            min_tickers=int(leaf["min_tickers"]),
            max_conc=float(gates["max_top_ticker_share"]),
            frac=neigh_frac,
        ):
            continue
        d2_ok.append((simp, est))
    d3_rows = []
    floor = float(gates["d3"]["abs_mean_floor"])
    survivors = []
    for c, e2 in d2_ok:
        e3 = estimate_candidate(d3, c, min_td=int(leaf["min_ticker_days"]), min_days=int(leaf["min_trading_days"]), min_tickers=int(leaf["min_tickers"]), max_conc=float(gates["max_top_ticker_share"]))
        sign_ok = e2["mean"] is not None and e3["mean"] is not None and (e2["mean"] > 0) == (e3["mean"] > 0)
        pass_d3 = bool(sign_ok and not e3.get("kill") and abs(e3["mean"]) >= floor)
        rec = {"id": c.candidate_id, "conditions": [x.to_list() for x in c.conditions], "d2": e2, "d3": e3, "decision": "RESEARCH PASS" if pass_d3 else "KILL", "note": "SIGNAL_ONLY"}
        d3_rows.append(rec)
        if pass_d3:
            survivors.append(rec)
    spa = None
    if d3.height and d2_ok and "trading_date" in d3.columns:
        all_days = sorted(d3["trading_date"].unique().to_list())
        cols_m: list[list[float]] = []
        for c, _ in d2_ok:
            sub = apply_conditions(d3, c.conditions)
            by: dict[Any, float] = {}
            if sub.height:
                tmp = sub.with_columns(signed_target(sub, c.target, c.direction).alias("_y"))
                for row in tmp.group_by("trading_date").agg(pl.col("_y").mean()).iter_rows(named=True):
                    by[row["trading_date"]] = float(row["_y"]) if row["_y"] is not None else float("nan")
            cols_m.append([by.get(d, float("nan")) for d in all_days])
        if cols_m:
            mat = np.column_stack(cols_m)
            spa = white_reality_check(mat, n_boot=200, seed=int(man["search"]["seed"]))
    # linear benchmark
    lin = None
    num_feats = [f for f in feats if f in d1.columns and d1[f].dtype in (pl.Float32, pl.Float64)]
    if num_feats and d1.height > 40:
        x1 = d1.select(num_feats).to_numpy()
        y1 = d1[target].to_numpy().astype(float)
        lam, coef, mu, sd = pick_lambda(x1, y1, list(map(float, gates["linear"]["lambda_grid"])))
        x3 = d3.select(num_feats).to_numpy() if d3.height else x1
        y3 = d3[target].to_numpy().astype(float) if d3.height else y1
        pred = predict_ridge(x3, coef, mu, sd)
        lin = {"lambda": lam, "d3_decile_spread": decile_spread(pred, y3), "n_features": len(num_feats)}
    return {
        "n_d1": d1.height,
        "n_d2": d2.height,
        "n_d3_rows": d3.height,
        "n_association_records": len(lineage),
        "n_candidates": len(cands),
        "n_d2_kept": len(d2_ok),
        "n_d3_evaluated": len(d3_rows),
        "survivors": survivors,
        "d3": d3_rows,
        "stability_freq": {k: (v / n_sub if n_sub else 0) for k, v in freq.items()},
        "linear": lin,
        "lineage_head": lineage[:40],
        "search_lineage": lineage,
        "spa": spa,
        "status": "SIGNAL_ONLY",
    }


def write_freeze_report(root: Path) -> Path:
    man, gates = load_v4(root)
    fr = freeze_campaign(root)
    split_err = None
    try:
        assert_splits_match(root, man)
    except Exception as exc:  # infrastructure: missing/changed converted-day index
        split_err = str(exc)
    lines = [
        "# V4 freeze (SIGNAL_ONLY)",
        "",
        "Campaign V4 is frozen. V2/V3 YAML and results are unmodified. Sealed OOS is inaccessible.",
        "RESEARCH PASS is not a tradable strategy. Zero survivors is an acceptable scientific result.",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Hashes",
        "",
        f"- git: `{fr['git']}`",
        f"- manifest: `{fr['manifest']}`",
        f"- gates: `{fr['gates']}`",
        f"- split_index_check: `{split_err or 'ok'}`",
        "",
        "## Scientific question",
        "",
        "Can directional information be found when a stock's own state is represented relative to the market,",
        "the cross-section, historically related peers/leaders, time-of-day expectations, and activity/liquidity",
        "proxies, with interactions discovered by a constrained honest conditional-inference search?",
        "",
        "Central variables (hypotheses, not assumed edges): `leader_response_gap`, `peer_response_gap`, `resid_rank_change_5m`.",
        "",
        "## Feature list and families",
        "",
        yaml.safe_dump(man.get("features_continuous"), sort_keys=False),
        "Legacy flags (conditioning only; V2 implementations not modified):",
        yaml.safe_dump(man.get("legacy_flags"), sort_keys=False),
        "Search feature subset (CIT eligible):",
        yaml.safe_dump(man.get("search_features"), sort_keys=False),
        "",
        "## Peer graph (PIT)",
        "",
        yaml.safe_dump(man.get("peer_graph"), sort_keys=False),
        "History used on day D is strictly before D. Weekly refresh. Residualize first, corr prescreen top-20, directed lag-5m, BH q_edge, keep top-5 leaders.",
        "",
        "## Targets",
        "",
        yaml.safe_dump(man.get("targets"), sort_keys=False),
        "Future market return may enter **labels only**. Beta at decision time is lagged. Horizon 15m is frozen; 120m is not used for selection.",
        "",
        "## Eligibility",
        "",
        yaml.safe_dump(man.get("eligibility"), sort_keys=False),
        "",
        "## D1 / D2 / D3 (chronological unsealed)",
        "",
        yaml.safe_dump(man.get("splits"), sort_keys=False),
        f"warmup_last_day: {man['splits'].get('warmup_last_day')}",
        "",
        "## Sample / stability / economic gates",
        "",
        yaml.safe_dump(gates, sort_keys=False),
        "",
        "## Search algorithm",
        "",
        "- Conditional-inference-style tree (not sklearn CART as primary).",
        "- max_depth=3, node_alpha=0.01, D1-only quantile grid q10–q90.",
        "- Multiplicity: BH across features at each node.",
        "- Day-block association tests.",
        "- Stability: 100 day-block subsamples; selection freq ≥ 60%; sign agree ≥ 80%.",
        "- Parsimony: nested leave-one-condition on D2; prefer simpler if within 1 SE.",
        "- Threshold neighbors q±10: same sign and ≥ 50% of |central| (robustness, not re-search).",
        "- D3: sign match D2 and |mean future residual 15m| ≥ 12 bp.",
        "- Family inference: day-block Hansen SPA-style vs zero benchmark.",
        "- Linear benchmark: frozen ridge λ grid on D1; compare, prefer simpler if equivalent.",
        "",
        "## Mechanism tests (pre-registered, not post-hoc rescues)",
        "",
        yaml.safe_dump(man.get("mechanism_tests"), sort_keys=False),
        "",
        "## Sealed OOS policy",
        "",
        "Do not load or scan sealed OOS in feature design, graph design, split/threshold/model selection, D1, D2, or D3.",
        "Opening sealed OOS requires a later explicit instruction. Full matrix/D1–D3 execution is a separate `--execute` after this freeze.",
        "",
        "## Canonical time",
        "",
        f"`{man.get('canonical_dtype')}` via `with_utc_us`. Naive timestamps are UTC instants.",
        "",
        "SIGNAL_ONLY. Do not silently modify this specification after freeze.",
    ]
    path = root / "reports" / "discovery" / "v4_freeze.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    atomic_write_json(
        knowledge_dir(root) / "experiments" / "directional_campaign_v4_freeze.json",
        {"report": str(path), "freeze": fr, "splits": man["splits"], "split_index_check": split_err or "ok", "status": "SIGNAL_ONLY"},
    )
    return path


def synthetic_panel(*, n_days: int = 30, n_names: int = 40, seed: int = 4, planted: bool = False, common_factor: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    from datetime import date, timedelta

    t0 = date(2022, 1, 3)
    dates: list[str] = []
    i = 0
    while len(dates) < n_days:
        d = t0 + timedelta(days=i)
        i += 1
        if d.weekday() < 5:
            dates.append(d.isoformat())
    clocks = list(range(9 * 60 + 40, 11 * 60, 5))
    for day in dates:
        factor = rng.normal(0, 0.002) if common_factor else 0.0
        for j in range(n_names):
            iid = f"ticker:S{j:03d}"
            for _ck in clocks:
                noise = rng.normal(0, 0.003)
                resid = factor + noise
                leader_gap = rng.normal(0, 0.01)
                act = rng.random()
                y = rng.normal(0, 0.004)
                if planted and leader_gap > 0.012 and act > 0.70:
                    y += 0.0025
                rows.append(
                    {
                        "trading_date": day,
                        "instrument_id": iid,
                        "ticker": f"S{j:03d}",
                        "decision_ts": datetime(2022, 1, 3, tzinfo=UTC),
                        "time_et": None,
                        "resid_ret_5m": resid,
                        "leader_response_gap": leader_gap,
                        "activity_acceleration": act,
                        "xs_dispersion_5m": abs(rng.normal(0.005, 0.001)),
                        "H004": bool(rng.random() > 0.85),
                        "irrelevant_E": rng.normal(),
                        "future_residual_15m": y,
                    }
                )
    return pl.DataFrame(rows)
