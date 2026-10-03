"""Execute frozen V4. Does not modify V2/V3 YAML or sealed OOS. SIGNAL_ONLY."""

from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import local_parquet_path
from quant_edge_lab.discovery.campaign_v2_runner import _ingest_hist
from quant_edge_lab.discovery.campaign_v4 import (
    BATCH_ID,
    GATES_REL,
    MANIFEST_REL,
    SharedInfrastructureError,
    build_day_matrix,
    freeze_campaign,
    load_v4,
    research_days,
    run_pipeline_on_frame,
    split_frame,
    v4_day_path,
    v4_store,
)
from quant_edge_lab.discovery.conditional.association import spearman_dayblock
from quant_edge_lab.discovery.eval_batch10 import med_from_hist, typ_from_hist
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json, knowledge_dir
from quant_edge_lab.discovery.runner import _batch_dir, load_state, save_state
from quant_edge_lab.features.causal_store import compute_day_features, day_path
from quant_edge_lab.hashing import git_sha, sha256_file, sha256_json
from quant_edge_lab.peers import PeerGraph, PeerGraphBuilder, attach_peer_features
from quant_edge_lab.universe.filters import add_session_columns

console = Console()

EXPECTED_MANIFEST = "ee59be114e76f3ecd78bd3ad2590e60c94f0754a42096818d9fbf7ee8a7b4a3e"
EXPECTED_GATES = "02ada62546ec70681c7d884f4cb820923e6d1417e21a6d31586fc8c955b8ac38"

SEARCH_COLS = [
    "trading_date",
    "instrument_id",
    "ticker",
    "decision_ts",
    "leader_response_gap",
    "resid_rank_change_5m",
    "resid_ret_5m",
    "xs_dispersion_5m",
    "xs_breadth_5m",
    "rvol_5m_tod",
    "activity_acceleration",
    "volume_price_disagreement",
    "same_clock_surprise",
    "H004",
    "impulse",
    "full_state",
    "drop_H004",
    "drop_impulse",
    "future_residual_15m",
    "future_residual_5m",
    "future_residual_30m",
    "future_residual_rank_15m",
    "future_raw_15m",
    "leader_shock_5m",
    "peer_response_gap",
    "peer_rank_gap",
]


class CampaignStop(SharedInfrastructureError):
    pass


def align_v4_schema(feat: pl.DataFrame, schema: list[str] | None) -> tuple[pl.DataFrame, list[str]]:
    """Drop later causal_v2 passthrough extras (e.g. V3 resid_5m). Error if frozen cols missing."""
    if schema is None:
        return feat, list(feat.columns)
    missing = [c for c in schema if c not in feat.columns]
    if missing:
        raise CampaignStop(f"missing frozen V4 columns {missing[:12]}")
    return feat.select(schema), schema


class Watchdog:
    def __init__(self) -> None:
        self.error_count = 0
        self.exc_counts: dict[str, int] = {}
        self.empty_streak = 0
        self.schema: list[str] | None = None
        self.last_exception: str | None = None

    def classify_exception(self, exc: BaseException) -> None:
        key = f"{type(exc).__name__}:{str(exc)[:240]}"
        self.exc_counts[key] = self.exc_counts.get(key, 0) + 1
        self.last_exception = key
        self.error_count += 1
        if self.exc_counts[key] >= 2:
            raise CampaignStop(f"repeated unexpected exception: {key}") from exc
        raise CampaignStop(f"terminating exception (first occurrence logged): {key}") from exc

    def note_partition(self, feat: pl.DataFrame, *, expected_nonempty: bool) -> None:
        if feat.height == 0:
            if expected_nonempty:
                self.empty_streak += 1
                if self.empty_streak >= 3:
                    raise CampaignStop("three consecutive expected-nonempty partitions were empty")
            return
        self.empty_streak = 0
        cols = list(feat.columns)
        if self.schema is None:
            self.schema = cols
        elif set(cols) != set(self.schema):
            added = set(cols) - set(self.schema)
            missing = set(self.schema) - set(cols)
            raise CampaignStop(f"output schema changed added={sorted(added)[:12]} missing={sorted(missing)[:12]}")
        if "resid_ret_5m" in feat.columns and feat["resid_ret_5m"].null_count() == feat.height:
            raise CampaignStop("resid_ret_5m entirely null")
        if "decision_ts" in feat.columns and str(feat.schema["decision_ts"]) != "Datetime(time_unit='us', time_zone='UTC')":
            raise CampaignStop(f"decision_ts dtype {feat.schema['decision_ts']} is not canonical UTC")


def verify_freeze(root: Path) -> tuple[dict, dict, dict[str, str]]:
    fr = freeze_campaign(root)
    if fr["manifest"] != EXPECTED_MANIFEST:
        raise CampaignStop(f"manifest hash drifted: {fr['manifest']}")
    if fr["gates"] != EXPECTED_GATES:
        raise CampaignStop(f"gates hash drifted: {fr['gates']}")
    man, gates = load_v4(root)
    if man["targets"]["primary"] != "future_residual_15m":
        raise CampaignStop("primary target drifted")
    if float(gates["d3"]["abs_mean_floor"]) != 0.0012:
        raise CampaignStop("D3 floor drifted")
    if int(gates["tree"]["max_depth"]) != 3 or float(gates["tree"]["node_alpha"]) != 0.01:
        raise CampaignStop("CIT hyperparameters drifted")
    q = list(gates["tree"]["quantile_grid"])
    if q != [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        raise CampaignStop("quantile grid drifted")
    if int(gates["stability"]["n_subsamples"]) != 100:
        raise CampaignStop("stability subsample count drifted")
    s = man["splits"]
    if s["D1"] != {"start": "2021-10-29", "end": "2023-10-17", "n_days": 494}:
        raise CampaignStop("D1 bounds drifted")
    if s["D2"]["start"] != "2023-10-18" or s["D3"]["end"] != "2026-10-01":
        raise CampaignStop("D2/D3 bounds drifted")
    if man.get("sealed_oos") != "inaccessible":
        raise CampaignStop("sealed OOS policy missing")
    return man, gates, fr


def assert_day_in_unsealed(day: str, man: dict) -> None:
    if "oos" in day.lower() or "sealed" in day.lower():
        raise CampaignStop(f"refusing path-like day token {day}")
    if day > man["data"]["panel_end"] or day > man["splits"]["D3"]["end"]:
        raise CampaignStop(f"attempted access beyond unsealed panel: {day}")
    if day < man["data"]["panel_start"]:
        raise CampaignStop(f"day {day} before panel_start")


def empty_hist() -> dict[str, Any]:
    return {
        "prev_close": {},
        "typ": {},
        "yday_ret": {},
        "yday_high": {},
        "yday_low": {},
        "yday_range": {},
        "yday_clv": {},
        "ret_2d": {},
        "ret_3d": {},
        "ret_5d": {},
        "ah_ret": {},
        "med20": {},
        "pm_ret": {},
        "closes": defaultdict(lambda: deque(maxlen=5)),
        "vol_hist": defaultdict(lambda: deque(maxlen=20)),
        "range_hist": defaultdict(lambda: deque(maxlen=20)),
    }


def _week_monday(day: str) -> str:
    from datetime import date, timedelta

    d = date.fromisoformat(day)
    return (d - timedelta(days=d.weekday())).isoformat()


def _health(feat: pl.DataFrame) -> dict[str, Any]:
    if feat.height == 0:
        return {"rows": 0}
    rec: dict[str, Any] = {
        "rows": feat.height,
        "tickers": int(feat["instrument_id"].n_unique()) if "instrument_id" in feat.columns else None,
        "schema_n": len(feat.columns),
    }
    for c in ("resid_ret_5m", "future_residual_15m", "leader_response_gap", "xs_dispersion_5m", "H004"):
        if c in feat.columns:
            rec[f"null_{c}"] = float(feat[c].null_count() / feat.height)
    if "resid_rank_5m" in feat.columns:
        rec["resid_rank_min"] = float(feat["resid_rank_5m"].min() or 0)
        rec["resid_rank_max"] = float(feat["resid_rank_5m"].max() or 0)
    return rec


def _progress(stage: str, done: int, total: int, t0: float, rows: int, wd: Watchdog) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed else 0
    remain = (total - done) / rate if rate else 0
    console.print(
        f"{stage} {done}/{total} days  ERROR={wd.error_count}  rows={rows}  "
        f"elapsed={elapsed/60:.1f}m  ETA={remain/60:.1f}m  health={'PASS' if wd.error_count==0 else 'ERROR'}"
    )


def rebuild_hist(root: Path, days: list[str], before: str) -> dict[str, Any]:
    hist = empty_hist()
    prior = [d for d in days if d < before][-25:]
    for d in prior:
        bp = local_parquet_path(root, d)
        if not bp.exists():
            raise CampaignStop(f"missing bars for hist rebuild {d}")
        sessioned = add_session_columns(pl.read_parquet(bp))
        _ingest_hist(sessioned, hist)
    return hist


def graph_dir(root: Path) -> Path:
    p = Paths(root).derived / "features" / "peer_graph_v4"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_resid_history(root: Path, days: list[str]) -> pl.DataFrame:
    frames = []
    for d in days:
        p = v4_day_path(root, d)
        if not p.exists():
            continue
        frames.append(
            pl.read_parquet(p, columns=["trading_date", "instrument_id", "decision_ts", "resid_ret_5m", "resid_rank_5m"])
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def apply_graph_to_day(root: Path, day: str, g: PeerGraph) -> None:
    import gc

    p = v4_day_path(root, day)
    feat = pl.read_parquet(p)
    drop = [c for c in ("leader_shock_5m", "leader_agreement", "leader_response_gap", "peer_return_5m", "peer_response_gap", "peer_rank_gap", "peer_breadth", "peer_dispersion") if c in feat.columns]
    if drop:
        feat = feat.drop(drop)
    feat = attach_peer_features(feat, g)
    tmp = p.parent / "part.new.parquet"
    feat.write_parquet(tmp)
    del feat
    gc.collect()
    for _ in range(8):
        try:
            if p.exists():
                p.unlink()
            break
        except OSError:
            gc.collect()
            time.sleep(0.25)
    tmp.replace(p)


def scan_search_days(root: Path, days: list[str]) -> pl.DataFrame:
    files = [v4_day_path(root, d) for d in days if v4_day_path(root, d).exists()]
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files)
    have = set(lf.collect_schema().names())
    cols = [c for c in SEARCH_COLS if c in have]
    return lf.select(cols).collect()


def mechanism_tests(df: pl.DataFrame) -> dict[str, Any]:
    if df.height < 200 or "future_residual_15m" not in df.columns:
        return {"status": "NOT EVALUATED", "reason": "insufficient rows"}
    y = df["future_residual_15m"].to_numpy().astype(float)
    days = df["trading_date"].to_numpy() if "trading_date" in df.columns else np.arange(df.height)
    out: dict[str, Any] = {}
    for name, col in (
        ("LL-CATCHUP", "leader_response_gap"),
        ("LL-NETWORK", "leader_shock_5m"),
        ("LL-RANK-DIFFUSION", "peer_rank_gap"),
    ):
        if col not in df.columns:
            out[name] = {"p": None, "status": "NOT EVALUATED"}
            continue
        p = spearman_dayblock(df[col].to_numpy().astype(float), y, days)
        out[name] = {"p": p, "note": "D1 association only; not a PASS", "status": "SIGNAL_ONLY"}
    if "H004" in df.columns and "leader_response_gap" in df.columns:
        inter = df["leader_response_gap"].to_numpy().astype(float) * df["H004"].cast(pl.Float64).to_numpy()
        out["LL-STATE-INTERACTION"] = {"p": spearman_dayblock(inter, y, days), "status": "SIGNAL_ONLY"}
    out["LL-SPARSE"] = {"note": "ridge linear benchmark is the frozen sparse family", "status": "see linear"}
    return out


def write_v4_report(root: Path, state: dict) -> Path:
    path = root / "reports" / "discovery" / "directional_campaign_v4.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    c = state.get("counts") or {}
    surv = state.get("survivors") or []
    lines = [
        "# Directional campaign V4 (SIGNAL_ONLY)",
        "",
        "RESEARCH PASS is not tradable. Sealed OOS was not opened. V2/V3 unmodified.",
        "",
        "## Executive summary",
        "",
        f"- RESEARCH PASS: **{len(surv)}**",
        f"- KILL: {c.get('KILL', 0)}",
        f"- ERROR: {c.get('ERROR', 0)}",
        f"- NOT EVALUATED: {c.get('NOT_EVALUATED', 0)}",
        f"- stop_reason: {state.get('stop_reason') or 'completed'}",
        "",
        "Zero survivors is a valid scientific result under frozen gates.",
        "",
        "## Freeze / execution lineage",
        "",
        f"```json\n{json.dumps(state.get('lineage'), indent=2, default=str)}\n```",
        "",
        "## Counts",
        "",
        f"```json\n{json.dumps(c, indent=2, default=str)}\n```",
        "",
        "## D1 → D3 candidates",
        "",
        f"```json\n{json.dumps(state.get('pipeline'), indent=2, default=str)[:20000]}\n```",
        "",
        "## Mechanism tests",
        "",
        f"```json\n{json.dumps(state.get('mechanisms'), indent=2, default=str)}\n```",
        "",
        "## Strongest failures / survivors",
        "",
        f"```json\n{json.dumps(surv or state.get('strongest_kill'), indent=2, default=str)[:15000]}\n```",
        "",
        "SIGNAL_ONLY. SEALED OOS NOT OPENED.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_matrix_days(
    root: Path,
    man: dict,
    days_all: list[str],
    write_days: list[str],
    state: dict,
    wd: Watchdog,
    *,
    instruments: pl.DataFrame,
    cfg_hash: str,
) -> None:
    write_set = set(write_days)
    t0 = time.time()
    hist = empty_hist()
    hist_ok_through = ""
    clock_sum: dict[int, float] = {int(k): float(v) for k, v in (state.get("clock_sum") or {}).items()}
    clock_n: dict[int, int] = {int(k): int(v) for k, v in (state.get("clock_n") or {}).items()}
    rows_total = int(state.get("matrix_rows") or 0)
    done = 0
    t_write = [d for d in days_all if d in write_set]
    if not state.get("output_schema"):
        for d0, ck0 in (state.get("days") or {}).items():
            p0 = v4_day_path(root, d0)
            if ck0.get("status") == "ok" and p0.exists():
                state["output_schema"] = list(pl.scan_parquet(p0).collect_schema().names())
                break
    if state.get("output_schema") and wd.schema is None:
        wd.schema = list(state["output_schema"])
    for day in t_write:
        assert_day_in_unsealed(day, man)
        ck = (state.get("days") or {}).get(day) or {}
        if ck.get("cfg_hash") == cfg_hash and ck.get("status") in {"ok", "ok_warmup"}:
            if ck.get("status") == "ok_warmup" or (v4_day_path(root, day).exists() and int(ck.get("rows") or 0) > 0):
                done += 1
                if done <= 25 or done % 5 == 0:
                    _progress("STAGE 1 MATRIX", done, len(t_write), t0, rows_total, wd)
                continue
        try:
            causal_p = day_path(root, day)
            idx_all = days_all.index(day)
            prev = days_all[idx_all - 1] if idx_all else ""
            warmup = day < man["splits"]["D1"]["start"]
            if not causal_p.exists():
                if hist_ok_through != prev:
                    hist = rebuild_hist(root, days_all, day)
                bars_p = local_parquet_path(root, day)
                if not bars_p.exists():
                    raise CampaignStop(f"missing normalized bars {day}")
                sessioned = add_session_columns(pl.read_parquet(bars_p))
                hist["typ"] = typ_from_hist(hist["vol_hist"])
                hist["med20"] = med_from_hist(hist["range_hist"])
                feat_c = compute_day_features(sessioned, instruments, hist)
                causal_p.parent.mkdir(parents=True, exist_ok=True)
                if feat_c.height:
                    feat_c.write_parquet(causal_p)
                _ingest_hist(sessioned, hist)
                hist_ok_through = day
            if (not causal_p.exists()) or (causal_p.exists() and pl.scan_parquet(causal_p).select(pl.len()).collect().item() == 0):
                if warmup:
                    state.setdefault("days", {})[day] = {"status": "ok_warmup", "cfg_hash": cfg_hash, "rows": 0, "health": {"rows": 0, "note": "warmup_no_prev_close"}}
                    done += 1
                    continue
                raise CampaignStop(f"empty causal_v2 for non-warmup day {day}")
            clock_mean = {k: clock_sum[k] / clock_n[k] for k in clock_n if clock_n[k]}
            feat = build_day_matrix(root, day, clock_mean=clock_mean)
            if feat.height == 0 and not warmup:
                raise CampaignStop(f"empty V4 matrix despite causal_v2 present for {day}")
            feat, schema = align_v4_schema(feat, state.get("output_schema") or wd.schema)
            state["output_schema"] = schema
            wd.schema = schema
            dest = v4_day_path(root, day)
            dest.parent.mkdir(parents=True, exist_ok=True)
            feat.write_parquet(dest)
            wd.note_partition(feat, expected_nonempty=not warmup)
            if "minutes_from_open" in feat.columns and "ret_5m" in feat.columns:
                g = feat.group_by("minutes_from_open").agg(pl.col("ret_5m").mean().alias("m"))
                for row in g.iter_rows(named=True):
                    if row["minutes_from_open"] is None or row["m"] is None:
                        continue
                    k = int(row["minutes_from_open"])
                    clock_sum[k] = float(clock_sum.get(k, 0.0) + float(row["m"]))
                    clock_n[k] = int(clock_n.get(k, 0) + 1)
            h = _health(feat)
            state.setdefault("days", {})[day] = {"status": "ok", "cfg_hash": cfg_hash, "rows": feat.height, "health": h, "schema_n": len(feat.columns)}
            rows_total += feat.height
        except CampaignStop:
            raise
        except Exception as exc:
            state.setdefault("days", {})[day] = {"status": "ERROR", "cfg_hash": cfg_hash, "error": f"{type(exc).__name__}: {exc}"}
            state.setdefault("errors", []).append({"day": day, "tb": traceback.format_exc()[-1500:]})
            wd.classify_exception(exc)
        done += 1
        if done <= 25 or done % 5 == 0:
            _progress("STAGE 1 MATRIX", done, len(t_write), t0, rows_total, wd)
        if done % 5 == 0:
            state["matrix_rows"] = rows_total
            state["clock_sum"] = clock_sum
            state["clock_n"] = clock_n
            save_state(root, state)
    state["matrix_rows"] = rows_total
    state["clock_sum"] = clock_sum
    state["clock_n"] = clock_n
    save_state(root, state)


def run_peer_graphs(root: Path, man: dict, days: list[str], state: dict, wd: Watchdog, cfg_hash: str) -> None:
    t0 = time.time()
    pg = man["peer_graph"]
    builder = PeerGraphBuilder(
        history_days=int(pg["history_trading_days"]),
        structural_top=int(pg["structural_top"]),
        leaders_top=int(pg["leaders_top"]),
        lag_minutes=int(pg["lag_minutes"]),
        q_edge=float(pg["q_edge"]),
    )
    weeks = sorted({_week_monday(d) for d in days if d >= man["splits"]["D1"]["start"]})
    gdir = graph_dir(root)
    n_edges = 0
    for i, as_of in enumerate(weeks):
        assert_day_in_unsealed(as_of, man)
        gp = gdir / f"as_of={as_of}" / "graph.json"
        if gp.exists() and (state.get("graphs") or {}).get(as_of, {}).get("cfg_hash") == cfg_hash:
            continue
        hist_days = [d for d in days if d < as_of][-int(pg["history_trading_days"]) :]
        hist = load_resid_history(root, hist_days)
        if hist.height:
            mx = hist["trading_date"].max()
            if str(mx) >= as_of:
                raise CampaignStop(f"graph {as_of} history_end {mx} not strictly before as_of")
        hist = hist.with_columns(pl.col("decision_ts").cast(pl.Utf8).alias("clock"))
        g = builder.fit(hist, as_of=as_of)
        if g.history_end and g.history_end >= as_of:
            raise CampaignStop(f"PIT fail graph history_end={g.history_end} as_of={as_of}")
        rec = {
            "as_of": as_of,
            "history_start": g.history_start,
            "history_end": g.history_end,
            "n_edges": len(g.edges),
            "lineage": g.lineage,
            "cfg_hash": cfg_hash,
            "edges": [
                {
                    "leader_id": e.leader_id,
                    "follower_id": e.follower_id,
                    "weight": e.weight,
                    "lag_minutes": e.lag_minutes,
                    "p_value": e.p_value,
                    "q_value": e.q_value,
                }
                for e in g.edges
            ],
        }
        gp.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(gp, rec)
        use_days = [d for d in days if _week_monday(d) == as_of and v4_day_path(root, d).exists()]
        for d in use_days:
            apply_graph_to_day(root, d, g)
        n_edges += len(g.edges)
        state.setdefault("graphs", {})[as_of] = {"n_edges": len(g.edges), "n_tests": g.lineage.get("n_tests"), "cfg_hash": cfg_hash, "history_end": g.history_end}
        if i % 1 == 0:
            _progress("STAGE 2 PEER GRAPH", i + 1, len(weeks), t0, n_edges, wd)
        if i % 3 == 0:
            save_state(root, state)
    state["n_graph_versions"] = len(weeks)
    save_state(root, state)


def run_campaign_v4(root: Path) -> dict[str, Any]:
    man, gates, fr = verify_freeze(root)
    cfg_hash = sha256_json({"manifest": fr["manifest"], "gates": fr["gates"]})
    days = [d for d in research_days(root) if d <= man["splits"]["D3"]["end"]]
    if not days:
        raise CampaignStop("no converted research days")
    inst_path = Paths(root).normalized / "massive-flat" / "instruments.parquet"
    if not inst_path.exists():
        raise CampaignStop("instruments.parquet missing")
    instruments = pl.read_parquet(inst_path)

    prev = load_state(root, BATCH_ID)
    if prev and prev.get("cfg_hash") not in {None, cfg_hash}:
        raise CampaignStop("checkpoint cfg_hash incompatible with frozen configuration")
    state = prev or {
        "batch_id": BATCH_ID,
        "cfg_hash": cfg_hash,
        "days": {},
        "graphs": {},
        "errors": [],
        "stage": "start",
        "oos_opened": False,
    }
    state["cfg_hash"] = cfg_hash
    state["lineage"] = {
        "execution_git": git_sha(root),
        "manifest": fr["manifest"],
        "gates": fr["gates"],
        "cfg_hash": cfg_hash,
        "feature_schema_hash": sha256_json(SEARCH_COLS),
        "working_tree": git_sha(root),
        "status": "SIGNAL_ONLY",
        "sealed_oos": "not_opened",
    }
    wd = Watchdog()
    save_state(root, state)

    d1s, d1e = man["splits"]["D1"]["start"], man["splits"]["D1"]["end"]
    d1_days = [d for d in days if d1s <= d <= d1e]
    if not d1_days:
        raise CampaignStop("D1 empty")

    phases = [
        ("A", d1_days[:1]),
        ("B", d1_days[:3]),
        ("C", d1_days[:5]),
        ("D", d1_days[:21]),
        ("FULL", days),
    ]
    t_all = time.time()
    try:
        for name, subset in phases:
            state["stage"] = f"matrix_{name}"
            console.print(f"PHASE {name} matrix days={len(subset)}")
            t1 = time.time()
            run_matrix_days(root, man, days, subset, state, wd, instruments=instruments, cfg_hash=cfg_hash)
            state.setdefault("perf", {})[name] = {"seconds": time.time() - t1, "days": len(subset), "rows": state.get("matrix_rows")}
            save_state(root, state)
            if name in {"A", "B"} and subset:
                sample = pl.read_parquet(v4_day_path(root, subset[0]))
                inspect = sample.sample(n=min(20, sample.height), seed=4) if sample.height else sample
                keep = [c for c in inspect.columns if not str(c).startswith("future_")]
                atomic_write_json(_batch_dir(root, BATCH_ID) / f"inspect_{name}.json", {"day": subset[0], "rows": inspect.select(keep).to_dicts()[:20]})
        state["stage"] = "graphs"
        run_peer_graphs(root, man, days, state, wd, cfg_hash)

        d2_days = [d for d in days if man["splits"]["D2"]["start"] <= d <= man["splits"]["D2"]["end"]]
        d3_days = [d for d in days if man["splits"]["D3"]["start"] <= d <= man["splits"]["D3"]["end"]]
        state["stage"] = "D1"
        console.print("D1 loading search frame (SIGNAL_ONLY)")
        d1 = scan_search_days(root, d1_days)
        d2 = scan_search_days(root, d2_days)
        d3 = scan_search_days(root, d3_days)
        frame = pl.concat([d1, d2, d3], how="diagonal_relaxed") if d1.height else pl.DataFrame()
        state["counts"] = {
            "matrix_rows": int(state.get("matrix_rows") or 0),
            "d1_rows": d1.height,
            "d2_rows": d2.height,
            "d3_rows": d3.height,
            "graph_versions": int(state.get("n_graph_versions") or 0),
            "graph_edges": int(sum((g or {}).get("n_edges") or 0 for g in (state.get("graphs") or {}).values())),
        }
        if d1.height == 0:
            state["pipeline"] = {"n_candidates": 0, "survivors": [], "note": "zero D1 rows"}
            state["mechanisms"] = {"status": "NOT EVALUATED"}
            state["counts"]["NOT_EVALUATED"] = 1
            state["counts"]["KILL"] = 0
            state["counts"]["ERROR"] = wd.error_count
            state["counts"]["RESEARCH_PASS"] = 0
        else:
            state["mechanisms"] = mechanism_tests(d1)
            pipe = run_pipeline_on_frame(frame, man, gates, features=list(man["search_features"]))
            state["pipeline"] = {k: v for k, v in pipe.items() if k != "search_lineage"}
            lineage_path = _batch_dir(root, BATCH_ID) / "search_lineage.jsonl"
            if lineage_path.exists():
                lineage_path.unlink()
            for rec in pipe.get("search_lineage") or []:
                append_jsonl(lineage_path, rec)
            surv = pipe.get("survivors") or []
            d3_eval = pipe.get("d3") or []
            state["survivors"] = surv
            kills = [r for r in d3_eval if r.get("decision") == "KILL"]
            state["strongest_kill"] = sorted(kills, key=lambda r: abs((r.get("d3") or {}).get("mean") or 0), reverse=True)[:8]
            state["counts"].update(
                {
                    "D1_search_tests": pipe.get("n_association_records"),
                    "candidates_generated": pipe.get("n_candidates"),
                    "D2_candidates": pipe.get("n_d2_kept"),
                    "D3_candidates": pipe.get("n_d3_evaluated"),
                    "RESEARCH_PASS": len(surv),
                    "KILL": len(kills),
                    "ERROR": wd.error_count,
                    "NOT_EVALUATED": 0,
                }
            )
        state["stage"] = "report"
        state["elapsed_sec"] = time.time() - t_all
        report = write_v4_report(root, state)
        state["report"] = str(report)
        atomic_write_json(knowledge_dir(root) / "experiments" / "directional_campaign_v4_summary.json", state)
        save_state(root, state)
        console.print(f"report={report} RESEARCH_PASS={state['counts'].get('RESEARCH_PASS')} SIGNAL_ONLY sealed OOS not opened")
        return state
    except CampaignStop as exc:
        state["stage"] = f"STOPPED:{state.get('stage')}"
        state["stop_reason"] = str(exc)
        state["counts"] = dict(state.get("counts") or {})
        state["counts"]["ERROR"] = wd.error_count or 1
        report = write_v4_report(root, state)
        state["report"] = str(report)
        save_state(root, state)
        console.print(f"STOPPED stage={state.get('stage')} {exc}")
        raise
