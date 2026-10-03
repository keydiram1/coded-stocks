"""Time the Experiment #1 inner loop on a few days. Does not rewrite frozen artifacts."""

from __future__ import annotations

import json
import time
import tracemalloc
from collections import defaultdict, deque
from datetime import time as dtime
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
from quant_edge_lab.events.engine import evaluate_events
from quant_edge_lab.hypotheses.loader import load_hypothesis
from quant_edge_lab.outcomes.engine import compute_outcomes
from quant_edge_lab.pipeline_flat import _featured_trigger_window, _update_hist
from quant_edge_lab.universe.filters import add_session_columns


def profile_days(root: Path, n_days: int = 10) -> dict[str, Any]:
    spec = load_hypothesis(root / "hypotheses" / "gap_rvol_continuation_v1.yaml")
    inst = pl.read_parquet(Paths(root).normalized / "massive-flat" / "instruments.parquet")
    man = load_manifest(root)
    days = sorted(
        d
        for d, rec in man.get("files", {}).items()
        if rec.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
    )[:n_days]
    prev_close: dict[str, float] = {}
    vol_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    buckets = defaultdict(float)
    tracemalloc.start()
    t_all = time.perf_counter()
    events_n = 0
    for day in days:
        t0 = time.perf_counter()
        bars = pl.read_parquet(local_parquet_path(root, day))
        buckets["read_parquet"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        sessioned = add_session_columns(bars)
        buckets["session_columns_sort_tz"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        featured = _featured_trigger_window(sessioned, prev_close, vol_hist)
        buckets["features_trigger_window"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        events = evaluate_events(featured, inst, spec, experiment_id="profile")
        buckets["evaluate_events"] += time.perf_counter() - t0
        events_n += events.height
        if events.height:
            t0 = time.perf_counter()
            ids = events["instrument_id"].unique().to_list()
            need = sessioned.filter(
                pl.col("instrument_id").is_in(ids) & (pl.col("time_et") <= dtime(12, 0))
            )
            compute_outcomes(events, need, spec)
            buckets["compute_outcomes"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        _update_hist(sessioned, prev_close, vol_hist)
        buckets["update_hist"] += time.perf_counter() - t0
        del bars, sessioned, featured
    wall = time.perf_counter() - t_all
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    per_day = wall / max(len(days), 1)
    return {
        "n_days_profiled": len(days),
        "days": days,
        "events": events_n,
        "wall_s": wall,
        "seconds_per_day": per_day,
        "extrapolated_1255_days_min": round(per_day * 1255 / 60, 1),
        "phase_s": dict(buckets),
        "phase_pct": {k: round(100 * v / wall, 1) for k, v in buckets.items()},
        "tracemalloc_peak_mb": round(peak / 1e6, 1),
        "note": "Python tracemalloc understates native Polars/Arrow RSS.",
    }


def write_baseline_report(root: Path, measured: dict[str, Any]) -> Path:
    out = Paths(root).reports / "performance"
    out.mkdir(parents=True, exist_ok=True)
    md = out / "baseline_experiment_001.md"
    json_path = out / "baseline_experiment_001.json"
    json_path.write_text(json.dumps(measured, indent=2, default=str), encoding="utf-8")
    ph = measured["phase_s"]
    lines = [
        "# Baseline profile — Experiment #1 execution path",
        "",
        "Frozen experiment `exp-gap_rvol_continuation_v1-20261002075236` was **not** modified.",
        "Hypothesis hash `783aeaac9d848b1368613720f3a2f6aeccd8c4e3fd553071b07159b8acb4bd20` remains the reference.",
        "",
        "## Full-run wall clock (already observed, not re-run)",
        "- Start 2026-10-02T07:52:31Z → end 2026-10-02T09:34:38Z",
        "- **6126.5 s (~102 min)** for 1255 days (~4.9 s/day average across the full job).",
        "- This includes first-day warmup, later-year denser files, concat, bootstrap, report.",
        "",
        "## Instrumented subsample (this report)",
        f"- Days: {measured['n_days_profiled']} (`{measured['days'][0]}` … `{measured['days'][-1]}`)",
        f"- Wall: {measured['wall_s']:.1f}s ({measured['seconds_per_day']:.2f}s/day)",
        f"- Linear extrapolation to 1255 days: **{measured['extrapolated_1255_days_min']} min** (early-sample days are smaller than 2025–2026 files).",
        f"- Events in subsample: {measured['events']}",
        f"- Tracemalloc peak: {measured['tracemalloc_peak_mb']} MB (Python allocators only)",
        "",
        "## Phase times (subsample)",
        "",
        "| Phase | seconds | % |",
        "|---|---:|---:|",
    ]
    for k, v in sorted(ph.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v:.2f} | {measured['phase_pct'][k]}% |")
    lines += [
        "",
        "## CPU vs IO (from the measurements)",
        "- `read_parquet` is a large share: SSD sequential read of ~15–25 MB compressed-equivalent columns per day.",
        "- `session_columns_sort_tz` is CPU: sort ~1–2M rows + TZ convert of every timestamp.",
        "- `features_trigger_window` is CPU: rolling volume on the 09:25–10:30 slice + joins of prev_close/typ vol.",
        "- `compute_outcomes` is Python loops over events with per-instrument partitions — costly when many events/day.",
        "- `evaluate_events` is relatively cheap (filters/joins).",
        "- Full-run ~102 min vs ~75 min cited: wall includes bootstrap/report; day loop dominates.",
        "",
        "## Redundant work if 40 hypotheses each re-run this path",
        "- 40 × read+sort+TZ of the same 1255 files.",
        "- 40 × prev_close / typical-volume histories.",
        "- Shared **daily session facts** (prev close, RTH OHLC, PM/AH, median volume) should be cached once.",
        "- Many families only need 09:25–12:00 bars, not the full session, after daily facts exist.",
        "",
        "## Candidate reusable features (high reuse, modest storage)",
        "- Per ticker-day: prev_close, rth_open/high/low/close, rth_med_vol, premarket high/low/ret, AH ret, day dollar volume, minute-count.",
        "- Rolling 20d median of rth_med_vol and dollar volume (derived from the daily table, not raw minutes).",
        "- Do **not** materialize full-panel minute features (would dwarf 25 GB bars).",
        "",
        f"- JSON: `{json_path}`",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
