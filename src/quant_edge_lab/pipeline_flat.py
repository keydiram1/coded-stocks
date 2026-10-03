"""Incremental gap/RVOL experiment on partitioned minute parquet (no full RAM load)."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime, time
from pathlib import Path

import polars as pl

from quant_edge_lab.config import Paths, load_research_config
from quant_edge_lab.data.catalog import dataset_manifest_hash, dataset_ref
from quant_edge_lab.data.massive.flatfiles import DATASET, load_manifest, local_parquet_path
from quant_edge_lab.events.engine import evaluate_events
from quant_edge_lab.execution.discovery import EXECUTION_MODEL_VERSION
from quant_edge_lab.features.engine import FEATURE_VERSION
from quant_edge_lab.hashing import git_sha, sha256_file
from quant_edge_lab.hypotheses.loader import hypothesis_hash, load_hypothesis
from quant_edge_lab.models.schemas import ExperimentRecord
from quant_edge_lab.outcomes.engine import compute_outcomes
from quant_edge_lab.pipeline import UNIVERSE_VERSION
from quant_edge_lab.registry.store import write_experiment
from quant_edge_lab.reporting.report import render_report, write_charts
from quant_edge_lab.statistics.robustness import robustness_tables
from quant_edge_lab.statistics.summarize import summarize_outcomes
from quant_edge_lab.universe.filters import add_session_columns


def _featured_trigger_window(
    df: pl.DataFrame,
    prev_close: dict[str, float],
    vol_hist: dict[str, deque],
) -> pl.DataFrame:
    rth_open = df.filter(pl.col("is_rth")).group_by("instrument_id").agg(pl.col("open").first().alias("rth_open"))
    trig = df.filter((pl.col("time_et") >= time(9, 25)) & (pl.col("time_et") <= time(10, 30)))
    trig = trig.with_columns(
        pl.col("volume").rolling_sum(window_size=5, min_samples=1).over("instrument_id").alias("roll_vol_5m"),
    )
    if prev_close:
        pc = pl.DataFrame({"instrument_id": list(prev_close), "prev_close": [prev_close[k] for k in prev_close]})
    else:
        pc = pl.DataFrame(schema={"instrument_id": pl.String, "prev_close": pl.Float64})
    typ_rows = [
        {"instrument_id": iid, "typ_1m_vol": float(sorted(hist)[len(hist) // 2])}
        for iid, hist in vol_hist.items()
        if hist
    ]
    typ = (
        pl.DataFrame(typ_rows)
        if typ_rows
        else pl.DataFrame(schema={"instrument_id": pl.String, "typ_1m_vol": pl.Float64})
    )
    trig = trig.join(pc, on="instrument_id", how="left").join(typ, on="instrument_id", how="left")
    trig = trig.join(rth_open, on="instrument_id", how="left")
    return trig.with_columns(
        (pl.col("rth_open") / pl.col("prev_close") - 1).alias("gap_pct"),
        (pl.col("roll_vol_5m") / (pl.col("typ_1m_vol") * 5)).alias("rvol_5m"),
        (pl.col("ts_utc") + pl.duration(minutes=1)).alias("decision_ts"),
        (pl.col("ts_utc") + pl.duration(minutes=1)).alias("available_at"),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
    )


def _update_hist(df: pl.DataFrame, prev_close: dict[str, float], vol_hist: dict[str, deque]) -> None:
    rth = df.filter(pl.col("is_rth"))
    if rth.height == 0:
        return
    daily = rth.group_by("instrument_id").agg(
        pl.col("close").last().alias("rth_close"),
        pl.col("volume").median().alias("rth_med_vol"),
    )
    for row in daily.iter_rows(named=True):
        iid = row["instrument_id"]
        prev_close[iid] = float(row["rth_close"])
        vol_hist[iid].append(float(row["rth_med_vol"] or 0.0))


def run_gap_rvol_on_flat(
    hypothesis_path: Path | str,
    root: Path | None = None,
    dataset: str = DATASET,
) -> ExperimentRecord:
    import json

    root = Path(root) if root else Path.cwd()
    paths = Paths(root)
    paths.ensure()
    cfg = load_research_config(paths.config / "research.yaml")
    spec = load_hypothesis(hypothesis_path)
    h_hash = hypothesis_hash(spec)
    ref = dataset_ref(root, dataset)
    instruments = pl.read_parquet(ref.instruments_path)
    man = load_manifest(root)
    days = sorted(
        d
        for d, rec in man.get("files", {}).items()
        if rec.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
    )
    exp_id = f"exp-{spec.id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    freeze_dir = paths.experiments / exp_id
    freeze_dir.mkdir(parents=True, exist_ok=True)
    freeze = {
        "experiment_id": exp_id,
        "hypothesis_id": spec.id,
        "hypothesis_hash": h_hash,
        "hypothesis_path": str(Path(hypothesis_path)),
        "frozen_at": datetime.now(UTC).isoformat(),
        "spec": spec.model_dump(mode="json"),
        "note": "Frozen before execution. Thresholds were not changed after seeing results.",
    }
    (freeze_dir / "hypothesis_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    prev_close: dict[str, float] = {}
    vol_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    event_frames: list[pl.DataFrame] = []
    outcome_frames: list[pl.DataFrame] = []
    featured_empty = None
    progress = freeze_dir / "run_progress.json"

    for i, day in enumerate(days):
        bars = pl.read_parquet(local_parquet_path(root, day))
        sessioned = add_session_columns(bars)
        featured = _featured_trigger_window(sessioned, prev_close, vol_hist)
        featured_empty = featured.head(0)
        events = evaluate_events(featured, instruments, spec, experiment_id=exp_id)
        if events.height:
            extra = featured.select(
                "instrument_id",
                pl.col("ts_utc").alias("event_ts"),
                "prev_close",
                "dollar_volume",
            ).unique(subset=["instrument_id", "event_ts"])
            events = events.join(extra, on=["instrument_id", "event_ts"], how="left")
            event_frames.append(events)
            ids = events["instrument_id"].unique().to_list()
            need = sessioned.filter(
                pl.col("instrument_id").is_in(ids) & (pl.col("time_et") <= time(12, 0))
            )
            outcome_frames.append(compute_outcomes(events, need, spec))
        _update_hist(sessioned, prev_close, vol_hist)
        if i % 10 == 0:
            progress.write_text(
                json.dumps({"i": i, "day": day, "n_days": len(days), "event_days": len(event_frames)}),
                encoding="utf-8",
            )

    if event_frames:
        events = pl.concat(event_frames, how="vertical_relaxed")
        outcomes = pl.concat(outcome_frames, how="vertical_relaxed")
    else:
        events = evaluate_events(featured_empty or pl.DataFrame(), instruments, spec, experiment_id=exp_id)
        outcomes = compute_outcomes(events, pl.DataFrame(), spec)

    ev_path = paths.events / f"{exp_id}.parquet"
    oc_path = paths.outcomes / f"{exp_id}.parquet"
    events.write_parquet(ev_path)
    outcomes.write_parquet(oc_path)

    summary = summarize_outcomes(
        events,
        outcomes,
        n_boot=cfg.bootstrap_draws,
        seed=cfg.bootstrap_seed,
        q_target=spec.statistics.q_target,
    )
    summary["robustness"] = robustness_tables(events, outcomes)
    summary["hypothesis_hash"] = h_hash
    rb = summary["robustness"]
    extra_fake = [
        "Stage is EXPLORATORY; this is not sealed OOS.",
        "Day-block bootstrap treats trading days as the resampling unit; events on the same day remain dependent.",
        "Universe uses last CS snapshot per ticker, not a daily PIT listing calendar.",
        "instrument_id is ticker:<symbol>; reuse/changes can mix issuers.",
        "No NBBO: a positive SIGNAL_ONLY mean is not an executable edge.",
        "As-printed prices: split mornings can look like gaps.",
        "gap>=15% and rvol>=5 selects a rare, news-sensitive tail that can be a few names/days.",
    ]
    top = (rb or {}).get("exclude_top_1pct_profitable") or {}
    if top.get("mean") is not None and summary.get("horizons"):
        extra_fake.append(
            f"Excluding the top 1% most profitable events changes the primary-horizon mean "
            f"(see robustness); concentration can manufacture an average."
        )
    summary["how_this_could_be_fake"] = extra_fake
    summary["labels"] = {
        "execution": "SIGNAL_ONLY",
        "stage": "EXPLORATORY",
        "sample_is_research_evidence": False,
        "dataset": ref.name,
        "data_source": "massive_flatfile",
        "is_smoke": False,
        "is_sample": False,
        "volume_liquidity": "VOLUME_LIQUIDITY_PROXY",
        "executable_liquidity": "NOT_MEASURED",
        "n_instruments": instruments.height,
        "bar_ts_min": days[0] if days else None,
        "bar_ts_max": days[-1] if days else None,
        "pipeline": "incremental_day_features_v1",
        "not_edge_evidence": True,
        "raw_statistical_effect_only": True,
        "nbbo": False,
    }
    summary_path = paths.experiments / exp_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    charts = write_charts(outcomes, paths.reports / exp_id / "charts")
    report_path = paths.reports / exp_id / "report.md"
    record = ExperimentRecord(
        experiment_id=exp_id,
        hypothesis_id=spec.id,
        hypothesis_hash=h_hash,
        family_id=spec.family,
        git_sha=git_sha(root),
        data_manifest_hash=dataset_manifest_hash(root, dataset),
        universe_version=UNIVERSE_VERSION,
        feature_version=FEATURE_VERSION,
        execution_model_version=EXECUTION_MODEL_VERSION,
        random_seed=cfg.bootstrap_seed,
        created_at=datetime.now(UTC),
        stage="EXPLORATORY",
        number_of_variants_tested=max(1, len(spec.outcomes.forward_returns)),
        hypothesis_path=str(Path(hypothesis_path)),
        events_path=str(ev_path),
        outcomes_path=str(oc_path),
        summary_path=str(summary_path),
        report_path=str(report_path),
        events_hash=sha256_file(ev_path) if ev_path.exists() else "",
        outcomes_hash=sha256_file(oc_path) if oc_path.exists() else "",
        dataset=ref.name,
        data_source="massive_flatfile",
    )
    render_report(spec, record, summary, charts, report_path)
    write_experiment(record, summary, root)
    if progress.exists():
        progress.unlink()
    return record
