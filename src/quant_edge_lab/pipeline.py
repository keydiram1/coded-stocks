from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_edge_lab.config import Paths, load_research_config
from quant_edge_lab.data.catalog import dataset_manifest_hash, dataset_ref, load_bars, load_instruments
from quant_edge_lab.events.engine import evaluate_events
from quant_edge_lab.execution.discovery import EXECUTION_MODEL_VERSION
from quant_edge_lab.features.engine import FEATURE_VERSION, compute_features
from quant_edge_lab.hashing import git_sha, sha256_json, sha256_text
from quant_edge_lab.hypotheses.loader import hypothesis_hash, load_hypothesis
from quant_edge_lab.models.schemas import ExperimentRecord
from quant_edge_lab.outcomes.engine import compute_outcomes
from quant_edge_lab.registry.store import write_experiment
from quant_edge_lab.reporting.report import render_report, write_charts
from quant_edge_lab.statistics.summarize import summarize_outcomes

UNIVERSE_VERSION = "universe_prev_close_v1"


def scientific_frame_hash(df: pl.DataFrame, exclude: tuple[str, ...] = ("experiment_id",)) -> str:
    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        return sha256_json({})
    payload = df.select(cols).sort(cols).write_csv()
    return sha256_text(payload)


def run_hypothesis(
    hypothesis_path: Path | str,
    root: Path | None = None,
    experiment_id: str | None = None,
    dataset: str = "sample",
) -> ExperimentRecord:
    root = Path(root) if root else Path.cwd()
    paths = Paths(root)
    paths.ensure()
    cfg = load_research_config(paths.config / "research.yaml")
    spec = load_hypothesis(hypothesis_path)
    ref = dataset_ref(root, dataset)
    bars = load_bars(root, dataset)
    instruments = load_instruments(root, dataset)
    featured = compute_features(bars)

    exp_id = experiment_id or f"exp-{spec.id}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    events = evaluate_events(featured, instruments, spec, experiment_id=exp_id)
    outcomes = compute_outcomes(events, bars, spec)

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
    n_instruments = instruments.height if instruments.height else 0
    date_min = str(bars["ts_utc"].min()) if bars.height else None
    date_max = str(bars["ts_utc"].max()) if bars.height else None
    summary["labels"] = {
        "execution": "SIGNAL_ONLY",
        "stage": "EXPLORATORY",
        "sample_is_research_evidence": False,
        "dataset": ref.name,
        "data_source": ref.source,
        "is_smoke": ref.is_smoke,
        "is_sample": ref.is_sample,
        "volume_liquidity": "VOLUME_LIQUIDITY_PROXY",
        "executable_liquidity": "NOT_MEASURED",
        "n_instruments": n_instruments,
        "bar_ts_min": date_min,
        "bar_ts_max": date_max,
        "not_edge_evidence": True,
    }
    summary_path = paths.experiments / exp_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    charts = write_charts(outcomes, paths.reports / exp_id / "charts")
    report_path = paths.reports / exp_id / "report.md"
    h_hash = hypothesis_hash(spec)
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
        events_hash=scientific_frame_hash(events),
        outcomes_hash=scientific_frame_hash(outcomes),
        dataset=ref.name,
        data_source=ref.source,
    )
    render_report(spec, record, summary, charts, report_path)
    write_experiment(record, summary, root)
    return record


def reproduce_experiment(experiment_id: str, root: Path | None = None) -> dict:
    from quant_edge_lab.registry.store import load_experiment

    root = Path(root) if root else Path.cwd()
    original = load_experiment(experiment_id, root)
    replay_id = f"{experiment_id}-repro"
    replay = run_hypothesis(
        original.hypothesis_path,
        root=root,
        experiment_id=replay_id,
        dataset=original.dataset,
    )
    match = (
        original.hypothesis_hash == replay.hypothesis_hash
        and original.events_hash == replay.events_hash
        and original.outcomes_hash == replay.outcomes_hash
        and original.data_manifest_hash == replay.data_manifest_hash
    )
    result = {
        "original_experiment_id": experiment_id,
        "replay_experiment_id": replay_id,
        "match": match,
        "original_events_hash": original.events_hash,
        "replay_events_hash": replay.events_hash,
        "original_outcomes_hash": original.outcomes_hash,
        "replay_outcomes_hash": replay.outcomes_hash,
    }
    out = Paths(root).experiments / experiment_id / "reproduce.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
