from __future__ import annotations

import json
from pathlib import Path

from quant_edge_lab.config import Paths
from quant_edge_lab.models.schemas import ExperimentRecord


def _dir(root: Path) -> Path:
    p = Paths(root).experiments
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_experiment(record: ExperimentRecord, summary: dict, root: Path) -> Path:
    d = _dir(root) / record.experiment_id
    d.mkdir(parents=True, exist_ok=True)
    rec_path = d / "record.json"
    rec_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    index_path = _dir(root) / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
    return rec_path


def load_experiment(experiment_id: str, root: Path) -> ExperimentRecord:
    path = _dir(root) / experiment_id / "record.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown experiment_id {experiment_id}")
    return ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_experiments(root: Path) -> list[ExperimentRecord]:
    idx = _dir(root) / "index.jsonl"
    if not idx.exists():
        return []
    recs = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        if line.strip():
            recs.append(ExperimentRecord.model_validate_json(line))
    return recs
