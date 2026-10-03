"""Atomic JSONL / JSON knowledge store. Failures are first-class records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.hashing import sha256_json


def knowledge_dir(root: Path) -> Path:
    p = root / "knowledge"
    p.mkdir(parents=True, exist_ok=True)
    (p / "experiments").mkdir(exist_ok=True)
    (p / "families").mkdir(exist_ok=True)
    return p


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def import_experiment_001(root: Path) -> Path:
    """Record Exp #1 in knowledge without altering frozen experiment files."""
    rec = {
        "hypothesis_id": "gap_rvol_continuation_v1",
        "family_id": "H000",
        "title": "Unconditional long continuation after large gap + early RVOL",
        "version": "v1",
        "frozen_experiment_id": "exp-gap_rvol_continuation_v1-20261002075236",
        "hypothesis_hash": "783aeaac9d848b1368613720f3a2f6aeccd8c4e3fd553071b07159b8acb4bd20",
        "artifacts_unchanged": True,
        "research_status": "REJECTED",
        "rejected": "unconditional long continuation after ≥15% gap + extreme early RVOL",
        "observation": "population showed increasingly negative SIGNAL_ONLY forward returns through 60m",
        "follow_up": [
            "fade / first-event-only (H001)",
            "structural partition hold-near-high vs lose-open (H002)",
        ],
        "results": {
            "events": 405071,
            "ticker_days": 8728,
            "tickers": 3121,
            "days_with_events": 1250,
            "means": {"1m": -0.000242, "5m": -0.001384, "15m": -0.003841, "30m": -0.006170, "60m": -0.008244},
        },
        "known_weaknesses": [
            "repeated events per ticker-day",
            "universe identity not fully PIT",
            "as-printed prices/splits",
            "SIGNAL_ONLY",
            "no NBBO/spread/fill model",
        ],
        "execution_model": "next_bar_open_v1",
        "imported_at": datetime.now(UTC).isoformat(),
        "record_hash": None,
    }
    rec["record_hash"] = sha256_json({k: rec[k] for k in rec if k != "record_hash"})
    path = knowledge_dir(root) / "experiments" / "exp-gap_rvol_continuation_v1-20261002075236.json"
    if path.exists():
        return path
    atomic_write_json(path, rec)
    append_jsonl(knowledge_dir(root) / "index.jsonl", {"type": "import_exp001", "path": str(path)})
    return path


def search_knowledge(root: Path, query: str) -> list[dict[str, Any]]:
    q = query.lower()
    hits = []
    kdir = knowledge_dir(root)
    for p in list(kdir.glob("**/*.json")) + list(kdir.glob("**/*.yaml")):
        text = p.read_text(encoding="utf-8").lower()
        if q in text:
            hits.append({"path": str(p), "name": p.name})
    return hits[:50]


def record_run(root: Path, payload: dict[str, Any]) -> None:
    payload = {**payload, "recorded_at": datetime.now(UTC).isoformat()}
    append_jsonl(knowledge_dir(root) / "index.jsonl", payload)
    fam = payload.get("family_id", "unknown")
    atomic_write_json(knowledge_dir(root) / "experiments" / f"{fam}-{payload.get('batch_id','run')}.json", payload)
