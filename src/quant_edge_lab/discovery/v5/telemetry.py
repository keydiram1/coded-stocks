"""Structured V5 telemetry. Never prints sealed-OOS results."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json
from quant_edge_lab.discovery.v5.checkpoint import v5_run_dir
from quant_edge_lab.discovery.v5.models import TelemetrySnapshot
from quant_edge_lab.discovery.v5.partitions import SEALED_OOS_STATE

SEALED_KEYS = ("sealed_oos_mean", "oos_mean", "D4", "sealed_effect")


def status_path(root: Path, run_id: str) -> Path:
    return v5_run_dir(root, run_id) / "status.json"


def events_log_path(root: Path, run_id: str) -> Path:
    return v5_run_dir(root, run_id) / "telemetry.jsonl"


def strip_sealed(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in payload.items():
        lk = str(k).lower()
        if any(s.lower() in lk for s in SEALED_KEYS):
            continue
        if isinstance(v, dict):
            out[k] = strip_sealed(v)
        else:
            out[k] = v
    return out


def eta_seconds(completed: int, total: int, elapsed_s: float) -> float | None:
    if completed <= 0 or total <= 0 or elapsed_s <= 0:
        return None
    rate = completed / elapsed_s
    remain = max(total - completed, 0)
    return remain / rate if rate > 0 else None


def percent(completed: int, total: int) -> float | None:
    if total <= 0:
        return None
    return 100.0 * completed / total


def fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    s = int(max(seconds, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def persist_snapshot(root: Path, run_id: str, snap: TelemetrySnapshot, extra: dict[str, Any] | None = None) -> Path:
    body = strip_sealed(
        {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "sealed_oos": SEALED_OOS_STATE,
            **snap.model_dump(),
            **(extra or {}),
        }
    )
    path = status_path(root, run_id)
    atomic_write_json(path, body)
    append_jsonl(events_log_path(root, run_id), body)
    return path


class StageClock:
    def __init__(self) -> None:
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
