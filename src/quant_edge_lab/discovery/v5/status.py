"""Read-only V5 status. Must not mutate artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.v5.telemetry import fmt_hms, status_path


def latest_run_id(root: Path) -> str | None:
    base = Paths(root).derived / "discovery" / "v5"
    if not base.exists():
        return None
    runs = [p.name for p in base.iterdir() if p.is_dir() and (p / "status.json").exists()]
    if not runs:
        return None
    return sorted(runs)[-1]


def read_status(root: Path, run_id: str | None = None) -> dict[str, Any] | None:
    rid = run_id or latest_run_id(root)
    if not rid:
        return None
    path = status_path(root, rid)
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.setdefault("run_id", rid)
    return rec


def format_status(rec: dict[str, Any] | None) -> str:
    if not rec:
        return "[V5] no run status"
    stage = rec.get("stage", "?")
    camp = rec.get("campaign_id") or rec.get("current_unit") or "V5"
    completed = rec.get("completed") or 0
    total = rec.get("total") or 0
    events = rec.get("events_generated") or 0
    effect = rec.get("allowed_split_mean_effect_bp")
    effect_s = f"{effect:.1f}bp" if isinstance(effect, int | float) else "n/a"
    elapsed = fmt_hms(rec.get("elapsed_s"))
    eta = fmt_hms(rec.get("eta_s"))
    ck = rec.get("checkpoint_state") or rec.get("last_successful_checkpoint") or "?"
    lines = [
        f"[V5][{camp}][{stage}]",
        f"day={completed}/{total}",
        f"rows={rec.get('rows_processed')}",
        f"events={events:,}" if isinstance(events, int) else f"events={events}",
        f"effect={effect_s}",
        f"elapsed={elapsed}",
        f"eta={eta}",
        f"checkpoint={ck}",
    ]
    for h in rec.get("hypothesis_lines") or []:
        lines.append(str(h))
    return "\n".join(lines)


def summarize_status(root: Path, run_id: str | None = None) -> str:
    return format_status(read_status(root, run_id))
