"""Durable V4R checkpoints with identity hashes. Incompatible artifacts are rejected."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v4r.stage1 import v4r_ckpt_dir


def identity_blob(*, cfg_hash: str, git: str, manifest: str, gates: str, science: str) -> dict[str, str]:
    return {"cfg_hash": cfg_hash, "git": git, "manifest": manifest, "gates": gates, "science": science}


def write_ckpt(root: Path, name: str, payload: dict[str, Any], ident: dict[str, str]) -> Path:
    rec = {
        "stage": name,
        "identity": ident,
        "timestamp": datetime.now(UTC).isoformat(),
        **payload,
    }
    path = v4r_ckpt_dir(root) / f"{name}.json"
    atomic_write_json(path, rec)
    return path


def load_ckpt(root: Path, name: str, ident: dict[str, str]) -> dict[str, Any] | None:
    path = v4r_ckpt_dir(root) / f"{name}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    got = rec.get("identity") or {}
    if got != ident:
        raise RuntimeError(
            f"checkpoint {name} incompatible: stored={got} required={ident}. Regenerate {path}"
        )
    return rec
