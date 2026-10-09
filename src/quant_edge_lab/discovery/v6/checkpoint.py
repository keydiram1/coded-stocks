from __future__ import annotations

from pathlib import Path

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v5.identity import assert_identity_match


def v6_run_dir(root: Path, run_id: str) -> Path:
    p = Paths(root).derived / "discovery" / "v6" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_identity(root: Path, run_id: str, ident: dict[str, str]) -> Path:
    path = v6_run_dir(root, run_id) / "run_identity.json"
    atomic_write_json(path, {"identity": ident})
    return path


def require_resume_identity(root: Path, run_id: str, required: dict[str, str]) -> None:
    path = v6_run_dir(root, run_id) / "run_identity.json"
    if not path.exists():
        write_identity(root, run_id, required)
        return
    import json

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert_identity_match(stored, required)
