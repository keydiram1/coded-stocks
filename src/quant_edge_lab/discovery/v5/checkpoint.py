"""Atomic V5 checkpoints. Scientific state is not RAM-only."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.knowledge import atomic_write_json
from quant_edge_lab.discovery.v5.identity import ResumeIdentityError, assert_identity_match


def v5_run_dir(root: Path, run_id: str) -> Path:
    p = Paths(root).derived / "discovery" / "v5" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def ckpt_dir(root: Path, run_id: str) -> Path:
    p = v5_run_dir(root, run_id) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_ckpt(root: Path, run_id: str, name: str, payload: dict[str, Any], ident: dict[str, str]) -> Path:
    rec = {
        "stage": name,
        "identity": ident,
        "timestamp": datetime.now(UTC).isoformat(),
        **payload,
    }
    path = ckpt_dir(root, run_id) / f"{name}.json"
    atomic_write_json(path, rec)
    return path


def load_ckpt(root: Path, run_id: str, name: str, ident: dict[str, str] | None = None) -> dict[str, Any] | None:
    path = ckpt_dir(root, run_id) / f"{name}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if ident is not None:
        assert_identity_match(rec, ident)
    return rec


def stage_complete(root: Path, run_id: str, name: str, ident: dict[str, str]) -> bool:
    rec = load_ckpt(root, run_id, name, ident)
    return bool(rec and rec.get("complete") is True)


def refuse_recompute_if_complete(root: Path, run_id: str, name: str, ident: dict[str, str]) -> bool:
    """True if the frozen stage is already complete for this identity."""
    return stage_complete(root, run_id, name, ident)


def atomic_write_parquet(path: Path, df: pl.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return path


def write_state_table(root: Path, run_id: str, name: str, df: pl.DataFrame, ident: dict[str, str]) -> Path:
    path = ckpt_dir(root, run_id) / f"{name}.parquet"
    atomic_write_parquet(path, df)
    write_ckpt(root, run_id, f"{name}_table", {"path": str(path), "n": df.height, "complete": True}, ident)
    return path


def load_state_table(root: Path, run_id: str, name: str) -> pl.DataFrame | None:
    path = ckpt_dir(root, run_id) / f"{name}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def require_resume_identity(root: Path, run_id: str, ident: dict[str, str]) -> None:
    path = ckpt_dir(root, run_id) / "run_identity.json"
    if not path.exists():
        write_ckpt(root, run_id, "run_identity", {"complete": True}, ident)
        return
    rec = json.loads(path.read_text(encoding="utf-8"))
    try:
        assert_identity_match(rec, ident)
    except ResumeIdentityError:
        raise
