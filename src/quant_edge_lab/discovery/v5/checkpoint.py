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


class InterruptAfter(RuntimeError):
    pass


class DayStore:
    """Durable per-day scientific state: beta panels, RVOL, event partitions."""

    def __init__(self, root: Path, run_id: str, ident: dict[str, str]) -> None:
        self.root = root
        self.run_id = run_id
        self.ident = ident
        require_resume_identity(root, run_id, ident)

    def events_part(self, day: str) -> Path:
        return v5_run_dir(self.root, self.run_id) / "events" / f"date={day}" / "part.parquet"

    def beta_path(self) -> Path:
        return ckpt_dir(self.root, self.run_id) / "beta_history.parquet"

    def rvol_path(self) -> Path:
        return ckpt_dir(self.root, self.run_id) / "rvol_state.json"

    def last_completed_day(self) -> str | None:
        rec = load_ckpt(self.root, self.run_id, "progress", self.ident)
        if not rec:
            return None
        return rec.get("last_completed_day")

    def counters(self) -> tuple[int, int]:
        rec = load_ckpt(self.root, self.run_id, "progress", self.ident) or {}
        return int(rec.get("cumulative_events") or 0), int(rec.get("cumulative_rows") or 0)

    def load_beta_history(self) -> list[pl.DataFrame]:
        p = self.beta_path()
        if not p.exists():
            return []
        df = pl.read_parquet(p)
        if df.height == 0 or "trading_date" not in df.columns:
            return []
        out = []
        for d in df["trading_date"].unique(maintain_order=True).to_list():
            out.append(df.filter(pl.col("trading_date") == d))
        return out

    def load_rvol(self) -> dict[str, list[tuple[str, float]]]:
        p = self.rvol_path()
        if not p.exists():
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: [(a, float(b)) for a, b in v] for k, v in raw.items()}

    def load_events(self) -> pl.DataFrame:
        base = v5_run_dir(self.root, self.run_id) / "events"
        if not base.exists():
            return pl.DataFrame()
        files = sorted(base.glob("date=*/part.parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")

    def commit_day(
        self,
        day: str,
        *,
        beta_history: list[pl.DataFrame],
        rvol_hist: dict[str, list[tuple[str, float]]],
        events_today: pl.DataFrame | None,
        cumulative_events: int,
        cumulative_rows: int,
    ) -> None:
        part = self.events_part(day)
        prev = self.last_completed_day()
        if prev is not None and day <= prev:
            raise ResumeIdentityError(f"refusing double-write events for completed day {day}")
        if events_today is not None and events_today.height:
            atomic_write_parquet(part, events_today)
        if beta_history:
            atomic_write_parquet(self.beta_path(), pl.concat(beta_history, how="diagonal_relaxed"))
        atomic_write_json(self.rvol_path(), {k: v for k, v in rvol_hist.items()})
        write_ckpt(
            self.root,
            self.run_id,
            "progress",
            {
                "complete": False,
                "last_completed_day": day,
                "cumulative_events": cumulative_events,
                "cumulative_rows": cumulative_rows,
            },
            self.ident,
        )
