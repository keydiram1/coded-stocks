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


def write_ckpt(
    root: Path, run_id: str, name: str, payload: dict[str, Any], ident: dict[str, str]
) -> Path:
    rec = {
        "stage": name,
        "identity": ident,
        "timestamp": datetime.now(UTC).isoformat(),
        **payload,
    }
    path = ckpt_dir(root, run_id) / f"{name}.json"
    atomic_write_json(path, rec)
    return path


def load_ckpt(
    root: Path, run_id: str, name: str, ident: dict[str, str] | None = None
) -> dict[str, Any] | None:
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


def write_state_table(
    root: Path, run_id: str, name: str, df: pl.DataFrame, ident: dict[str, str]
) -> Path:
    path = ckpt_dir(root, run_id) / f"{name}.parquet"
    atomic_write_parquet(path, df)
    write_ckpt(
        root, run_id, f"{name}_table", {"path": str(path), "n": df.height, "complete": True}, ident
    )
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


class CrashAfter(RuntimeError):
    """Test hook: crash after a named artifact write, before progress advance."""


class DayStore:
    """Per-day immutable snapshots. Progress pointer advances only after all artifacts exist."""

    def __init__(self, root: Path, run_id: str, ident: dict[str, str]) -> None:
        self.root = root
        self.run_id = run_id
        self.ident = ident
        require_resume_identity(root, run_id, ident)

    def events_part(self, day: str) -> Path:
        return v5_run_dir(self.root, self.run_id) / "events" / f"date={day}" / "part.parquet"

    def state_dir(self, day: str) -> Path:
        p = ckpt_dir(self.root, self.run_id) / "state" / f"day={day}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def last_completed_day(self) -> str | None:
        rec = load_ckpt(self.root, self.run_id, "progress", self.ident)
        if not rec:
            return None
        return rec.get("last_completed_day")

    def counters(self) -> tuple[int, int]:
        rec = load_ckpt(self.root, self.run_id, "progress", self.ident) or {}
        return int(rec.get("cumulative_events") or 0), int(rec.get("cumulative_rows") or 0)

    def _split_beta(self, df: pl.DataFrame) -> list[pl.DataFrame]:
        if df.height == 0 or "trading_date" not in df.columns:
            return []
        return [
            df.filter(pl.col("trading_date") == d)
            for d in df["trading_date"].unique(maintain_order=True).to_list()
        ]

    def load_beta_history(self) -> list[pl.DataFrame]:
        last = self.last_completed_day()
        if not last:
            return []
        p = self.state_dir(last) / "beta_history.parquet"
        if not p.exists():
            return []
        return self._split_beta(pl.read_parquet(p))

    def load_rvol(self) -> dict[str, list[tuple[str, float]]]:
        last = self.last_completed_day()
        if not last:
            return {}
        p = self.state_dir(last) / "rvol_state.json"
        if not p.exists():
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: [(a, float(b)) for a, b in v] for k, v in raw.items()}

    def load_elig(self) -> dict[str, dict[str, list[tuple[str, float]]]]:
        last = self.last_completed_day()
        if not last:
            return {}
        p = self.state_dir(last) / "elig_state.json"
        if not p.exists():
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        out: dict[str, dict[str, list[tuple[str, float]]]] = {}
        for iid, rec in raw.items():
            out[iid] = {
                "closes": [(a, float(b)) for a, b in rec.get("closes") or []],
                "dvols": [(a, float(b)) for a, b in rec.get("dvols") or []],
            }
        return out

    def load_events(self) -> pl.DataFrame:
        last = self.last_completed_day()
        base = v5_run_dir(self.root, self.run_id) / "events"
        if not last or not base.exists():
            return pl.DataFrame()
        files = []
        for p in sorted(base.glob("date=*/part.parquet")):
            day = p.parent.name.replace("date=", "")
            if day <= last:
                files.append(p)
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
        elig_hist: dict[str, dict[str, list[tuple[str, float]]]] | None = None,
        crash_after: str | None = None,
    ) -> None:
        prev = self.last_completed_day()
        if prev is not None and day <= prev:
            raise ResumeIdentityError(f"refusing double-write events for completed day {day}")
        part = self.events_part(day)
        if events_today is not None and events_today.height:
            atomic_write_parquet(part, events_today)
        if crash_after == "events":
            raise CrashAfter("events")
        snap = self.state_dir(day)
        if beta_history:
            atomic_write_parquet(
                snap / "beta_history.parquet", pl.concat(beta_history, how="diagonal_relaxed")
            )
        else:
            atomic_write_parquet(snap / "beta_history.parquet", pl.DataFrame())
        if crash_after == "beta":
            raise CrashAfter("beta")
        atomic_write_json(snap / "rvol_state.json", rvol_hist)
        if crash_after == "rvol":
            raise CrashAfter("rvol")
        atomic_write_json(snap / "elig_state.json", elig_hist or {})
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
