"""CLOSE_DISLOCATION_CONTINUATION_V1. Confirmation-only. Does not mutate reversal science."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from quant_edge_lab.config import Paths
from quant_edge_lab.discovery.v5.campaigns.close_features import (
    resolution_direction,
    signed_resolution,
)
from quant_edge_lab.discovery.v5.identity import (
    ResumeIdentityError,
    build_identity,
    data_manifest_identity,
)
from quant_edge_lab.discovery.v5.models import CandidateRule, Direction
from quant_edge_lab.hashing import sha256_file

CAMPAIGN_ID = "CLOSE_DISLOCATION_CONTINUATION_V1"
MECHANISM_ID = "forced_eod_preclose_flow_continuation"
SCIENCE_ID = "v5_close_dislocation_continuation_confirmation_v1"
SOURCE_CAMPAIGN = "CLOSE_DISLOCATION_REVERSAL_V1"
SOURCE_RUN_ID = "v5-close-dislocation"
DEFAULT_RUN_ID = "v5-close-continuation"
MANIFEST_REL = Path("knowledge/campaigns/v5_continuation_manifest.yaml")
GATES_REL = Path("knowledge/campaigns/v5_continuation_gates.yaml")
D3 = ("2024-10-11", "2026-10-01")
APPROVED = frozenset({"APPROVED", "FROZEN"})
D3_PAYLOAD_LOADS = 0

H1_DISLOC = 0.0046049372056016155
H1_RVOL = 1.3190530517761354
H2_DISLOC = 0.0063376019291080795
H2_RVOL = 1.3190530517761354
H3_DISLOC = 0.0046049372056016155

SOURCE_IDENTITY = {
    "git": "47e8592b8d18438ed11741a637bba033b952855d",
    "manifest": "8bca7cc6e6a7529be004267433616011e9799ec54ee1a09a71b16d4d3cc55875",
    "gates": "bfa3450a2f0a035ea979e86d2c298374ef7f9d4cb4a312656a7f75a1e182f88a",
    "data_manifest": "fd208f2a48228d4567100ed3b5e973a1226a4f7a58151ce5c873230cf5eb25e8",
    "instruments": "447785f34279d46d1b350dd0af85a34a9caccd9327a6a813bdfc4b4b15fa62b0",
    "calendar": "434da4e718c19ddf7389125d2644741f3112ace1c2276478131c449f1f250ea1",
}


def continuation_direction(disc: float | None) -> Direction | None:
    if disc is None or not np.isfinite(disc) or disc == 0.0:
        return None
    return "LONG" if disc > 0 else "SHORT"


def continuation_signed(forward: float | None, disc: float | None) -> float | None:
    return signed_resolution(forward, continuation_direction(disc))


def reversal_signed(forward: float | None, disc: float | None) -> float | None:
    return signed_resolution(forward, resolution_direction(disc))


def load_continuation(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    man = yaml.safe_load((root / MANIFEST_REL).read_text(encoding="utf-8"))
    gates = yaml.safe_load((root / GATES_REL).read_text(encoding="utf-8"))
    return man, gates


def freeze_continuation(root: Path) -> dict[str, str]:
    ident = build_identity(
        root,
        manifest_hash=sha256_file(root / MANIFEST_REL),
        gates_hash=sha256_file(root / GATES_REL),
    )
    ident["science"] = SCIENCE_ID
    ident["data_manifest"] = data_manifest_identity(root)
    return ident


def campaign_block(man: dict[str, Any]) -> dict[str, Any]:
    return man[man["primary_campaign"]]


def frozen_rules(camp: dict[str, Any] | None = None) -> list[CandidateRule]:
    specs = (camp or {}).get("hypotheses") if camp else None
    if not specs:
        specs = [
            {
                "hypothesis_id": "H1_PRIMARY",
                "role": "primary",
                "dislocation_abs_min": H1_DISLOC,
                "rvol_min": H1_RVOL,
                "require_rvol": True,
            },
            {
                "hypothesis_id": "H2_ABS_Q95",
                "role": "robustness",
                "dislocation_abs_min": H2_DISLOC,
                "rvol_min": H2_RVOL,
                "require_rvol": True,
            },
            {
                "hypothesis_id": "H3_DISLOCATION_ONLY",
                "role": "robustness",
                "dislocation_abs_min": H3_DISLOC,
                "rvol_min": None,
                "require_rvol": False,
            },
        ]
    out: list[CandidateRule] = []
    for spec in specs:
        out.append(
            CandidateRule(
                hypothesis_id=spec["hypothesis_id"],
                mechanism_id=MECHANISM_ID,
                role=spec["role"],
                dislocation_abs_min=spec.get("dislocation_abs_min"),
                rvol_min=spec.get("rvol_min"),
                require_rvol=bool(spec["require_rvol"]),
                direction_policy="continuation_no_flip",
                primary_outcome="next_open_to_15m",
                frozen_from_split="D1",
                frozen=True,
            )
        )
    return out


def assert_frozen_thresholds(rules: list[CandidateRule]) -> None:
    by = {r.hypothesis_id: r for r in rules}
    if by["H1_PRIMARY"].role != "primary":
        raise ResumeIdentityError("H1 must remain primary")
    if by["H1_PRIMARY"].dislocation_abs_min != H1_DISLOC or by["H1_PRIMARY"].rvol_min != H1_RVOL:
        raise ResumeIdentityError("H1 frozen threshold drifted")
    if by["H2_ABS_Q95"].dislocation_abs_min != H2_DISLOC or by["H2_ABS_Q95"].rvol_min != H2_RVOL:
        raise ResumeIdentityError("H2 frozen threshold drifted")
    if by["H3_DISLOCATION_ONLY"].dislocation_abs_min != H3_DISLOC:
        raise ResumeIdentityError("H3 frozen threshold drifted")
    if by["H3_DISLOCATION_ONLY"].rvol_min is not None or by["H3_DISLOCATION_ONLY"].require_rvol:
        raise ResumeIdentityError("H3 must remain dislocation-only")


def assert_execution_approved(man: dict[str, Any], gates: dict[str, Any]) -> None:
    ms = str(man.get("execution_status") or "NOT_APPROVED")
    gs = str(gates.get("execution_status") or "NOT_APPROVED")
    if ms not in APPROVED or gs not in APPROVED:
        raise RuntimeError(
            f"execute_v5_continuation refused: manifest.execution_status={ms} "
            f"gates.execution_status={gs} (both must be APPROVED or FROZEN)."
        )


def refuse_discovery_split(split: str) -> None:
    if split in {"D1", "D2"}:
        raise RuntimeError(
            "continuation campaign refuses D1/D2 empirical evaluation; "
            "D3 is the sole confirmation split"
        )
    if split != "D3":
        raise RuntimeError(f"continuation campaign only evaluates D3, got {split}")


def source_run_dir(root: Path) -> Path:
    return Paths(root).derived / "discovery" / "v5" / SOURCE_RUN_ID


def read_source_identity(root: Path) -> dict[str, str]:
    progress = source_run_dir(root) / "checkpoints" / "progress.json"
    run_id_path = source_run_dir(root) / "checkpoints" / "run_identity.json"
    path = progress if progress.exists() else run_id_path
    if not path.exists():
        raise ResumeIdentityError(
            "source reversal run identity missing; refuse continuation execute"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    ident = rec.get("identity") if isinstance(rec, dict) else None
    if not isinstance(ident, dict):
        raise ResumeIdentityError("source checkpoint missing identity")
    missing = [k for k in SOURCE_IDENTITY if k not in ident]
    if missing:
        raise ResumeIdentityError(f"source checkpoint identity missing keys {missing}")
    return {k: str(ident[k]) for k in SOURCE_IDENTITY}


def assert_source_identity(root: Path) -> dict[str, str]:
    got = read_source_identity(root)
    if got != SOURCE_IDENTITY:
        raise ResumeIdentityError(
            f"source identity mismatch stored={got} required={SOURCE_IDENTITY}"
        )
    return got


def attach_continuation(events: pl.DataFrame) -> pl.DataFrame:
    if events.height == 0:
        return events
    if "next_open_to_15m" in events.columns:
        fwd = events["next_open_to_15m"].to_list()
    else:
        fwd = [None] * events.height
    disc = events["discrepancy"].to_list()
    rev = [reversal_signed(f, d) for f, d in zip(fwd, disc, strict=True)]
    cont = [continuation_signed(f, d) for f, d in zip(fwd, disc, strict=True)]
    direc = [continuation_direction(d) for d in disc]
    out = events
    if "primary_signed" in out.columns:
        out = out.rename({"primary_signed": "reversal_primary_signed"})
    return out.with_columns(
        pl.Series("reversal_signed_next_open_to_15m", rev, dtype=pl.Float64),
        pl.Series("continuation_signed", cont, dtype=pl.Float64),
        pl.Series("primary_signed", cont, dtype=pl.Float64),
        pl.Series("continuation_direction", direc),
        pl.lit(MECHANISM_ID).alias("mechanism_id"),
    )


def assert_d3_payload_allowed(man: dict[str, Any], gates: dict[str, Any]) -> None:
    assert_execution_approved(man, gates)


def load_d3_event_partitions(root: Path) -> pl.DataFrame:
    """Load source D3 event parquet. Forbidden until both YAML files are APPROVED/FROZEN."""
    global D3_PAYLOAD_LOADS
    man, gates = load_continuation(root)
    assert_d3_payload_allowed(man, gates)
    base = source_run_dir(root) / "events"
    if not base.exists():
        raise ResumeIdentityError("source event store missing")
    files = []
    for p in sorted(base.glob("date=*/part.parquet")):
        day = p.parent.name.replace("date=", "")
        if D3[0] <= day <= D3[1]:
            files.append(p)
    if not files:
        raise ResumeIdentityError("no D3 event partitions in source run")
    D3_PAYLOAD_LOADS += 1
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def new_run_dir(root: Path, run_id: str = DEFAULT_RUN_ID) -> Path:
    from quant_edge_lab.discovery.v5.checkpoint import v5_run_dir

    if run_id == SOURCE_RUN_ID:
        raise ResumeIdentityError("refusing to write into the frozen reversal run directory")
    return v5_run_dir(root, run_id)


def assert_continuation_negates_reversal(events: pl.DataFrame) -> None:
    if events.height == 0:
        return
    cont = events["continuation_signed"].to_list()
    if "reversal_signed_next_open_to_15m" in events.columns:
        rev = events["reversal_signed_next_open_to_15m"].to_list()
    elif "reversal_primary_signed" in events.columns:
        rev = events["reversal_primary_signed"].to_list()
    else:
        return
    for c, r in zip(cont, rev, strict=True):
        cf = c is not None and bool(np.isfinite(c))
        rf = r is not None and bool(np.isfinite(r))
        if cf != rf:
            raise AssertionError("continuation/reversal finite mask mismatch")
        if cf and float(c) != -float(r):
            raise AssertionError("continuation_signed must equal -reversal primary_signed")
