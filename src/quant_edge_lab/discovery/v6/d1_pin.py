"""Pin the completed D1 artifact. D2 reads this; it does not recompute D1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_edge_lab.hashing import sha256_file

FROZEN_D1_JSON_SHA256 = (
    "d7a7c0996f9602639873475e5fed2a5bdee09bd2cf41b98efaafade0293e8d32"
)
FROZEN_SELECTED_ID = "W15_W15"
FROZEN_D1_DECISION = "FREEZE"
FROZEN_CELL = {
    "id": "W15_W15",
    "impulse_window_minutes": 15,
    "wait_window_minutes": 15,
    "min_abs_shock_z": 2.0,
    "high_retention_min": 0.50,
    "low_retention_max": 0.00,
}
IDENTITY_PIN_KEYS = (
    "manifest",
    "gates",
    "data_manifest",
    "science",
    "instruments",
    "calendar",
    "design",
    "selection",
)
FROZEN_D1_IDENTITY = {
    "manifest": "b6519eb1c9d7ec619bddec1bef1d7b08b0beee6f44ed121bcf578e34b4b4ed85",
    "gates": "1c2172b146589a9b961b509b08e439576f159290cd8bad727bd4777b1483fa0b",
    "data_manifest": "fd208f2a48228d4567100ed3b5e973a1226a4f7a58151ce5c873230cf5eb25e8",
    "science": "v6_market_residual_shock_retention_v1",
    "instruments": "447785f34279d46d1b350dd0af85a34a9caccd9327a6a813bdfc4b4b15fa62b0",
    "calendar": "434da4e718c19ddf7389125d2644741f3112ace1c2276478131c449f1f250ea1",
    "design": "b6519eb1c9d7ec619bddec1bef1d7b08b0beee6f44ed121bcf578e34b4b4ed85",
    "selection": "1c2172b146589a9b961b509b08e439576f159290cd8bad727bd4777b1483fa0b",
}


class D1PinError(RuntimeError):
    pass


def identity_pin(ident: dict[str, Any]) -> dict[str, str]:
    return {k: str(ident.get(k)) for k in IDENTITY_PIN_KEYS}


def frozen_cell_match(cell: dict[str, Any] | None) -> bool:
    if not cell:
        return False
    return (
        str(cell.get("id")) == FROZEN_CELL["id"]
        and int(cell.get("impulse_window_minutes")) == 15
        and int(cell.get("wait_window_minutes")) == 15
        and float(cell.get("min_abs_shock_z")) == 2.0
        and float(cell.get("high_retention_min")) == 0.50
        and float(cell.get("low_retention_max")) == 0.00
    )


def assert_d1_selection(d1: dict[str, Any]) -> dict[str, Any]:
    sel = d1.get("selection") or {}
    if sel.get("decision") != FROZEN_D1_DECISION:
        raise D1PinError(f"D2 refused: D1 decision={sel.get('decision')} (must be FREEZE).")
    if sel.get("selected_id") != FROZEN_SELECTED_ID:
        raise D1PinError(
            f"D2 refused: D1 selected_id={sel.get('selected_id')} (must be {FROZEN_SELECTED_ID})."
        )
    cell = sel.get("frozen_cell")
    if not frozen_cell_match(cell):
        raise D1PinError(f"D2 refused: frozen cell mismatch {cell}")
    return dict(cell)


def assert_d1_sha256(path: Path, expected: str = FROZEN_D1_JSON_SHA256) -> str:
    if not path.exists():
        raise D1PinError(f"D2 refused: missing D1 artifact {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise D1PinError("D2 refused: d1.json sha256 mismatch.")
    return actual


def assert_d2_identity(
    stored: dict[str, Any],
    current: dict[str, Any],
    expected: dict[str, str] | None = None,
) -> None:
    need = expected or FROZEN_D1_IDENTITY
    got = stored.get("identity") if "identity" in stored else stored
    if not isinstance(got, dict):
        raise D1PinError("D2 refused: D1 run identity missing.")
    have_stored = identity_pin(got)
    have_current = identity_pin(current)
    if have_stored != need:
        raise D1PinError("D2 refused: stored D1 identity does not match freeze pin.")
    if have_current != need:
        raise D1PinError("D2 refused: live identity does not match freeze pin.")


def load_d1_pin(
    run_dir: Path,
    *,
    expected_sha256: str = FROZEN_D1_JSON_SHA256,
    expected_identity: dict[str, str] | None = None,
    current_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    d1_path = run_dir / "d1.json"
    sha = assert_d1_sha256(d1_path, expected_sha256)
    d1 = json.loads(d1_path.read_text(encoding="utf-8"))
    cell = assert_d1_selection(d1)
    ident_path = run_dir / "run_identity.json"
    if not ident_path.exists():
        raise D1PinError("D2 refused: missing run_identity.json")
    stored = json.loads(ident_path.read_text(encoding="utf-8"))
    if current_identity is not None:
        assert_d2_identity(stored, current_identity, expected_identity)
    return {
        "d1_sha256": sha,
        "d1_path": str(d1_path),
        "decision": FROZEN_D1_DECISION,
        "selected_id": FROZEN_SELECTED_ID,
        "frozen_cell": cell,
        "stored_identity": stored.get("identity") or stored,
    }
