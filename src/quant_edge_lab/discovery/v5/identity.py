"""V5 run identity. Data-manifest hash is required (V4R omitted it)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_edge_lab.data.massive.flatfiles import manifest_path
from quant_edge_lab.hashing import git_sha, sha256_file, sha256_json

SCIENCE_ID = "v5_mechanism_expected_observed_discrepancy_v1"


class ResumeIdentityError(RuntimeError):
    pass


def data_manifest_identity(root: Path) -> str:
    """Hash the stocks flat-files JSON manifest, not a full parquet scan."""
    path = manifest_path(root)
    if not path.exists():
        return "MISSING"
    return sha256_file(path)


def identity_blob(
    *,
    manifest_hash: str,
    gates_hash: str,
    git: str,
    data_manifest_hash: str,
    science: str = SCIENCE_ID,
) -> dict[str, str]:
    return {
        "manifest": manifest_hash,
        "gates": gates_hash,
        "git": git,
        "data_manifest": data_manifest_hash,
        "science": science,
    }


def build_identity(root: Path, *, manifest_hash: str, gates_hash: str) -> dict[str, str]:
    return identity_blob(
        manifest_hash=manifest_hash,
        gates_hash=gates_hash,
        git=git_sha(root),
        data_manifest_hash=data_manifest_identity(root),
    )


def assert_identity_match(stored: dict[str, Any], required: dict[str, str]) -> None:
    got = stored.get("identity") if "identity" in stored else stored
    if not isinstance(got, dict):
        raise ResumeIdentityError("checkpoint missing identity")
    need = {k: required[k] for k in required}
    have = {k: got.get(k) for k in required}
    if have != need:
        if have.get("data_manifest") != need.get("data_manifest"):
            raise ResumeIdentityError(
                f"data-manifest mismatch stored={have.get('data_manifest')} required={need.get('data_manifest')}"
            )
        raise ResumeIdentityError(f"checkpoint identity mismatch stored={have} required={need}")


def identity_digest(ident: dict[str, str]) -> str:
    return sha256_json(ident)
