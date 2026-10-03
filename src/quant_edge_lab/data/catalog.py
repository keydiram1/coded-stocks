from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.store import sample_bar_path, sample_instrument_path
from quant_edge_lab.hashing import hash_paths


@dataclass(frozen=True)
class DatasetRef:
    name: str
    source: str
    bars_glob: Path
    instruments_path: Path
    is_sample: bool
    is_smoke: bool
    evidence_ok: bool  # never True for smoke/sample


def _normalized_root(root: Path, name: str) -> Path:
    return Paths(root).normalized / name


def resolve_dataset_name(root: Path, name: str) -> str:
    if name == "massive":
        for cand in ("massive-full", "massive-pilot", "massive-smoke"):
            inst = _normalized_root(root, cand) / "instruments.parquet"
            if inst.exists():
                return cand
        raise FileNotFoundError(
            "No Massive dataset found. Download smoke/pilot/full first "
            "(python -m quant_edge_lab massive download --dataset massive-smoke ...)."
        )
    return name


def dataset_ref(root: Path, name: str) -> DatasetRef:
    resolved = resolve_dataset_name(root, name)
    if resolved == "sample":
        return DatasetRef(
            name="sample",
            source="sample",
            bars_glob=sample_bar_path(root),
            instruments_path=sample_instrument_path(root),
            is_sample=True,
            is_smoke=False,
            evidence_ok=False,
        )
    base = _normalized_root(root, resolved)
    return DatasetRef(
        name=resolved,
        source="massive",
        bars_glob=base / "bars_1m",
        instruments_path=base / "instruments.parquet",
        is_sample=False,
        is_smoke=resolved.endswith("smoke"),
        evidence_ok=False,
    )


def load_bars(root: Path, name: str) -> pl.DataFrame:
    ref = dataset_ref(root, name)
    path = ref.bars_glob
    if path.is_file():
        return pl.read_parquet(path)
    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No bar parquet under {path}")
        return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    raise FileNotFoundError(f"Dataset bars not found: {path}")


def load_instruments(root: Path, name: str) -> pl.DataFrame:
    ref = dataset_ref(root, name)
    if not ref.instruments_path.exists():
        raise FileNotFoundError(f"Instruments parquet missing: {ref.instruments_path}")
    return pl.read_parquet(ref.instruments_path)


def dataset_manifest_hash(root: Path, name: str) -> str:
    """Identity of a dataset without loading bar payloads into RAM."""
    from quant_edge_lab.hashing import sha256_json

    ref = dataset_ref(root, name)
    files: list[Path] = []
    if ref.bars_glob.is_file():
        files.append(ref.bars_glob)
    elif ref.bars_glob.is_dir():
        files.extend(sorted(ref.bars_glob.rglob("*.parquet")))
    if ref.instruments_path.exists():
        files.append(ref.instruments_path)
    # Sample/smoke stay byte-hashed (tiny). Large partitioned history hashes path+size.
    if len(files) <= 20:
        return hash_paths(files)
    payload = []
    for p in files:
        st = p.stat()
        payload.append({"rel": p.as_posix(), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return sha256_json(payload)
