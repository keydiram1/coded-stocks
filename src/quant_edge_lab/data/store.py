from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.hashing import hash_paths


BARS_COLUMNS = [
    "instrument_id",
    "ticker",
    "ts_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "source",
]


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def sample_bar_path(root: Path) -> Path:
    return Paths(root).sample / "bars_1m.parquet"


def sample_instrument_path(root: Path) -> Path:
    return Paths(root).sample / "instruments.parquet"


def sample_actions_path(root: Path) -> Path:
    return Paths(root).sample / "corporate_actions.parquet"


def data_manifest_hash(root: Path) -> str:
    sample = Paths(root).sample
    files = sorted(sample.glob("*.parquet")) if sample.exists() else []
    return hash_paths(files)
