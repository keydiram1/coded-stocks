"""PIT instrument snapshots from Massive REST. Does not download bars."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.client import MassiveClient, probe_bases
from quant_edge_lab.data.massive.flatfiles import DATASET
from quant_edge_lab.data.massive.ingest import fetch_universe_cs
from quant_edge_lab.data.massive.normalize import normalize_splits, normalize_tickers
from quant_edge_lab.secrets import massive_api_key


def prepare_pit_reference(root: Path, asof_dates: list[str] | None = None) -> dict[str, Any]:
    key = massive_api_key(root)
    base = probe_bases(key)
    client = MassiveClient(api_key=key, base_url=base)
    dates = asof_dates or ["2021-10-01", "2022-01-03", "2023-01-03", "2024-01-02", "2025-01-02", "2026-10-01"]
    frames: list[pl.DataFrame] = []
    counts = {}
    for d in dates:
        rows = fetch_universe_cs(
            client,
            asof=date.fromisoformat(d),
            include_inactive=True,
            max_results=None,
            page_sleep_s=0.05,
        )
        # Starter: no 12.5s sleep; paginate default is 0.
        df = normalize_tickers(rows)
        if df.height:
            df = df.with_columns(pl.lit(d).alias("asof_date"))
            frames.append(df)
        counts[d] = df.height
    inst = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    out_dir = Paths(root).normalized / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_path = out_dir / "instruments.parquet"
    if inst.height:
        inst = inst.with_columns((pl.lit("ticker:") + pl.col("ticker")).alias("instrument_id"))
        inst = inst.sort("asof_date").unique(subset=["ticker"], keep="last")
        inst.write_parquet(inst_path)

    splits_rows = list(
        client.paginate(
            "/v3/reference/splits",
            {"limit": 1000, "order": "asc", "sort": "execution_date", "execution_date.gte": dates[0]},
        )
    )
    splits = normalize_splits(splits_rows)
    ca_path = out_dir / "corporate_actions.parquet"
    if splits.height:
        splits.write_parquet(ca_path)

    return {
        "instruments_path": str(inst_path),
        "instruments_rows": inst.height,
        "asof_counts": counts,
        "splits_rows": splits.height,
        "corporate_actions_path": str(ca_path),
        "instrument_id_scheme": "ticker:<symbol>",
    }
