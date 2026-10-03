"""Resumable Massive 1-minute ingest. Raw prices unadjusted. No fake fills."""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.client import MassiveClient, MassiveHTTPError
from quant_edge_lab.data.massive.normalize import (
    EXCHANGE_TO_CANON,
    instrument_id_for,
    normalize_aggs,
    normalize_splits,
    normalize_tickers,
)

INGEST_VERSION = "massive_bars_1m_v1"
SMOKE_TICKERS = ("AAPL", "AMD", "GME", "F", "NOK")
SMOKE_START = date(2026, 9, 15)
SMOKE_END = date(2026, 9, 17)


def _checkpoint_path(root: Path, dataset: str) -> Path:
    return Paths(root).manifests / "massive" / dataset / "checkpoint.json"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "failures": [], "meta": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _write_day_parts(root: Path, dataset: str, bars: pl.DataFrame) -> list[Path]:
    if bars.height == 0:
        return []
    df = bars.with_columns(pl.col("ts_utc").dt.date().alias("date"))
    written = []
    raw_root = Paths(root).raw / "massive" / dataset / "bars_1m"
    norm_root = Paths(root).normalized / dataset / "bars_1m"
    for day, part in df.partition_by("date", as_dict=True).items():
        d = day[0] if isinstance(day, tuple) else day
        raw_p = raw_root / f"date={d}" / "part.parquet"
        norm_p = norm_root / f"date={d}" / "part.parquet"
        raw_p.parent.mkdir(parents=True, exist_ok=True)
        norm_p.parent.mkdir(parents=True, exist_ok=True)
        # Merge with existing (idempotent).
        if norm_p.exists():
            prev = pl.read_parquet(norm_p)
            part = pl.concat([prev, part], how="diagonal_relaxed")
        part = part.unique(subset=["instrument_id", "ts_utc"], keep="last").sort(
            ["instrument_id", "ts_utc"]
        )
        part.write_parquet(raw_p)
        canon = [c for c in part.columns if c != "date"]
        part.select(canon).write_parquet(norm_p)
        written.append(norm_p)
    return written


def fetch_aggs_range(
    client: MassiveClient,
    ticker: str,
    start: date,
    end: date,
    limit: int = 50000,
) -> list[dict]:
    path = f"/v2/aggs/ticker/{ticker}/range/1/minute/{start.isoformat()}/{end.isoformat()}"
    return list(
        client.paginate(
            path,
            {"adjusted": "false", "sort": "asc", "limit": limit},
        )
    )


def fetch_ticker_metadata(client: MassiveClient, ticker: str) -> dict:
    data = client.get("/v3/reference/tickers", {"ticker": ticker, "market": "stocks", "limit": 1})
    results = data.get("results") or []
    return results[0] if results else {"ticker": ticker}


def fetch_universe_cs(
    client: MassiveClient,
    asof: date | None = None,
    include_inactive: bool = True,
    limit: int = 1000,
    max_results: int | None = None,
    page_sleep_s: float = 0.05,
) -> list[dict]:
    rows: list[dict] = []
    params_base: dict[str, Any] = {
        "market": "stocks",
        "type": "CS",
        "limit": limit,
        "order": "asc",
        "sort": "ticker",
    }
    if asof:
        params_base["date"] = asof.isoformat()
    actives = [True]
    if include_inactive:
        actives.append(False)
    for active in actives:
        params = dict(params_base)
        params["active"] = "true" if active else "false"
        remaining = None if max_results is None else max(0, max_results - len(rows))
        if remaining == 0:
            break
        rows.extend(
            list(
                client.paginate(
                    "/v3/reference/tickers",
                    params,
                    max_items=remaining,
                    page_sleep_s=page_sleep_s,
                )
            )
        )
        if max_results is not None and len(rows) >= max_results:
            break
    return rows


def ingest_tickers(
    client: MassiveClient,
    root: Path,
    dataset: str,
    tickers: Iterable[str],
    start: date,
    end: date,
    sleep_s: float = 12.5,
) -> dict[str, Any]:
    ck_path = _checkpoint_path(root, dataset)
    ck = _load_checkpoint(ck_path)
    ck.setdefault("completed", {})
    ck.setdefault("failures", [])
    ck["meta"] = {
        "dataset": dataset,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "ingest_version": INGEST_VERSION,
        "adjusted": False,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    inst_rows = []
    api_calls = 0
    t0 = time.time()
    ticker_list = list(tickers)
    for ticker in ticker_list:
        key = f"{ticker}|{start}|{end}"
        if ck["completed"].get(key, {}).get("ok"):
            continue
        try:
            meta = fetch_ticker_metadata(client, ticker)
            api_calls += 1
            iid = instrument_id_for(
                ticker, meta.get("composite_figi"), str(meta["cik"]) if meta.get("cik") else None
            )
            results = fetch_aggs_range(client, ticker, start, end)
            api_calls += 1
            bars = normalize_aggs(ticker, results, instrument_id=iid)
            _write_day_parts(root, dataset, bars)
            inst_rows.append(
                {
                    "instrument_id": iid,
                    "ticker": ticker,
                    "effective_from": datetime.combine(start, datetime.min.time()),
                    "effective_to": None,
                    "exchange": EXCHANGE_TO_CANON.get(
                        str(meta.get("primary_exchange") or "").upper(), "NASDAQ"
                    ),
                    "security_type": "COMMON_STOCK",
                    "cik": str(meta["cik"]) if meta.get("cik") else None,
                    "active": bool(meta.get("active", True)),
                    "composite_figi": meta.get("composite_figi"),
                    "source": "massive",
                }
            )
            ck["completed"][key] = {
                "ok": True,
                "rows": bars.height,
                "finished_at": datetime.now(UTC).isoformat(),
            }
        except MassiveHTTPError as exc:
            ck["failures"].append(
                {
                    "ticker": ticker,
                    "status": exc.status,
                    "endpoint": exc.endpoint,
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            ck["completed"][key] = {"ok": False, "status": exc.status}
        _save_checkpoint(ck_path, ck)
        if sleep_s:
            time.sleep(sleep_s)

    inst_path = Paths(root).normalized / dataset / "instruments.parquet"
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    new_inst = pl.DataFrame(inst_rows) if inst_rows else pl.DataFrame()
    if inst_path.exists() and inst_path.stat().st_size > 0:
        prev = pl.read_parquet(inst_path)
        if new_inst.height:
            new_inst = pl.concat([prev, new_inst], how="diagonal_relaxed").unique(
                subset=["instrument_id"], keep="last"
            )
        else:
            new_inst = prev
    if new_inst.height:
        new_inst.write_parquet(inst_path)

    # Optional splits (do not adjust bars).
    try:
        splits = []
        for ticker in ticker_list:
            try:
                splits.extend(
                    list(client.paginate("/v3/reference/splits", {"ticker": ticker, "limit": 100}))
                )
                api_calls += 1
            except MassiveHTTPError:
                splits.extend(
                    list(client.paginate("/stocks/v1/splits", {"ticker": ticker, "limit": 100}))
                )
                api_calls += 1
        sdf = normalize_splits(splits)
        if sdf.height:
            sp = Paths(root).normalized / dataset / "corporate_actions.parquet"
            sdf.write_parquet(sp)
    except MassiveHTTPError:
        ck["meta"]["splits"] = "unavailable"

    elapsed = time.time() - t0
    ck["meta"]["api_calls_last_run"] = api_calls
    ck["meta"]["elapsed_s_last_run"] = elapsed
    _save_checkpoint(ck_path, ck)
    return ck


def ingest_smoke(client: MassiveClient, root: Path) -> dict[str, Any]:
    return ingest_tickers(
        client,
        root,
        dataset="massive-smoke",
        tickers=SMOKE_TICKERS,
        start=SMOKE_START,
        end=SMOKE_END,
    )


def ingest_pilot(
    client: MassiveClient,
    root: Path,
    start: date,
    end: date,
    max_tickers: int = 40,
) -> dict[str, Any]:
    universe = fetch_universe_cs(
        client, asof=start, include_inactive=False, max_results=max_tickers
    )
    # Prefer listed CS; keep a bounded pilot.
    tickers = []
    seen = set()
    for item in universe:
        t = item.get("ticker")
        if not t or t in seen:
            continue
        seen.add(t)
        tickers.append(t)
        if len(tickers) >= max_tickers:
            break
    inst = normalize_tickers(universe[:max_tickers])
    inst_path = Paths(root).normalized / "massive-pilot" / "instruments.parquet"
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    if inst.height:
        inst.write_parquet(inst_path)
    return ingest_tickers(client, root, "massive-pilot", tickers, start, end)


def ingest_full(
    client: MassiveClient,
    root: Path,
    start: date,
    end: date,
    max_tickers: int | None = None,
) -> dict[str, Any]:
    universe = fetch_universe_cs(
        client, asof=start, include_inactive=True, max_results=max_tickers
    )
    tickers = []
    seen = set()
    for item in universe:
        t = item.get("ticker")
        if not t or t in seen:
            continue
        seen.add(t)
        tickers.append(t)
        if max_tickers and len(tickers) >= max_tickers:
            break
    inst = normalize_tickers(universe if max_tickers is None else universe[:max_tickers])
    inst_path = Paths(root).normalized / "massive-full" / "instruments.parquet"
    inst_path.parent.mkdir(parents=True, exist_ok=True)
    if inst.height:
        inst.write_parquet(inst_path)
    return ingest_tickers(client, root, "massive-full", tickers, start, end, sleep_s=12.5)


def status(root: Path, dataset: str) -> dict[str, Any]:
    ck = _load_checkpoint(_checkpoint_path(root, dataset))
    bars_dir = Paths(root).normalized / dataset / "bars_1m"
    files = list(bars_dir.rglob("*.parquet")) if bars_dir.exists() else []
    nbytes = sum(p.stat().st_size for p in files)
    completed = [k for k, v in ck.get("completed", {}).items() if v.get("ok")]
    return {
        "dataset": dataset,
        "completed_jobs": len(completed),
        "failed_jobs": len(ck.get("failures", [])),
        "parquet_files": len(files),
        "bytes": nbytes,
        "data_root": str(Paths(root).data),
        "meta": ck.get("meta", {}),
        "checkpoint": str(_checkpoint_path(root, dataset)),
    }
