"""Local OPRA inventory and quote-quality audit. No hypothesis search."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.options_flatfiles import (
    list_local_minute_days,
    load_manifest,
    local_gz_path,
    local_options_root,
    local_product_dirs,
    remote_opra_prefixes,
)
from quant_edge_lab.data.massive.options_occ import parse_occ_ticker
from quant_edge_lab.data.massive.verify_starter import window_start_to_utc

ET = ZoneInfo("America/New_York")

ABSENT_FIELDS = (
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mid",
    "nbbo",
    "exchange",
    "venue",
    "condition",
    "sequence",
    "correction",
    "cancel",
    "open_interest",
    "implied_volatility",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "underlying_price",
    "greeks",
)


def _read_csv_gz(path: Path, n_rows: int | None = None) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=5000, n_rows=n_rows)


def _normalize_ts(df: pl.DataFrame) -> pl.DataFrame:
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(rename)
    ts_col = "window_start" if "window_start" in df.columns else "t"
    sample = int(df[ts_col][0])
    digits = len(str(abs(sample)))
    ns = pl.col(ts_col).cast(pl.Int64)
    if digits >= 18:
        ts_expr = (ns / 1_000).cast(pl.Datetime("us"))
    elif digits >= 16:
        ts_expr = ns.cast(pl.Datetime("us"))
    else:
        ts_expr = (ns * 1_000).cast(pl.Datetime("us"))
    return df.with_columns(ts_expr.alias("ts_utc"))


def _parse_root_frame(tickers: pl.Series) -> pl.DataFrame:
    rows = []
    for t in tickers.to_list():
        rec = parse_occ_ticker(str(t))
        if rec is None:
            rows.append(
                {
                    "ticker": t,
                    "underlying_root": None,
                    "expiration": None,
                    "call_put": None,
                    "strike": None,
                    "parse_ok": False,
                }
            )
        else:
            rows.append(
                {
                    "ticker": t,
                    "underlying_root": rec["underlying_root"],
                    "expiration": rec["expiration"],
                    "call_put": rec["call_put"],
                    "strike": rec["strike"],
                    "parse_ok": True,
                }
            )
    return pl.DataFrame(rows)


def inventory_local(root: Path) -> dict[str, Any]:
    products = local_product_dirs(root)
    days = list_local_minute_days(root)
    man = load_manifest(root)
    files = man.get("files") or {}
    sizes = []
    missing_on_disk = []
    for day in days:
        p = local_gz_path(root, day)
        if p.exists():
            sizes.append({"date": day, "bytes": p.stat().st_size})
        else:
            missing_on_disk.append(day)
    total = sum(s["bytes"] for s in sizes)
    return {
        "local_root": str(local_options_root(root)),
        "products_present": products,
        "products_absent": [k for k, v in products.items() if not v],
        "minute_days": days,
        "n_minute_days": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "compressed_bytes": total,
        "compressed_gb": round(total / (1024**3), 3),
        "median_file_bytes": sorted(s["bytes"] for s in sizes)[len(sizes) // 2] if sizes else None,
        "manifest_path": str(
            Paths(root).manifests / "massive" / "flatfiles" / "options_minute_aggs_v1_manifest.json"
        ),
        "manifest_file_count": len(files),
        "manifest_meta": man.get("meta") or {},
        "missing_on_disk": missing_on_disk[:20],
        "other_opra_files": [
            str(p.relative_to(local_options_root(root)))
            for p in local_options_root(root).rglob("*")
            if p.is_file() and "minute_aggs_v1" not in p.as_posix()
        ][:50],
    }


def schema_sample(root: Path, day: str | None = None) -> dict[str, Any]:
    days = list_local_minute_days(root)
    if not days:
        return {"ok": False, "reason": "no local minute files"}
    day = day or days[len(days) // 2]
    path = local_gz_path(root, day)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip()
        first = fh.readline().strip()
    df = _read_csv_gz(path, n_rows=5)
    cols = [c.lower() for c in df.columns]
    ts_col = "window_start" if "window_start" in cols else ("t" if "t" in cols else None)
    ts_meta = None
    if ts_col:
        raw = int(df.rename({c: c.lower() for c in df.columns})[ts_col][0])
        utc = window_start_to_utc(raw)
        et = utc.replace(tzinfo=UTC).astimezone(ET)
        ts_meta = {
            "column": ts_col,
            "raw_value": raw,
            "raw_digits": len(str(abs(raw))),
            "interpreted_utc": utc.isoformat(),
            "interpreted_america_new_york": et.isoformat(),
            "precision_note": "unix epoch; digit length selects ns/us/ms (stock-flat rule)",
        }
    present = set(cols)
    return {
        "ok": True,
        "sample_day": day,
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "header_line": header,
        "first_data_line": first,
        "columns": cols,
        "dtypes": {c: str(t) for c, t in zip(df.columns, df.dtypes, strict=True)},
        "timestamp": ts_meta,
        "fields_present": sorted(present),
        "fields_absent_from_this_file": [f for f in ABSENT_FIELDS if f not in present],
        "granularity": "minute_aggregate",
        "event_level": False,
        "quote_level": False,
        "trade_prints": False,
    }


def day_quality(root: Path, day: str) -> dict[str, Any]:
    path = local_gz_path(root, day)
    raw = _read_csv_gz(path)
    n = raw.height
    df = _normalize_ts(raw)
    vol_col = "volume" if "volume" in df.columns else None
    tx_col = "transactions" if "transactions" in df.columns else None
    zero_vol = float((df[vol_col] == 0).mean()) if vol_col else None
    zero_tx = float((df[tx_col] == 0).mean()) if tx_col else None
    ohlc_bad = None
    if all(c in df.columns for c in ("open", "high", "low", "close")):
        ohlc_bad = float(
            (
                (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            ).mean()
        )
    unique_tickers = df["ticker"].n_unique()
    sample_t = df["ticker"].unique().head(8000)
    parsed = _parse_root_frame(sample_t)
    parse_ok = float(parsed["parse_ok"].mean()) if parsed.height else None
    okp = parsed.filter(pl.col("parse_ok"))
    underlyings = int(okp["underlying_root"].n_unique()) if okp.height else 0
    expiries = int(okp["expiration"].n_unique()) if okp.height else 0
    ts_min = str(df["ts_utc"].min())
    ts_max = str(df["ts_utc"].max())
    bars_per = df.group_by("ticker").len()
    return {
        "day": day,
        "rows": n,
        "unique_contracts": unique_tickers,
        "zero_volume_fraction": zero_vol,
        "zero_transactions_fraction": zero_tx,
        "ohlc_violation_fraction": ohlc_bad,
        "parse_ok_fraction_sample": parse_ok,
        "parse_sample_n": parsed.height,
        "unique_underlyings_in_parse_sample": underlyings,
        "unique_expirations_in_parse_sample": expiries,
        "median_minute_bars_per_contract": float(bars_per["len"].median()),
        "ts_utc_min": ts_min,
        "ts_utc_max": ts_max,
        "cannot_measure": [
            "bid>0 fraction",
            "ask>bid fraction",
            "locked/crossed",
            "dollar spread",
            "percent spread",
            "spread by price bucket",
            "quote update frequency",
            "stale quote prevalence",
            "NBBO status",
        ],
    }


def structure_sample(root: Path, day: str, underlying: str = "AAPL") -> dict[str, Any]:
    path = local_gz_path(root, day)
    df = _normalize_ts(_read_csv_gz(path))
    prefix = f"O:{underlying}"
    cand = df.filter(pl.col("ticker").cast(pl.String).str.starts_with(prefix))
    if cand.height == 0:
        return {"day": day, "underlying": underlying, "rows": 0}
    parsed = _parse_root_frame(cand["ticker"].unique())
    joined = cand.join(parsed, on="ticker", how="left")
    sub = joined.filter(pl.col("underlying_root") == underlying)
    if sub.height == 0:
        return {"day": day, "underlying": underlying, "rows": 0}
    n_exp = sub["expiration"].n_unique()
    by_exp = (
        sub.group_by(["expiration", "call_put"])
        .agg(
            pl.col("ticker").n_unique().alias("contracts"),
            pl.col("strike").n_unique().alias("strikes"),
        )
        .sort("expiration")
    )
    return {
        "day": day,
        "underlying": underlying,
        "rows": sub.height,
        "contracts": sub["ticker"].n_unique(),
        "expirations": n_exp,
        "call_put_counts": sub.group_by("call_put").agg(pl.len().alias("n")).to_dicts(),
        "sample_expiries": by_exp.head(12).to_dicts(),
    }


def pit_field_table(observed_columns: list[str]) -> list[dict[str, str]]:
    cols = {c.lower() for c in observed_columns}
    rows = []

    def add(field: str, cls: str, value_time: str, first_available_at: str, note: str) -> None:
        rows.append(
            {
                "field": field,
                "class": cls,
                "value_time": value_time,
                "first_available_at": first_available_at,
                "note": note,
            }
        )

    if "window_start" in cols or "t" in cols:
        add(
            "window_start/ts_utc",
            "A",
            "minute bar start (UTC epoch)",
            "end of that minute (same convention as stock flats: +1 minute)",
            "Present. Minute aggregate clock, not quote event time.",
        )
    for px in ("open", "high", "low", "close"):
        if px in cols:
            add(
                px,
                "A",
                "inside the minute window",
                "end of minute",
                "Trade-aggregate OHLC for the option contract. Not NBBO.",
            )
    if "volume" in cols:
        add("volume", "A", "trades during the minute", "end of minute", "Present.")
    if "transactions" in cols:
        add("transactions", "A", "trades during the minute", "end of minute", "Present.")
    if "ticker" in cols:
        add(
            "ticker (OCC)",
            "A",
            "contract identity on that file date",
            "same as row",
            "Contemporaneous OSI/OCC in the daily file. Root is not issuer identity.",
        )
    add("bid/ask/NBBO", "D", "unknown", "unknown", "Not in local files.")
    add(
        "open_interest",
        "D",
        "unknown",
        "unknown",
        "Not in local files. Typically EOD and lagged if obtained later.",
    )
    add(
        "implied_volatility",
        "D",
        "unknown",
        "unknown",
        "Not in local files. Vendor IV is usually C or D.",
    )
    add("greeks", "D", "unknown", "unknown", "Not in local files.")
    add(
        "underlying_price",
        "B",
        "stock minute bar (separate dataset)",
        "end of that stock minute",
        "Must join stock minute bars. Not on the option row.",
    )
    add(
        "corporate_actions / contract_adjustments",
        "D",
        "unknown",
        "unknown",
        "Not on option minute files. OCC strike is contemporaneous OSI.",
    )
    return rows


def underlying_linkage_note() -> dict[str, Any]:
    return {
        "option_id": "OCC ticker in the minute file (O:ROOT+YYMMDD+C/P+strike*1000)",
        "underlying_join_key_available": "parsed OSI root (e.g. AAPL)",
        "stock_id_scheme": "ticker:<symbol> on stock minute bars / V5 events",
        "pit_risks": [
            "OSI root is not a permanent issuer id (FB vs META, ticker reuse).",
            "instruments.parquet is last-snapshot listing, not session-dated PIT.",
            "Option files do not carry FIGI/CIK.",
            "Timestamp join must use the same ET session clock as stock bars.",
            "OCC strike is the listed OSI strike that day, not split-backadjusted.",
        ],
        "usable_join": (
            "Align option minute close to stock minute close when OSI root "
            "equals the stock ticker and that ticker did not change."
        ),
    }


def run_options_audit(root: Path, *, sample_underlying: str = "AAPL") -> dict[str, Any]:
    inv = inventory_local(root)
    schema = schema_sample(root)
    quality = None
    structure = None
    days = inv.get("minute_days") or []
    if days:
        mid = days[len(days) // 2]
        quality = day_quality(root, mid)
        structure = structure_sample(root, mid, sample_underlying)
    remote = remote_opra_prefixes(root)
    observed = schema.get("columns") or []
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "inventory": inv,
        "schema": schema,
        "day_quality": quality,
        "structure_sample": structure,
        "pit": pit_field_table(observed),
        "underlying_linkage": underlying_linkage_note(),
        "remote_list": remote,
        "readiness": _readiness(inv, schema),
    }


def _readiness(inv: dict[str, Any], schema: dict[str, Any]) -> str:
    if not inv.get("n_minute_days"):
        return "NOT_READY"
    if inv.get("products_present", {}).get("quotes_v1"):
        return "READY_FOR_OPTIONS_EXPERIMENT"
    if schema.get("ok") and inv["n_minute_days"] >= 20:
        return "READY_WITH_LIMITATIONS"
    return "NOT_READY"
