"""One-off Starter / Flat Files verification. Never logs credentials."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.client import MassiveClient, MassiveHTTPError, probe_bases
from quant_edge_lab.data.massive.normalize import instrument_id_for, unix_ms_to_utc_naive
from quant_edge_lab.secrets import load_dotenv, massive_api_key

ET = ZoneInfo("America/New_York")
S3_ENDPOINT = "https://files.massive.com"
S3_BUCKET = "flatfiles"
MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1/"


def _ok(client: MassiveClient, path: str, params: dict | None = None) -> dict[str, Any]:
    try:
        payload = client.get(path, params)
        results = payload.get("results")
        n = len(results) if isinstance(results, list) else payload.get("count")
        return {
            "ok": True,
            "status": 200,
            "path": path,
            "result_count": n,
            "adjusted": payload.get("adjusted"),
        }
    except MassiveHTTPError as exc:
        return {"ok": False, "status": exc.status, "path": path, "note": str(exc)}


def rest_burst(client: MassiveClient, n: int = 20) -> dict[str, Any]:
    t0 = time.perf_counter()
    codes: list[int | None] = []
    for _ in range(n):
        try:
            client.get("/v3/reference/tickers", {"ticker": "AAPL", "market": "stocks", "limit": 1})
            codes.append(200)
        except MassiveHTTPError as exc:
            codes.append(exc.status)
    elapsed = time.perf_counter() - t0
    return {
        "requests": n,
        "elapsed_s": round(elapsed, 3),
        "requests_per_minute_equiv": round(n / elapsed * 60, 1) if elapsed else None,
        "http_429": codes.count(429),
        "http_200": codes.count(200),
        "other": {str(c): codes.count(c) for c in set(codes) if c not in {200, 429}},
        "looks_unlimited_vs_basic_5_per_min": codes.count(429) == 0 and elapsed < 30,
    }


def s3_keys_present(root: Path) -> dict[str, bool]:
    import os

    load_dotenv(root)
    return {
        "MASSIVE_S3_ACCESS_KEY": bool(os.environ.get("MASSIVE_S3_ACCESS_KEY", "").strip()),
        "MASSIVE_S3_SECRET_KEY": bool(os.environ.get("MASSIVE_S3_SECRET_KEY", "").strip()),
    }


def _s3_client(*, access: str | None, secret: str | None, unsigned: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError  # noqa: F401

    if unsigned:
        return boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            config=Config(signature_version=UNSIGNED),
        )
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def list_prefix(s3, prefix: str, delimiter: str | None = None, max_keys: int = 50) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"Bucket": S3_BUCKET, "Prefix": prefix, "MaxKeys": max_keys}
    if delimiter:
        kwargs["Delimiter"] = delimiter
    resp = s3.list_objects_v2(**kwargs)
    return {
        "prefix": prefix,
        "key_count": len(resp.get("Contents") or []),
        "common_prefixes": [p.get("Prefix") for p in resp.get("CommonPrefixes") or []],
        "sample_keys": [o.get("Key") for o in (resp.get("Contents") or [])[:5]],
    }


def list_minute_flat_files(s3, prefix: str = MINUTE_PREFIX, max_keys: int = 100_000) -> dict[str, Any]:
    keys: list[tuple[str, int]] = []
    token = None
    pages = 0
    while True:
        kwargs: dict[str, Any] = {"Bucket": S3_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        pages += 1
        for obj in resp.get("Contents") or []:
            keys.append((obj["Key"], int(obj.get("Size") or 0)))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if len(keys) >= max_keys:
            break
    dates = []
    size_by_date: dict[str, int] = {}
    for key, sz in keys:
        name = key.rsplit("/", 1)[-1]
        if name.endswith(".csv.gz") and len(name) >= 14:
            d = name[:10]
            dates.append(d)
            size_by_date[d] = sz
    dates = sorted(set(dates))
    total = sum(sz for _k, sz in keys)
    return {
        "object_count": len(keys),
        "pages": pages,
        "earliest_date": dates[0] if dates else None,
        "latest_date": dates[-1] if dates else None,
        "unique_days": len(dates),
        "dates": dates,
        "size_by_date": size_by_date,
        "total_compressed_bytes": total,
        "total_compressed_gb": round(total / (1024**3), 3) if total else 0,
        "sample_keys": [k for k, _ in keys[:3] + keys[-2:]],
        "whole_market_file_per_day": True,
        "path_convention": prefix + "YYYY/MM/YYYY-MM-DD.csv.gz",
        "prefix": prefix,
    }


def window_start_to_utc(value: int) -> datetime:
    v = int(value)
    digits = len(str(abs(v)))
    if digits >= 18:
        ms = v / 1_000_000
    elif digits >= 16:
        ms = v / 1_000
    else:
        ms = v
    return unix_ms_to_utc_naive(ms)


def normalize_minute_flat_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=5000)
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
    df = df.with_columns(ts_expr.alias("ts_utc"), pl.col("ticker").cast(pl.String))
    out = df.select(
        (pl.lit("ticker:") + pl.col("ticker")).alias("instrument_id"),
        pl.col("ticker"),
        pl.col("ts_utc"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.lit(None).cast(pl.Float64).alias("vwap"),
        (
            pl.col("transactions").cast(pl.Int64)
            if "transactions" in df.columns
            else pl.lit(None).cast(pl.Int64)
        ).alias("transactions"),
        pl.lit("massive_flatfile").alias("source"),
    )
    return out


def run_verify(root: Path) -> dict[str, Any]:
    import os

    load_dotenv(root)
    paths = Paths(root)
    paths.ensure()
    key = massive_api_key(root)
    base = probe_bases(key)
    client = MassiveClient(api_key=key, base_url=base)
    burst_client = MassiveClient(api_key=key, base_url=base, max_retries=1)

    rest = {
        "tickers": _ok(client, "/v3/reference/tickers", {"ticker": "AAPL", "limit": 1, "market": "stocks"}),
        "aggs_1m_recent": _ok(
            client,
            "/v2/aggs/ticker/AAPL/range/1/minute/2026-09-15/2026-09-16",
            {"adjusted": "false", "limit": 50, "sort": "asc"},
        ),
        "aggs_1m_5y": _ok(
            client,
            "/v2/aggs/ticker/AAPL/range/1/minute/2021-10-04/2021-10-05",
            {"adjusted": "false", "limit": 50, "sort": "asc"},
        ),
        "aggs_1m_beyond_5y": _ok(
            client,
            "/v2/aggs/ticker/AAPL/range/1/minute/2019-10-07/2019-10-08",
            {"adjusted": "false", "limit": 50, "sort": "asc"},
        ),
        "aggs_1s": _ok(
            client,
            "/v2/aggs/ticker/AAPL/range/1/second/2026-09-30/2026-09-30",
            {"adjusted": "false", "limit": 50, "sort": "asc"},
        ),
        "snapshot": _ok(client, "/v2/snapshot/locale/us/markets/stocks/tickers/AAPL"),
        "grouped_daily": _ok(
            client,
            "/v2/aggs/grouped/locale/us/market/stocks/2026-09-15",
            {"adjusted": "false"},
        ),
        "splits": _ok(client, "/v3/reference/splits", {"ticker": "AAPL", "limit": 1}),
        "burst": rest_burst(burst_client),
    }

    s3_env = s3_keys_present(root)
    s3_report: dict[str, Any] = {
        "official_mechanism": {
            "endpoint": S3_ENDPOINT,
            "bucket": S3_BUCKET,
            "auth": "S3 Access Key ID + Secret from Massive Dashboard (NOT the REST API key)",
            "docs": "https://massive.com/knowledge-base/article/how-to-get-started-with-s3",
            "minute_prefix_from_docs_and_examples": MINUTE_PREFIX,
            "signature": "s3v4",
        },
        "dashboard_s3_keys_in_env": s3_env,
        "attempts": [],
        "listing": None,
        "download": None,
    }

    # Attempt 1: unsigned list (docs: listing may work without dataset access)
    try:
        unsigned = _s3_client(access=None, secret=None, unsigned=True)
        listing = list_minute_flat_files(unsigned)
        s3_report["attempts"].append({"mode": "unsigned_list", "ok": True, "objects": listing["object_count"]})
        s3_report["listing"] = listing
    except Exception as exc:
        s3_report["attempts"].append({"mode": "unsigned_list", "ok": False, "error_type": type(exc).__name__})

    # Attempt 2: REST API key used as S3 credentials — official docs say this fails.
    try:
        signed_wrong = _s3_client(access=key, secret=key, unsigned=False)
        listing = list_minute_flat_files(signed_wrong)
        s3_report["attempts"].append(
            {"mode": "rest_api_key_as_s3", "ok": True, "objects": listing["object_count"]}
        )
        if s3_report["listing"] is None:
            s3_report["listing"] = listing
    except Exception as exc:
        err = getattr(exc, "response", None)
        code = None
        if err is not None:
            code = err.get("Error", {}).get("Code") if isinstance(err, dict) else getattr(err, "get", lambda *_: None)("Error")
        s3_report["attempts"].append(
            {
                "mode": "rest_api_key_as_s3",
                "ok": False,
                "error_type": type(exc).__name__,
                "matches_official_docs": True,
            }
        )

    access = os.environ.get("MASSIVE_S3_ACCESS_KEY", "").strip()
    secret = os.environ.get("MASSIVE_S3_SECRET_KEY", "").strip()
    download_meta = None
    normalized_path = None
    duckdb_ok = False
    inspect: dict[str, Any] = {}

    if access and secret:
        try:
            s3 = _s3_client(access=access, secret=secret, unsigned=False)
            root_list = list_prefix(s3, "", delimiter="/")
            sip = list_prefix(s3, "us_stocks_sip/", delimiter="/")
            s3_report["root_prefixes"] = root_list.get("common_prefixes")
            s3_report["us_stocks_sip_datasets"] = sip.get("common_prefixes")
            listing = list_minute_flat_files(s3)
            if listing["object_count"] == 0:
                listing = list_minute_flat_files(s3, prefix="stocks/minute-aggregates/")
            s3_report["attempts"].append(
                {"mode": "dashboard_s3_keys", "ok": True, "objects": listing["object_count"]}
            )
            s3_report["listing"] = listing
        except Exception as exc:
            from botocore.exceptions import ClientError

            code = None
            if isinstance(exc, ClientError):
                code = exc.response.get("Error", {}).get("Code")
            s3_report["attempts"].append(
                {
                    "mode": "dashboard_s3_keys",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_code": code,
                }
            )
            listing = None
        latest = (s3_report.get("listing") or {}).get("latest_date") if s3_report.get("listing") else None
        prefix_used = (s3_report.get("listing") or {}).get("prefix") or MINUTE_PREFIX
        if latest:
            y, m, _d = latest.split("-")
            key_name = f"{prefix_used}{y}/{m}/{latest}.csv.gz"
            dest_dir = paths.raw / "massive" / "flatfiles" / "minute_aggs_v1" / y / m
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{latest}.csv.gz"
            s3.download_file(S3_BUCKET, key_name, str(dest))
            raw_size = dest.stat().st_size
            df = pl.read_csv(dest)
            n_tickers = df["ticker"].n_unique() if "ticker" in df.columns else None
            ts_col = "window_start" if "window_start" in df.columns else None
            inspect = {
                "filename": dest.name,
                "s3_key": key_name,
                "path": str(dest),
                "compressed_bytes": raw_size,
                "rows": df.height,
                "columns": df.columns,
                "n_tickers": n_tickers,
                "on_drive_d": str(dest).lower().startswith("d:"),
            }
            if ts_col:
                ts = df[ts_col].cast(pl.Int64)
                inspect["timestamp_column"] = ts_col
                inspect["timestamp_sample"] = int(ts[0]) if df.height else None
                utc0 = window_start_to_utc(int(ts.min()))
                utc1 = window_start_to_utc(int(ts.max()))
                inspect["earliest_ts_utc"] = str(utc0)
                inspect["latest_ts_utc"] = str(utc1)
                et0 = utc0.replace(tzinfo=UTC).astimezone(ET).time()
                et1 = utc1.replace(tzinfo=UTC).astimezone(ET).time()
                inspect["earliest_time_et"] = str(et0)
                inspect["latest_time_et"] = str(et1)
                inspect["premarket_or_extended"] = et0 < datetime.strptime("09:30", "%H:%M").time() or et1 > datetime.strptime("16:00", "%H:%M").time()
            bars = normalize_minute_flat_csv(dest)
            outp = paths.normalized / "massive-flat-smoke" / "bars_1m" / f"date={latest}" / "part.parquet"
            outp.parent.mkdir(parents=True, exist_ok=True)
            bars.write_parquet(outp)
            normalized_path = str(outp)
            import duckdb

            con = duckdb.connect()
            n = con.execute(f"SELECT count(*) FROM read_parquet('{outp.as_posix()}')").fetchone()[0]
            nsym = con.execute(
                f"SELECT count(DISTINCT ticker) FROM read_parquet('{outp.as_posix()}')"
            ).fetchone()[0]
            duckdb_ok = n == bars.height
            inspect["normalized_rows"] = bars.height
            inspect["duckdb_count"] = n
            inspect["duckdb_distinct_tickers"] = nsym
            download_meta = {"ok": True, "s3_key": key_name}
    else:
        s3_report["download"] = {
            "ok": False,
            "blocked": "MASSIVE_S3_ACCESS_KEY / MASSIVE_S3_SECRET_KEY not set",
            "action": "Create S3 keys in Massive Dashboard (Keys / Flat Files), put them in .env. Do not paste them in chat.",
        }

    starter_likely = (
        rest["aggs_1m_5y"].get("ok") is True
        and rest["burst"].get("http_429", 1) == 0
        and rest["burst"].get("looks_unlimited_vs_basic_5_per_min") is True
    )
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "data_root": str(paths.data),
        "rest": rest,
        "starter_rest_signals": {
            "five_year_minute_history": rest["aggs_1m_5y"].get("ok"),
            "beyond_five_year_minute": rest["aggs_1m_beyond_5y"],
            "second_aggregates": rest["aggs_1s"].get("ok"),
            "snapshot": rest["snapshot"].get("ok"),
            "not_basic_5rpm": rest["burst"].get("looks_unlimited_vs_basic_5_per_min"),
            "likely_starter_via_rest": starter_likely,
        },
        "flat_files": s3_report,
        "one_day_test": inspect,
        "download": download_meta,
        "normalized_path": normalized_path,
        "duckdb_ok": duckdb_ok,
        "adjustment_docs": "Minute Flat Files docs do not expose an adjusted flag. REST minute aggs default adjusted=true; we request adjusted=false. Treat Flat Files as as-printed unless Massive metadata says otherwise; keep raw immutable.",
        "websocket": "not_tested_this_pass",
    }
    out = paths.manifests / "massive" / "starter_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
