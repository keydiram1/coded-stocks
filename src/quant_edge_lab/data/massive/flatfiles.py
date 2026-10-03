"""Massive Stocks minute Flat Files: access tests, resumable sync, conversion.

Never logs credentials. Writes only under QUANT_EDGE_DATA_ROOT (D:).
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from botocore.exceptions import ClientError

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.verify_starter import (
    MINUTE_PREFIX,
    S3_BUCKET,
    _s3_client,
    list_minute_flat_files,
    normalize_minute_flat_csv,
)
from quant_edge_lab.secrets import load_dotenv

DATASET = "massive-flat"
MANIFEST_NAME = "minute_aggs_v1_manifest.json"


def s3_from_env(root: Path):
    load_dotenv(root)
    access = os.environ.get("MASSIVE_S3_ACCESS_KEY", "").strip()
    secret = os.environ.get("MASSIVE_S3_SECRET_KEY", "").strip()
    if not access or not secret:
        raise RuntimeError("MASSIVE_S3_ACCESS_KEY / MASSIVE_S3_SECRET_KEY missing from .env")
    return _s3_client(access=access, secret=secret, unsigned=False)


def object_key(day: str, prefix: str = MINUTE_PREFIX) -> str:
    y, m, _ = day.split("-")
    p = prefix if prefix.endswith("/") else prefix + "/"
    return f"{p}{y}/{m}/{day}.csv.gz"


def local_gz_path(root: Path, day: str) -> Path:
    y, m, _ = day.split("-")
    return Paths(root).raw / "massive" / "flatfiles" / "minute_aggs_v1" / y / m / f"{day}.csv.gz"


def local_parquet_path(root: Path, day: str) -> Path:
    return Paths(root).normalized / DATASET / "bars_1m" / f"date={day}" / "part.parquet"


def manifest_path(root: Path) -> Path:
    return Paths(root).manifests / "massive" / "flatfiles" / MANIFEST_NAME


def load_manifest(root: Path) -> dict[str, Any]:
    p = manifest_path(root)
    if not p.exists():
        return {"files": {}, "meta": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(root: Path, data: dict[str, Any]) -> None:
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def http_status_from_client_error(exc: ClientError) -> int:
    return int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)


def probe_get(s3, day: str, dest: Path | None, *, keep: bool) -> dict[str, Any]:
    """Actual GET. Range probes for search; full object GET for named tests."""
    key = object_key(day)
    t0 = time.perf_counter()
    try:
        if dest is None:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key, Range="bytes=0-65535")
            body = resp["Body"].read()
            http = int(resp.get("ResponseMetadata", {}).get("HTTPStatusCode") or 206)
            return {
                "date": day,
                "key": key,
                "get_ok": True,
                "http": http,
                "bytes_read": len(body),
                "mode": "range",
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        s3.download_file(S3_BUCKET, key, str(part))
        size = part.stat().st_size
        if size < 100 or not gzip_ok(part):
            part.unlink(missing_ok=True)
            return {
                "date": day,
                "key": key,
                "get_ok": False,
                "http": 200,
                "error_code": "INVALID_BODY",
                "local_bytes": size,
            }
        part.replace(dest)
        ok = {
            "date": day,
            "key": key,
            "get_ok": True,
            "http": 200,
            "local_bytes": size,
            "path": str(dest),
            "mode": "full",
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
        if not keep and dest.exists():
            dest.unlink()
            ok["deleted_after_test"] = True
        return ok
    except ClientError as exc:
        return {
            "date": day,
            "key": key,
            "get_ok": False,
            "http": http_status_from_client_error(exc),
            "error_code": exc.response.get("Error", {}).get("Code"),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }


def nearest_listed_day(dates: list[str], target: str) -> str:
    if target in dates:
        return target
    after = [d for d in dates if d >= target]
    before = [d for d in dates if d <= target]
    if after:
        return after[0]
    return before[-1]


def find_earliest_get(s3, dates: list[str]) -> dict[str, Any]:
    """Binary search first date that GET succeeds. Does not keep files."""
    lo, hi = 0, len(dates) - 1
    earliest = None
    probes = []
    while lo <= hi:
        mid = (lo + hi) // 2
        day = dates[mid]
        r = probe_get(s3, day, dest=None, keep=False)
        probes.append({"date": day, "get_ok": r["get_ok"], "http": r.get("http"), "code": r.get("error_code")})
        if r["get_ok"]:
            earliest = day
            hi = mid - 1
        else:
            lo = mid + 1
    latest_ok = dates[-1] if dates else None
    if latest_ok:
        r = probe_get(s3, latest_ok, dest=None, keep=False)
        probes.append({"date": latest_ok, "get_ok": r["get_ok"], "http": r.get("http"), "code": r.get("error_code")})
        if not r["get_ok"]:
            latest_ok = None
    return {"earliest_get_ok": earliest, "latest_get_ok": latest_ok, "probes": probes}


def run_access_boundary(root: Path) -> dict[str, Any]:
    s3 = s3_from_env(root)
    listing = list_minute_flat_files(s3)
    dates = listing.get("dates") or []
    named = {
        "around_2021_10": nearest_listed_day(dates, "2021-10-01"),
        "around_2020_10": nearest_listed_day(dates, "2020-10-01"),
        "around_2003": nearest_listed_day(dates, "2003-09-10"),
    }
    tmp = Paths(root).raw / "massive" / "flatfiles" / "_access_probes"
    named_results = {}
    for label, day in named.items():
        dest = tmp / f"{day}.csv.gz"
        named_results[label] = probe_get(s3, day, dest=dest, keep=False)
    bound = find_earliest_get(s3, dates)
    if tmp.exists():
        for leftover in tmp.glob("*"):
            leftover.unlink()
        tmp.rmdir()
    years = None
    if bound["earliest_get_ok"] and bound["latest_get_ok"]:
        d0 = date.fromisoformat(bound["earliest_get_ok"])
        d1 = date.fromisoformat(bound["latest_get_ok"])
        years = round((d1 - d0).days / 365.25, 2)
    report = {
        "list_earliest": listing.get("earliest_date"),
        "list_latest": listing.get("latest_date"),
        "list_unique_days": listing.get("unique_days"),
        "named_get_tests": named_results,
        "get_boundary": bound,
        "accessible_span_years": years,
        "looks_like_starter_5y": bool(years is not None and 4.5 <= years <= 5.5),
        "prefix": MINUTE_PREFIX,
    }
    out = Paths(root).manifests / "massive" / "flatfiles" / "access_boundary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def gzip_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            f.read(64)
        return True
    except OSError:
        return False


def convert_day(root: Path, day: str) -> dict[str, Any]:
    gz = local_gz_path(root, day)
    out = local_parquet_path(root, day)
    if out.exists() and out.stat().st_size > 0:
        df = pl.scan_parquet(out).select(
            pl.len().alias("n"),
            pl.col("ticker").n_unique().alias("tickers"),
        ).collect()
        return {"ok": True, "skipped": True, "rows": int(df["n"][0]), "tickers": int(df["tickers"][0]), "path": str(out)}
    bars = normalize_minute_flat_csv(gz)
    if "vwap" in bars.columns:
        bars = bars.drop("vwap")
    bars = bars.with_columns(pl.lit(day).alias("session_date"))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    bars.write_parquet(tmp, compression="zstd")
    tmp.replace(out)
    return {
        "ok": True,
        "skipped": False,
        "rows": bars.height,
        "tickers": bars["ticker"].n_unique(),
        "path": str(out),
    }


_lock = threading.Lock()
_convert_sem = threading.Semaphore(2)


def _download_one(root: Path, day: str, remote_size: int) -> dict[str, Any]:
    s3 = s3_from_env(root)
    dest = local_gz_path(root, day)
    key = object_key(day)
    if dest.exists() and dest.stat().st_size == remote_size and gzip_ok(dest):
        return {"date": day, "status": "skipped_verified", "local_bytes": dest.stat().st_size, "remote_bytes": remote_size}
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, 5):
        try:
            s3.download_file(S3_BUCKET, key, str(part))
            size = part.stat().st_size
            if size != remote_size:
                last_err = f"size {size} != {remote_size}"
                part.unlink(missing_ok=True)
                continue
            if not gzip_ok(part):
                last_err = "gzip_invalid"
                part.unlink(missing_ok=True)
                continue
            part.replace(dest)
            return {
                "date": day,
                "status": "downloaded",
                "local_bytes": size,
                "remote_bytes": remote_size,
                "attempts": attempt,
            }
        except Exception as exc:
            last_err = type(exc).__name__
            part.unlink(missing_ok=True)
            time.sleep(min(30, 2**attempt))
    return {"date": day, "status": "failed", "error": last_err, "remote_bytes": remote_size}


def sync_range(
    root: Path,
    start: str,
    end: str,
    *,
    workers: int = 6,
    convert: bool = True,
) -> dict[str, Any]:
    s3 = s3_from_env(root)
    listing = list_minute_flat_files(s3)
    sizes: dict[str, int] = {str(k): int(v) for k, v in listing.get("size_by_date", {}).items()}
    days = [d for d in sorted(sizes) if start <= d <= end]
    man = load_manifest(root)
    man.setdefault("files", {})
    man["meta"] = {
        "prefix": MINUTE_PREFIX,
        "accessible_start": start,
        "accessible_end": end,
        "expected_days": len(days),
        "expected_compressed_bytes": sum(sizes[d] for d in days),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    save_manifest(root, man)
    t0 = time.perf_counter()
    downloaded = skipped = failed = 0
    converted = 0
    rows = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, root, d, sizes[d]): d for d in days}
        for fut in as_completed(futs):
            day = futs[fut]
            rec = fut.result()
            rec["s3_key"] = object_key(day)
            rec["local_path"] = str(local_gz_path(root, day))
            rec["download_ts"] = datetime.now(UTC).isoformat()
            rec["remote_compressed_size"] = sizes[day]
            if rec["status"] == "failed":
                failed += 1
                rec["validation_status"] = "failed"
            else:
                rec["validation_status"] = "gzip_ok"
                if rec["status"] == "skipped_verified":
                    skipped += 1
                else:
                    downloaded += 1
                if convert:
                    try:
                        with _convert_sem:
                            cres = convert_day(root, day)
                        rec["conversion_status"] = "ok"
                        rec["row_count"] = cres["rows"]
                        rec["distinct_ticker_count"] = cres["tickers"]
                        converted += 1
                        rows += cres["rows"]
                    except Exception as exc:
                        rec["conversion_status"] = "failed"
                        rec["error"] = type(exc).__name__
            with _lock:
                man = load_manifest(root)
                man["files"][day] = rec
                save_manifest(root, man)
    elapsed = time.perf_counter() - t0
    man = load_manifest(root)
    man["meta"].update(
        {
            "downloaded": downloaded,
            "skipped_verified": skipped,
            "failed": failed,
            "converted": converted,
            "rows_converted_this_run": rows,
            "elapsed_s": elapsed,
            "finished_at": datetime.now(UTC).isoformat(),
        }
    )
    save_manifest(root, man)
    return man["meta"] | {"start": start, "end": end, "days": len(days)}


def flatfiles_status(root: Path) -> dict[str, Any]:
    man = load_manifest(root)
    files = man.get("files", {})
    meta = man.get("meta", {})
    done = [v for v in files.values() if v.get("status") in {"downloaded", "skipped_verified"}]
    conv = [v for v in files.values() if v.get("conversion_status") == "ok"]
    fail = [v for v in files.values() if v.get("status") == "failed" or v.get("conversion_status") == "failed"]
    local_bytes = sum(int(v.get("local_bytes") or v.get("remote_compressed_size") or 0) for v in done)
    rows = sum(int(v.get("row_count") or 0) for v in conv)
    expected_n = int(meta.get("expected_days") or 0)
    remaining = max(0, expected_n - len(done))
    stamps = sorted(str(v.get("download_ts")) for v in done if v.get("download_ts"))
    elapsed = float(meta.get("elapsed_s") or 0)
    if stamps and not elapsed:
        t0 = datetime.fromisoformat(stamps[0])
        t1 = datetime.fromisoformat(stamps[-1])
        elapsed = max(0.001, (t1 - t0).total_seconds())
    rate = (len(done) / elapsed) if elapsed else None
    eta = (remaining / rate) if rate else None
    return {
        "dataset": DATASET,
        "data_root": str(Paths(root).data),
        "accessible_start": meta.get("accessible_start"),
        "accessible_end": meta.get("accessible_end"),
        "expected_daily_files": expected_n,
        "downloaded_files": len(done),
        "converted_files": len(conv),
        "failed_files": len(fail),
        "compressed_gb_downloaded": round(local_bytes / (1024**3), 3),
        "expected_compressed_gb": round(int(meta.get("expected_compressed_bytes") or 0) / (1024**3), 3),
        "rows_converted": rows,
        "files_per_second": round(rate, 4) if rate else None,
        "eta_remaining_s": round(eta, 1) if eta else None,
        "prefix": MINUTE_PREFIX,
        "manifest": str(manifest_path(root)),
    }
