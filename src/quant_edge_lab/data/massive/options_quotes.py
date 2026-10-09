"""OPRA quotes_v1 inventory and entitlement. LIST first. No bulk download. No science."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
from botocore.exceptions import ClientError

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import s3_from_env
from quant_edge_lab.data.massive.verify_starter import (
    S3_BUCKET,
    list_minute_flat_files,
    list_prefix,
)

QUOTES_PREFIX = "us_options_opra/quotes_v1/"
AUDIT_DAY = "2025-10-08"
# One quotes day is ~100+ GB compressed. Never GET a full object above this.
MAX_FULL_GET_BYTES = 8 * 1024**3
DEFAULT_PREFIX_BYTES = 32 * 1024 * 1024


def quotes_object_key(day: str) -> str:
    y, m, _ = day.split("-")
    return f"{QUOTES_PREFIX}{y}/{m}/{day}.csv.gz"


def quotes_prefix_path(root: Path, day: str) -> Path:
    return (
        Paths(root).raw
        / "massive"
        / "flatfiles"
        / "us_options_opra"
        / "quotes_v1"
        / "_prefix"
        / f"{day}.csv.gz.prefix"
    )


def size_distribution(sizes: list[int]) -> dict[str, int | None]:
    if not sizes:
        return {"min": None, "median": None, "p90": None, "max": None, "n": 0}
    arr = np.asarray(sizes, dtype=np.int64)
    return {
        "min": int(arr.min()),
        "median": int(np.median(arr)),
        "p90": int(np.quantile(arr, 0.90)),
        "max": int(arr.max()),
        "n": int(arr.size),
    }


def list_quotes_inventory(root: Path) -> dict[str, Any]:
    """S3 LIST only. Does not GET quote objects."""
    s3 = s3_from_env(root)
    listing = list_minute_flat_files(s3, prefix=QUOTES_PREFIX)
    sizes = [int(v) for v in (listing.get("size_by_date") or {}).values()]
    dist = size_distribution(sizes)
    day_size = (listing.get("size_by_date") or {}).get(AUDIT_DAY)
    return {
        "mode": "LIST_ONLY",
        "prefix": QUOTES_PREFIX,
        "key_convention": QUOTES_PREFIX + "YYYY/MM/YYYY-MM-DD.csv.gz",
        "earliest_date": listing.get("earliest_date"),
        "latest_date": listing.get("latest_date"),
        "n_files": listing.get("object_count"),
        "unique_days": listing.get("unique_days"),
        "total_compressed_bytes": listing.get("total_compressed_bytes"),
        "total_compressed_gb": listing.get("total_compressed_gb"),
        "size_bytes": dist,
        "audit_day": AUDIT_DAY,
        "audit_day_present": AUDIT_DAY in (listing.get("size_by_date") or {}),
        "audit_day_compressed_bytes": int(day_size) if day_size is not None else None,
        "audit_day_key": quotes_object_key(AUDIT_DAY),
        "sample_keys": listing.get("sample_keys"),
        "root_prefixes": list_prefix(s3, "us_options_opra/", delimiter="/"),
    }


def probe_quotes_get(root: Path, day: str = AUDIT_DAY) -> dict[str, Any]:
    """HEAD + Range GET. Never downloads a full quotes day."""
    s3 = s3_from_env(root)
    key = quotes_object_key(day)
    out: dict[str, Any] = {
        "day": day,
        "key": key,
        "max_full_get_bytes": MAX_FULL_GET_BYTES,
        "full_get_attempted": False,
    }
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=key)
        out["head"] = {
            "ok": True,
            "http": int(head.get("ResponseMetadata", {}).get("HTTPStatusCode") or 200),
            "content_length": int(head.get("ContentLength") or 0),
        }
    except ClientError as exc:
        out["head"] = {
            "ok": False,
            "http": int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0),
            "error_code": exc.response.get("Error", {}).get("Code"),
        }
    size = (out.get("head") or {}).get("content_length")
    out["full_object_too_large"] = bool(size and size > MAX_FULL_GET_BYTES)
    try:
        resp = s3.get_object(
            Bucket=S3_BUCKET,
            Key=key,
            Range=f"bytes=0-{DEFAULT_PREFIX_BYTES - 1}",
        )
        body = resp["Body"].read()
        dest = quotes_prefix_path(root, day)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        out["range_get"] = {
            "ok": True,
            "http": int(resp.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0),
            "bytes_read": len(body),
            "content_range": resp.get("ContentRange"),
            "prefix_path": str(dest),
            "note": "prefix bytes only; not a complete csv.gz",
        }
        out["schema_from_prefix"] = _schema_from_gzip_prefix(body)
    except ClientError as exc:
        out["range_get"] = {
            "ok": False,
            "http": int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0),
            "error_code": exc.response.get("Error", {}).get("Code"),
        }
        out["schema_from_prefix"] = None
    return out


def _schema_from_gzip_prefix(body: bytes) -> dict[str, Any]:
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    partial = dec.decompress(body)
    text = partial.decode("utf-8", errors="replace")
    nl = text.rfind("\n")
    if nl >= 0:
        text = text[:nl]
    lines = text.splitlines()
    header = lines[0] if lines else ""
    cols = [c.strip() for c in header.split(",")] if header else []
    return {
        "header_line": header,
        "columns": cols,
        "n_complete_lines": len(lines),
        "sample_rows": lines[1:4],
        "partial_uncompressed_bytes": len(partial),
    }


def contrast_entitled_aggregates(root: Path, day: str = AUDIT_DAY) -> dict[str, Any]:
    """GET-ability of other OPRA products. Not a quotes substitute."""
    s3 = s3_from_env(root)
    y, m, _ = day.split("-")
    out = {}
    for name, pfx in (
        ("minute_aggs_v1", "us_options_opra/minute_aggs_v1/"),
        ("day_aggs_v1", "us_options_opra/day_aggs_v1/"),
        ("trades_v1", "us_options_opra/trades_v1/"),
    ):
        key = f"{pfx}{y}/{m}/{day}.csv.gz"
        rec: dict[str, Any] = {"key": key}
        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=key)
            rec["head_ok"] = True
            rec["http"] = 200
            rec["content_length"] = int(head.get("ContentLength") or 0)
        except ClientError as exc:
            rec["head_ok"] = False
            rec["http"] = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)
            rec["error_code"] = exc.response.get("Error", {}).get("Code")
        out[name] = rec
    return out


def storage_plan(audit_day_bytes: int | None, dist: dict[str, Any]) -> dict[str, Any]:
    median = dist.get("median")
    n = dist.get("n") or 0

    def gb(n_days: int, per: int | None) -> float | None:
        if not per:
            return None
        return round(n_days * per / (1024**3), 3)

    day_b = audit_day_bytes
    return {
        "audit_day_compressed_gb": round(day_b / (1024**3), 3) if day_b else None,
        "uncompressed_gb_day": None,
        "uncompressed_note": "not observed; full object was not downloaded",
        "rows_day": None,
        "rows_note": "not observed; GET quotes_v1 returned 403",
        "compressed_gb_20d_at_median": gb(20, median),
        "compressed_gb_100d_at_median": gb(100, median),
        "compressed_gb_252d_at_median": gb(252, median),
        "compressed_gb_available_history_at_n_files": gb(int(n), median),
        "recommendation": [
            "do_not_download_full_quotes_history",
            "do_not_download_even_one_full_day_at_100GB_class",
            "keep_any_future_raw_gzip_immutable",
            "if_entitled_later_normalize_selected_columns_to_date_partitioned_parquet",
            "filter_contract_universe_during_derived_conversion",
        ],
    }
