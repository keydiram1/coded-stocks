"""OPRA options minute Flat Files. Raw csv.gz under QUANT_EDGE_DATA_ROOT. No science."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import gzip_ok, s3_from_env
from quant_edge_lab.data.massive.verify_starter import (
    S3_BUCKET,
    list_minute_flat_files,
    list_prefix,
)

OPTIONS_PREFIX = "us_options_opra/minute_aggs_v1/"
KNOWN_OPRA_PREFIXES = (
    "us_options_opra/minute_aggs_v1/",
    "us_options_opra/day_aggs_v1/",
    "us_options_opra/quotes_v1/",
    "us_options_opra/trades_v1/",
)
MANIFEST_NAME = "options_minute_aggs_v1_manifest.json"
DATASET = "massive-options-minute"


def local_options_root(root: Path) -> Path:
    return Paths(root).raw / "massive" / "flatfiles" / "us_options_opra"


def local_gz_path(root: Path, day: str) -> Path:
    y, m, _ = day.split("-")
    return local_options_root(root) / "minute_aggs_v1" / y / m / f"{day}.csv.gz"


def object_key(day: str) -> str:
    y, m, _ = day.split("-")
    return f"{OPTIONS_PREFIX}{y}/{m}/{day}.csv.gz"


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


def local_product_dirs(root: Path) -> dict[str, bool]:
    base = local_options_root(root)
    out: dict[str, bool] = {}
    names = (
        "minute_aggs_v1",
        "day_aggs_v1",
        "quotes_v1",
        "trades_v1",
        "nbbo_v1",
        "underlying_quotes_v1",
    )
    for name in names:
        p = base / name
        out[name] = p.is_dir() and any(p.rglob("*.csv.gz"))
    return out


def list_local_minute_days(root: Path) -> list[str]:
    base = local_options_root(root) / "minute_aggs_v1"
    days: list[str] = []
    if not base.exists():
        return days
    for p in base.rglob("*.csv.gz"):
        stem = p.name[:10]
        if len(stem) == 10 and stem[4] == "-":
            days.append(stem)
    return sorted(set(days))


_lock = threading.Lock()


def _download_one(root: Path, day: str, remote_size: int) -> dict[str, Any]:
    s3 = s3_from_env(root)
    dest = local_gz_path(root, day)
    key = object_key(day)
    if dest.exists() and dest.stat().st_size == remote_size and gzip_ok(dest):
        return {
            "date": day,
            "status": "skipped_verified",
            "local_bytes": dest.stat().st_size,
            "remote_bytes": remote_size,
        }
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for _attempt in range(1, 5):
        try:
            s3.download_file(S3_BUCKET, key, str(part))
            size = part.stat().st_size
            if size != remote_size:
                last_err = f"size {size} != {remote_size}"
                part.unlink(missing_ok=True)
                continue
            if not gzip_ok(part):
                last_err = "INVALID_BODY"
                part.unlink(missing_ok=True)
                continue
            part.replace(dest)
            return {
                "date": day,
                "status": "downloaded",
                "local_bytes": size,
                "remote_bytes": remote_size,
            }
        except Exception as exc:
            last_err = type(exc).__name__
            part.unlink(missing_ok=True)
    return {"date": day, "status": "failed", "error": last_err, "remote_bytes": remote_size}


def sync_options_range(root: Path, start: str, end: str, workers: int = 4) -> dict[str, Any]:
    """Download OPRA minute aggregates as raw csv.gz. Does not convert. Does not run science."""
    s3 = s3_from_env(root)
    listing = list_minute_flat_files(s3, prefix=OPTIONS_PREFIX)
    sizes: dict[str, int] = {str(k): int(v) for k, v in listing.get("size_by_date", {}).items()}
    days = [d for d in sorted(sizes) if start <= d <= end]
    man = load_manifest(root)
    man.setdefault("files", {})
    man["meta"] = {
        "prefix": OPTIONS_PREFIX,
        "accessible_start": start,
        "accessible_end": end,
        "expected_days": len(days),
        "expected_compressed_bytes": sum(sizes[d] for d in days),
        "updated_at": datetime.now(UTC).isoformat(),
        "convert": False,
    }
    save_manifest(root, man)
    t0 = time.perf_counter()
    downloaded = skipped = failed = 0
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
            elif rec["status"] == "skipped_verified":
                skipped += 1
            else:
                downloaded += 1
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
            "elapsed_s": elapsed,
            "finished_at": datetime.now(UTC).isoformat(),
        }
    )
    save_manifest(root, man)
    return man["meta"] | {"start": start, "end": end, "days": len(days)}


def remote_opra_prefixes(root: Path) -> dict[str, Any]:
    """LIST only. Does not GET objects."""
    try:
        s3 = s3_from_env(root)
    except RuntimeError as exc:
        return {"ok": False, "reason": str(exc)}
    try:
        root_list = list_prefix(s3, "us_options_opra/", delimiter="/")
        datasets = {}
        for pfx in KNOWN_OPRA_PREFIXES:
            datasets[pfx] = list_prefix(s3, pfx, delimiter="/")
        return {
            "ok": True,
            "root": root_list,
            "datasets": datasets,
        }
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__}
