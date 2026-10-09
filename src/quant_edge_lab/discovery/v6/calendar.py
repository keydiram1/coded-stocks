"""Pin V6 to the canonical stock-panel research calendar. No bar payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_edge_lab.discovery.v5.continuation import SOURCE_IDENTITY
from quant_edge_lab.discovery.v5.preflight import (
    CalendarError,
    clip_to_panel,
    validate_research_calendar,
)
from quant_edge_lab.hashing import sha256_json

EXPECTED_CALENDAR_HASH = SOURCE_IDENTITY["calendar"]
EXPECTED_CALENDAR_N_DAYS = 1255


class CalendarIdentityError(RuntimeError):
    pass


def calendar_manifest(des: dict[str, Any]) -> dict[str, Any]:
    """Adapter: V6 YAML split names -> V5 preflight shape. No new science values."""
    camp = des["MARKET_RESIDUAL_SHOCK_RETENTION_V1"]
    s = camp["splits"]
    return {
        "data": {"panel_start": "2021-10-01", "panel_end": "2026-10-01"},
        "splits": {
            "warmup_last_day": s["warmup_last_day"],
            "D1": {
                "start": s["discovery_D1"]["start"],
                "end": s["discovery_D1"]["end"],
                "n_days": s["discovery_D1"]["n_days"],
            },
            "D2": {
                "start": s["validation_D2"]["start"],
                "end": s["validation_D2"]["end"],
                "n_days": s["validation_D2"]["n_days"],
            },
            "D3": {
                "start": s["confirmation_D3"]["start"],
                "end": s["confirmation_D3"]["end"],
                "n_days": s["confirmation_D3"]["n_days"],
            },
            "sealed_oos": s.get("sealed_oos") or des.get("sealed_oos"),
        },
        "eligibility": {"warmup_trading_days": 20},
        "sealed_oos": des.get("sealed_oos"),
    }


def _parquet_exists_no_read(root: Path, day: str) -> bool:
    from quant_edge_lab.data.massive.flatfiles import local_parquet_path

    return local_parquet_path(root, day).exists()


def v6_calendar_report(
    root: Path,
    des: dict[str, Any],
    *,
    days: list[str] | None = None,
    parquet_exists=None,
) -> dict[str, Any]:
    from quant_edge_lab.discovery.campaign_v4 import research_days

    man = calendar_manifest(des)
    raw = list(days) if days is not None else research_days(root)
    exists = parquet_exists or (lambda d: _parquet_exists_no_read(root, d))
    rec = validate_research_calendar(raw, man, parquet_exists=exists)
    ordered = rec["days"]
    actual = rec["calendar_hash"]
    if actual != sha256_json(ordered):
        raise CalendarError("calendar hash is not sha256_json of the validated day list")
    rec["calendar_hash_actual"] = actual
    rec["calendar_hash_expected"] = EXPECTED_CALENDAR_HASH
    rec["calendar_identity_match"] = actual == EXPECTED_CALENDAR_HASH
    rec["calendar_n_days"] = rec["n_days"]
    rec["days"] = ordered
    return rec


def require_calendar_identity(cal: dict[str, Any]) -> None:
    if not cal.get("calendar_identity_match"):
        raise CalendarIdentityError(
            f"calendar identity mismatch actual={cal.get('calendar_hash_actual')} "
            f"expected={cal.get('calendar_hash_expected')}"
        )
    if int(cal.get("calendar_n_days") or 0) != EXPECTED_CALENDAR_N_DAYS:
        raise CalendarIdentityError(
            f"calendar n_days={cal.get('calendar_n_days')} expected={EXPECTED_CALENDAR_N_DAYS}"
        )


def d1_loadable_days(calendar: list[str], cal_man: dict[str, Any]) -> list[str]:
    from quant_edge_lab.discovery.v5.partitions import split_for_day

    out: list[str] = []
    for day in clip_to_panel(calendar, cal_man):
        sp = split_for_day(day, cal_man)
        if sp in {"D2", "D3"}:
            break
        if sp in {"WARMUP", "D1"}:
            out.append(day)
    return out


def assert_no_future_splits(days: list[str], cal_man: dict[str, Any]) -> None:
    from quant_edge_lab.discovery.v5.partitions import split_for_day

    for day in days:
        sp = split_for_day(day, cal_man)
        if sp in {"D2", "D3"}:
            raise RuntimeError(f"D1 execute refuses to load {day} ({sp})")
