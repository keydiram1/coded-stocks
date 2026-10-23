"""Massive dated SIC membership sample store and source audit.

Not true PIT. Not GICS. Does not compute feasibility or any future return.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.hashing import sha256_json

SEMANTICS = "MASSIVE_SIC_EFFECTIVE_DATED_AS_CURRENTLY_MAINTAINED"
PIT_STATUS = "NOT_TRUE_PIT"
CLASSIFICATION = "SIC_not_GICS"
STORE_VERSION = "massive_sic_asof_v2"
AUDIT_DATES = (
    "2021-10-01",
    "2022-01-03",
    "2023-01-03",
    "2024-01-02",
    "2025-01-02",
    "2026-10-01",
)
LIST_SAMPLE_LIMIT = 25
COHORT_N = 40
OVERVIEW_SAMPLE_N = 40
FORBIDDEN_OUTPUT_KEYS = (
    "future_1m_return",
    "future_5m_return",
    "future_30m_return",
    "next_day_return",
    "convergence",
    "reversal",
    "continuation",
    "response_gap",
    "group_shock_z",
)
STORE_MUST_NOT_CHOOSE = ("group_id", "group_level", "assignment")


def assert_no_forbidden_keys(payload: Any) -> None:
    blob = json.dumps(payload, default=str)
    for key in FORBIDDEN_OUTPUT_KEYS:
        if key in blob:
            raise ValueError(f"forbidden Stage 0/1 key in membership output: {key}")


def normalize_sic(raw: Any) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    digits = "".join(ch for ch in str(raw).strip() if ch.isdigit())
    if not digits:
        return None, None
    sic4 = digits.zfill(4)[-4:]
    return sic4, sic4[:2]


def _row_from_item(item: dict[str, Any], asof: str) -> dict[str, Any]:
    ticker = str(item.get("ticker") or "")
    sic_raw = item.get("sic_code") if item.get("sic_code") is not None else item.get("sic")
    sic4, sic2 = normalize_sic(sic_raw)
    return {
        "asof_date": asof,
        "ticker": ticker,
        "instrument_id": f"ticker:{ticker}" if ticker else None,
        "sic_code": None if sic_raw is None else str(sic_raw),
        "sic_description": item.get("sic_description"),
        "sic4": sic4,
        "sic2": sic2,
        "sic4_id": None if sic4 is None else f"sic4:{sic4}",
        "sic2_id": None if sic2 is None else f"sic2:{sic2}",
        "sic2_level": None if sic2 is None else "sic_major_group",
        "exchange": item.get("primary_exchange"),
        "type": item.get("type"),
        "active": item.get("active"),
        "cik": None if item.get("cik") is None else str(item.get("cik")),
        "composite_figi": item.get("composite_figi"),
        "list_date": item.get("list_date"),
        "delisted_utc": item.get("delisted_utc") or item.get("delist_date"),
        "last_updated_utc": item.get("last_updated_utc"),
        "source": "massive",
        "classification": CLASSIFICATION,
        "pit_status": PIT_STATUS,
        "semantics": SEMANTICS,
        "store_version": STORE_VERSION,
    }


def membership_from_tickers(items: list[dict[str, Any]], asof: str) -> list[dict[str, Any]]:
    """Hierarchy only. Stage 0 chooses sic4 vs sic2 from eligible counts at the event."""
    return [_row_from_item(item, asof) for item in items]


def write_asof_part(root: Path, asof: str, rows: list[dict[str, Any]]) -> Path:
    assert_no_forbidden_keys(rows)
    for row in rows:
        for key in STORE_MUST_NOT_CHOOSE:
            if key in row:
                raise ValueError(f"membership store must not choose {key}")
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    dest = Paths(root).derived / "groups" / "massive_sic_asof" / f"date={asof}" / "part.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    return dest


def _keys_of(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return sorted(obj.keys())
    return []


def _blank(val: Any) -> bool:
    return val is None or str(val).strip() == ""


def _missing_pct(rows: list[dict[str, Any]], *fields: str) -> float:
    if not rows:
        return float("nan")
    n_miss = sum(1 for item in rows if all(_blank(item.get(f)) for f in fields))
    return round(100.0 * n_miss / len(rows), 2)


def _list_cs(
    client: Any, asof: str, *, active: bool, max_items: int | None
) -> list[dict[str, Any]]:
    params = {
        "market": "stocks",
        "type": "CS",
        "date": asof,
        "active": "true" if active else "false",
        "limit": 1000 if max_items is None else min(1000, max_items),
        "order": "asc",
        "sort": "ticker",
    }
    if hasattr(client, "paginate"):
        return list(
            client.paginate(
                "/v3/reference/tickers",
                params,
                max_items=max_items,
                page_sleep_s=0.05,
            )
        )
    payload = client.get("/v3/reference/tickers", params)
    rows = list(payload.get("results") or [])
    return rows[:max_items] if max_items is not None else rows


def _overview(client: Any, ticker: str, asof: str) -> dict[str, Any]:
    payload = client.get(f"/v3/reference/tickers/{ticker}", {"date": asof})
    body = payload.get("results") or payload
    return body if isinstance(body, dict) else {}


def _last_updated_after_asof(item: dict[str, Any], asof: str) -> bool:
    raw = str(item.get("last_updated_utc") or "")
    if len(raw) < 10:
        return False
    return raw[:10] > asof


def audit_source(
    client: Any,
    *,
    root: Path,
    dates: tuple[str, ...] = AUDIT_DATES,
    list_limit: int | None = LIST_SAMPLE_LIMIT,
    write_sample_store: bool = True,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dated-ticker source audit. Six as-of snapshots, not a 1255-day panel. No returns."""
    date_reports: list[dict[str, Any]] = []
    written: list[str] = []
    by_date_tickers: dict[str, set[str]] = {}
    sic_by_ticker: dict[str, dict[str, str | None]] = {}
    last_updated_after = 0
    last_updated_n = 0

    first_rows = _list_cs(client, dates[0], active=True, max_items=list_limit)
    refetch = _list_cs(client, dates[0], active=True, max_items=list_limit)
    revision_identical_refetch = sha256_json(first_rows) == sha256_json(refetch)
    cohort = [str(r.get("ticker")) for r in first_rows if r.get("ticker")][:COHORT_N]

    for asof in dates:
        active_rows = _list_cs(client, asof, active=True, max_items=list_limit)
        inactive_rows = _list_cs(client, asof, active=False, max_items=list_limit)
        list_rows = active_rows + inactive_rows
        list_keys: set[str] = set()
        for item in list_rows:
            list_keys.update(_keys_of(item))
            if not _blank(item.get("last_updated_utc")):
                last_updated_n += 1
                if _last_updated_after_asof(item, asof):
                    last_updated_after += 1
        overview = _overview(client, "AAPL", asof)
        overview_keys = _keys_of(overview)
        sic_on_list = "sic_code" in list_keys or "sic" in list_keys
        tickers = {str(r.get("ticker")) for r in list_rows if r.get("ticker")}
        by_date_tickers[asof] = tickers
        for item in list_rows:
            t = str(item.get("ticker") or "")
            if t:
                sic_by_ticker.setdefault(t, {})[asof] = None if _blank(
                    item.get("sic_code") if item.get("sic_code") is not None else item.get("sic")
                ) else str(item.get("sic_code") or item.get("sic"))
        overview_sample = []
        if not sic_on_list:
            for t in list(tickers)[:OVERVIEW_SAMPLE_N]:
                body = _overview(client, t, asof)
                overview_sample.append(
                    {
                        "ticker": t,
                        "sic_code": body.get("sic_code") or body.get("sic"),
                    }
                )
        membership = membership_from_tickers(active_rows, asof)
        if write_sample_store and membership:
            path = write_asof_part(root, asof, membership)
            written.append(str(path))
        date_reports.append(
            {
                "asof": asof,
                "n_active": len(active_rows),
                "n_inactive": len(inactive_rows),
                "list_keys": sorted(list_keys),
                "list_sic_field_present": sic_on_list,
                "missing_sic_pct": _missing_pct(list_rows, "sic_code", "sic"),
                "missing_cik_pct": _missing_pct(list_rows, "cik"),
                "missing_figi_pct": _missing_pct(list_rows, "composite_figi"),
                "n_with_delisted_utc": sum(
                    1
                    for r in list_rows
                    if not _blank(r.get("delisted_utc") or r.get("delist_date"))
                ),
                "overview_keys": overview_keys,
                "overview_sic_field_present": "sic_code" in overview_keys or "sic" in overview_keys,
                "overview_sic_code": overview.get("sic_code") or overview.get("sic"),
                "overview_sample_sic_missing_pct": _missing_pct(overview_sample, "sic_code"),
                "n_with_sic4": sum(1 for r in membership if r.get("sic4")),
                "n_with_sic2": sum(1 for r in membership if r.get("sic2")),
            }
        )

    first, last = dates[0], dates[-1]
    appeared = sorted(by_date_tickers.get(last, set()) - by_date_tickers.get(first, set()))
    disappeared = sorted(by_date_tickers.get(first, set()) - by_date_tickers.get(last, set()))
    for t in cohort:
        for d in dates:
            if not _blank(sic_by_ticker.get(t, {}).get(d)):
                continue
            body = _overview(client, t, d)
            code = body.get("sic_code") or body.get("sic")
            sic_by_ticker.setdefault(t, {})[d] = None if _blank(code) else str(code)
    sic_changed = 0
    sic_compared = 0
    for t in cohort:
        present = [sic_by_ticker.get(t, {}).get(d) for d in dates]
        present = [s for s in present if s]
        if len(present) >= 2:
            sic_compared += 1
            if len(set(present)) > 1:
                sic_changed += 1

    report = {
        "store_version": STORE_VERSION,
        "classification": CLASSIFICATION,
        "pit_status": PIT_STATUS,
        "semantics": SEMANTICS,
        "call_it_pit_gics": False,
        "call_it_true_pit_sic": False,
        "not_as_known_on_that_day": True,
        "stage0_feasibility_ran": False,
        "future_returns_computed": False,
        "full_panel_ingest": False,
        "auth": auth or {},
        "sample_parts_written": written,
        "dates": date_reports,
        "sic_coverage": {
            "asofs": list(dates),
            "list_has_sic_on_any_date": any(d["list_sic_field_present"] for d in date_reports),
            "overview_has_sic_on_any_date": any(
                d["overview_sic_field_present"] for d in date_reports
            ),
            "missing_sic_pct_by_asof": {d["asof"]: d["missing_sic_pct"] for d in date_reports},
        },
        "missing_identifiers": {
            "missing_sic_pct_by_asof": {d["asof"]: d["missing_sic_pct"] for d in date_reports},
            "missing_cik_pct_by_asof": {d["asof"]: d["missing_cik_pct"] for d in date_reports},
            "missing_figi_pct_by_asof": {d["asof"]: d["missing_figi_pct"] for d in date_reports},
        },
        "ticker_changes_delistings": {
            "first_asof": first,
            "last_asof": last,
            "n_appeared": len(appeared),
            "n_disappeared": len(disappeared),
            "appeared_sample": appeared[:25],
            "disappeared_sample": disappeared[:25],
            "n_with_delisted_utc_by_asof": {
                d["asof"]: d["n_with_delisted_utc"] for d in date_reports
            },
        },
        "sic_code_changes": {
            "cohort_n": len(cohort),
            "n_compared": sic_compared,
            "n_sic_changed": sic_changed,
        },
        "revision_behaviour": {
            "identical_refetch_of_first_asof": revision_identical_refetch,
            "n_rows_with_last_updated_utc": last_updated_n,
            "n_last_updated_utc_after_asof": last_updated_after,
            "last_updated_after_asof_is_consistent_with_as_currently_maintained": last_updated_after
            > 0,
        },
    }
    report["provenance_hash"] = sha256_json(
        {k: report[k] for k in report if k not in {"provenance_hash", "audit_path"}}
    )
    assert_no_forbidden_keys(report)
    out = Paths(root).manifests / "groups" / "massive_sic_source_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["audit_path"] = str(out)
    return report


def run_source_audit(root: Path, *, list_limit: int | None = None) -> dict[str, Any]:
    from quant_edge_lab.data.massive.client import MassiveClient, MassiveHTTPError, probe_bases
    from quant_edge_lab.secrets import massive_api_key, massive_key_fingerprint

    auth = dict(massive_key_fingerprint(root))
    key = massive_api_key(root)
    try:
        base = probe_bases(key)
        auth["base_url"] = base
        auth["http_status"] = 200
        auth["interpretation"] = "401_would_mean_unknown_or_missing_key_403_would_mean_entitlement"
        client = MassiveClient(api_key=key, base_url=base)
        return audit_source(client, root=root, list_limit=list_limit, auth=auth)
    except MassiveHTTPError as exc:
        auth["base_url"] = None
        auth["http_status"] = exc.status
        auth["error"] = str(exc)
        if exc.status == 401:
            auth["interpretation"] = "key_missing_or_unknown_not_plan_entitlement"
        elif exc.status == 403:
            auth["interpretation"] = "plan_not_authorized"
        else:
            auth["interpretation"] = "other_http_error"
        report = {
            "store_version": STORE_VERSION,
            "auth": auth,
            "stage0_feasibility_ran": False,
            "future_returns_computed": False,
            "full_panel_ingest": False,
            "audit_ran": False,
        }
        out = Paths(root).manifests / "groups" / "massive_sic_source_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["audit_path"] = str(out)
        return report
