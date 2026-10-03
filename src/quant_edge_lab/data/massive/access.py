from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.client import MassiveClient, MassiveHTTPError, probe_bases
from quant_edge_lab.secrets import massive_api_key


PROBE_ENDPOINTS = (
    ("tickers", "/v3/reference/tickers", {"ticker": "AAPL", "market": "stocks", "limit": 1}),
    (
        "aggs_1m",
        "/v2/aggs/ticker/AAPL/range/1/minute/2026-09-15/2026-09-16",
        {"adjusted": "false", "limit": 50, "sort": "asc"},
    ),
    (
        "grouped_daily",
        "/v2/aggs/grouped/locale/us/market/stocks/2026-09-15",
        {"adjusted": "false"},
    ),
    ("splits_v3", "/v3/reference/splits", {"ticker": "AAPL", "limit": 1}),
    ("splits_v1", "/stocks/v1/splits", {"ticker": "AAPL", "limit": 1}),
    (
        "tickers_pit",
        "/v3/reference/tickers",
        {"market": "stocks", "type": "CS", "date": "2026-09-15", "active": "true", "limit": 1},
    ),
    (
        "tickers_inactive",
        "/v3/reference/tickers",
        {"market": "stocks", "active": "false", "limit": 1},
    ),
)


def check_access(root: Path) -> dict[str, Any]:
    key = massive_api_key(root)
    base = probe_bases(key)
    client = MassiveClient(api_key=key, base_url=base)
    endpoints: dict[str, Any] = {}
    for name, path, params in PROBE_ENDPOINTS:
        try:
            payload = client.get(path, params)
            results = payload.get("results") or []
            endpoints[name] = {
                "ok": True,
                "status": 200,
                "path": path,
                "result_count": len(results) if isinstance(results, list) else None,
                "adjusted": payload.get("adjusted"),
                "note": None,
            }
        except MassiveHTTPError as exc:
            note = "rate_limited_not_necessarily_unauthorized" if exc.status == 429 else str(exc)
            endpoints[name] = {
                "ok": False,
                "status": exc.status,
                "path": path,
                "result_count": 0,
                "note": note,
            }
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "base_url": base,
        "endpoints": endpoints,
        "minute_aggregates": endpoints.get("aggs_1m", {}).get("ok"),
        "reference_tickers": endpoints.get("tickers", {}).get("ok"),
        "corporate_actions": bool(
            endpoints.get("splits_v3", {}).get("ok") or endpoints.get("splits_v1", {}).get("ok")
        ),
        "warnings": [
            n for n, v in endpoints.items() if not v.get("ok")
        ],
    }
    out = Paths(root).manifests / "massive" / "access_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
