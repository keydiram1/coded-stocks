"""Targeted REST options quotes. Entitlement probe only until quotes succeed."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.client import DEFAULT_BASES
from quant_edge_lab.secrets import massive_api_key

QUOTE_PATH_TICKER = "O:SPY251114C00600000"
AUDIT_DAY = "2025-10-08"


def _quote_path(ticker: str) -> str:
    return "/v3/quotes/" + urllib.parse.quote(ticker, safe="")


def probe_quotes_rest_entitlement(
    root: Path,
    *,
    ticker: str = QUOTE_PATH_TICKER,
    timestamp_gte: str = "2025-10-08T14:30:00Z",
    timestamp_lte: str = "2025-10-08T14:31:00Z",
) -> dict[str, Any]:
    """One tiny historical quotes request. Does not paginate. Does not use flat files."""
    key = massive_api_key(root)
    path = _quote_path(ticker)
    params = {
        "timestamp.gte": timestamp_gte,
        "timestamp.lte": timestamp_lte,
        "limit": 1,
        "sort": "timestamp",
        "order": "asc",
        "apiKey": key,
    }
    attempts: list[dict[str, Any]] = []
    for base in DEFAULT_BASES:
        url = base.rstrip("/") + path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        rec: dict[str, Any] = {
            "base_host": urllib.parse.urlparse(base).netloc,
            "endpoint": path,
            "ticker": ticker,
            "timestamp_gte": timestamp_gte,
            "timestamp_lte": timestamp_lte,
            "limit": 1,
        }
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                payload = json.loads(raw.decode("utf-8"))
                rec.update(
                    {
                        "ok": True,
                        "http": int(resp.status),
                        "bytes": len(raw),
                        "api_status": payload.get("status") if isinstance(payload, dict) else None,
                        "n_results": len(payload.get("results") or [])
                        if isinstance(payload, dict)
                        else None,
                        "result_keys": list((payload.get("results") or [{}])[0].keys())
                        if isinstance(payload, dict) and payload.get("results")
                        else [],
                    }
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            err = None
            try:
                parsed = json.loads(body)
                err = parsed.get("error") if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                parsed = {"raw": body[:500]}
            rec.update(
                {
                    "ok": False,
                    "http": int(exc.code),
                    "error": err,
                    "api_status": parsed.get("status") if isinstance(parsed, dict) else None,
                    "body_excerpt": body[:500],
                }
            )
        attempts.append(rec)
        if rec.get("ok"):
            break
    first = attempts[0] if attempts else {}
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "REST_QUOTES_ENTITLEMENT_PROBE",
        "audit_day": AUDIT_DAY,
        "used_flatfiles": False,
        "entitled": bool(first.get("ok")),
        "http": first.get("http"),
        "error": first.get("error"),
        "attempts": attempts,
        "readiness": "NOT_READY" if not first.get("ok") else "READY_WITH_LIMITATIONS",
        "stop": not bool(first.get("ok")),
        "did_not": [
            "quotes_v1_flatfile_get",
            "options_hypothesis",
            "forward_returns",
            "convergence",
            "threshold_optimization",
            "sealed_oos",
        ],
    }
    out_dir = Paths(root).derived / "discovery" / "v5" / "v5-options-quotes-rest"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "entitlement.json"
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written"] = str(dest)
    return report
