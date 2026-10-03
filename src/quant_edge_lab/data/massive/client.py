"""Massive (formerly Polygon) REST client. Never logs credentials or full URLs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from quant_edge_lab.secrets import massive_api_key

DEFAULT_BASES = (
    "https://api.massive.com",
    "https://api.polygon.io",
)


class MassiveHTTPError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, endpoint: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint


def _endpoint_only(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.path


@dataclass
class MassiveClient:
    api_key: str
    base_url: str = DEFAULT_BASES[0]
    timeout_s: float = 60.0
    max_retries: int = 6
    opener: Callable[..., Any] | None = None

    @classmethod
    def from_env(cls, root=None, base_url: str | None = None) -> MassiveClient:
        return cls(api_key=massive_api_key(root), base_url=base_url or DEFAULT_BASES[0])

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            url = self.base_url.rstrip("/") + path
            if params:
                url = url + "?" + urllib.parse.urlencode(params, doseq=True)
            req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
            try:
                if self.opener:
                    raw = self.opener(req, timeout=self.timeout_s)
                    payload = json.loads(raw)
                else:
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, dict) and str(payload.get("status", "")).upper() in {
                    "ERROR",
                    "NOT_AUTHORIZED",
                }:
                    raise MassiveHTTPError(
                        f"API status {payload.get('status')} for {path}",
                        status=403,
                        endpoint=path,
                    )
                return payload if isinstance(payload, dict) else {"results": payload}
            except urllib.error.HTTPError as exc:
                last_error = MassiveHTTPError(
                    f"HTTP {exc.code} for {_endpoint_only(url)}",
                    status=exc.code,
                    endpoint=_endpoint_only(url),
                )
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0**attempt)
                    time.sleep(delay)
                    continue
                if exc.code in {500, 502, 503, 504}:
                    time.sleep(min(30.0, 1.5**attempt))
                    continue
                raise last_error from None
            except TimeoutError as exc:
                last_error = MassiveHTTPError(f"Timeout for {path}", endpoint=path)
                time.sleep(min(30.0, 1.5**attempt))
                continue
            except urllib.error.URLError as exc:
                last_error = MassiveHTTPError(f"Network error for {path}", endpoint=path)
                time.sleep(min(30.0, 1.5**attempt))
                continue
        raise last_error or MassiveHTTPError(f"Failed GET {path}", endpoint=path)

    def paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        results_key: str = "results",
        page_sleep_s: float = 0.0,
        max_items: int | None = None,
    ):
        params = dict(params or {})
        yielded = 0
        data = self.get(path, params)
        for item in data.get(results_key) or []:
            yield item
            yielded += 1
            if max_items is not None and yielded >= max_items:
                return
        next_url = data.get("next_url")
        while next_url:
            if page_sleep_s:
                time.sleep(page_sleep_s)
            parsed = urllib.parse.urlparse(next_url)
            q = dict(urllib.parse.parse_qsl(parsed.query))
            q.pop("apiKey", None)
            data = self.get(parsed.path, q)
            for item in data.get(results_key) or []:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            next_url = data.get("next_url")


def probe_bases(api_key: str, opener: Callable[..., Any] | None = None) -> str:
    """Return the first base URL that authenticates. Never logs the key."""
    client = MassiveClient(api_key=api_key, opener=opener)
    last: Exception | None = None
    for base in DEFAULT_BASES:
        client.base_url = base
        try:
            client.get("/v3/reference/tickers", {"ticker": "AAPL", "market": "stocks", "limit": 1})
            return base
        except MassiveHTTPError as exc:
            last = exc
            if exc.status in {401, 403}:
                # Auth failed on this host; try next host before giving up.
                continue
            continue
    raise last or MassiveHTTPError("Could not authenticate against Massive hosts")
