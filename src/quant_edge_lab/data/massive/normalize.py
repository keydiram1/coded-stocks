from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from quant_edge_lab.data.store import BARS_COLUMNS

SOURCE = "massive"


def unix_ms_to_utc_naive(ms: int | float) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=UTC).replace(tzinfo=None)


def instrument_id_for(ticker: str, composite_figi: str | None = None, cik: str | None = None) -> str:
    if composite_figi:
        return f"figi:{composite_figi}"
    if cik:
        return f"cik:{cik}"
    return f"ticker:{ticker}"


def normalize_aggs(ticker: str, results: list[dict], instrument_id: str | None = None) -> pl.DataFrame:
    iid = instrument_id or instrument_id_for(ticker)
    rows = []
    seen: set[tuple[str, datetime]] = set()
    for item in results:
        ts = unix_ms_to_utc_naive(item["t"])
        key = (iid, ts)
        if key in seen:
            continue
        seen.add(key)
        o, h, l, c = float(item["o"]), float(item["h"]), float(item["l"]), float(item["c"])
        rows.append(
            {
                "instrument_id": iid,
                "ticker": ticker,
                "ts_utc": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": float(item.get("v") or 0),
                "vwap": float(item["vw"]) if item.get("vw") is not None else None,
                "transactions": int(item["n"]) if item.get("n") is not None else None,
                "source": SOURCE,
                "source_ts_ms": int(item["t"]),
                "adjusted": False,
            }
        )
    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 if c in {"open","high","low","close","volume","vwap"} else pl.String for c in BARS_COLUMNS})
    df = pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
    return df.sort(["instrument_id", "ts_utc"])


EXCHANGE_TO_CANON = {
    "XNAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "XNYS": "NYSE",
    "NYSE": "NYSE",
    "XASE": "NYSE_AMERICAN",
    "AMEX": "NYSE_AMERICAN",
    "NYE": "NYSE_AMERICAN",
    "ASE": "NYSE_AMERICAN",
}

ALLOWED_EXCHANGES = {"NASDAQ", "NYSE", "NYSE_AMERICAN"}
EXCLUDED_TYPES = {
    "ETF",
    "ETN",
    "ETV",
    "ETS",
    "WT",
    "Warrant",
    "RIGHT",
    "RIGHTS",
    "UNIT",
    "PFD",
    "PREFERRED",
    "FUND",
    "OPEN",
    "CLOSED",
    "CSVM",
}


def normalize_tickers(results: list[dict]) -> pl.DataFrame:
    rows = []
    for item in results:
        ttype = str(item.get("type") or "")
        if ttype.upper() in EXCLUDED_TYPES:
            continue
        exch = EXCHANGE_TO_CANON.get(str(item.get("primary_exchange") or "").upper())
        if exch not in ALLOWED_EXCHANGES:
            continue
        ticker = item["ticker"]
        rows.append(
            {
                "instrument_id": instrument_id_for(
                    ticker, item.get("composite_figi"), str(item["cik"]) if item.get("cik") else None
                ),
                "ticker": ticker,
                "effective_from": None,
                "effective_to": None,
                "exchange": exch,
                "security_type": "COMMON_STOCK" if ttype in {"", "CS"} else ttype,
                "cik": str(item["cik"]) if item.get("cik") else None,
                "active": bool(item.get("active", True)),
                "list_date": str(item["list_date"]) if item.get("list_date") is not None else None,
                "delist_date": str(item["delisted_utc"]) if item.get("delisted_utc") is not None else None,
                "composite_figi": str(item["composite_figi"]) if item.get("composite_figi") else None,
                "source": SOURCE,
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=len(rows))


def normalize_splits(results: list[dict]) -> pl.DataFrame:
    rows = []
    for item in results:
        ticker = item.get("ticker")
        if not ticker:
            continue
        split_from = item.get("split_from") or item.get("from")
        split_to = item.get("split_to") or item.get("to")
        ratio = None
        if split_from and split_to:
            ratio = float(split_to) / float(split_from)
        rows.append(
            {
                "instrument_id": instrument_id_for(ticker),
                "ticker": ticker,
                "type": "SPLIT",
                "effective_date": item.get("execution_date"),
                "ratio": ratio,
                "source": SOURCE,
                "raw_split_from": split_from,
                "raw_split_to": split_to,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
