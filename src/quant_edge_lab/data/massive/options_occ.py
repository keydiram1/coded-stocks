"""OCC option ticker parsing. Does not invent fields the files do not carry."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

OCC_RE = re.compile(r"^O:([A-Z][A-Z0-9.]{0,5})(\d{6})([CP])(\d{8})$")


def parse_occ_ticker(ticker: str) -> dict[str, Any] | None:
    if not ticker:
        return None
    m = OCC_RE.match(str(ticker).strip())
    if m is None:
        return None
    root, yymmdd, cp, strike_raw = m.group(1), m.group(2), m.group(3), m.group(4)
    yy = int(yymmdd[:2])
    year = 2000 + yy
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    try:
        expiry = date(year, month, day)
    except ValueError:
        return None
    return {
        "occ_ticker": ticker,
        "underlying_root": root,
        "expiration": expiry.isoformat(),
        "call_put": "call" if cp == "C" else "put",
        "strike": int(strike_raw) / 1000.0,
        "strike_raw": strike_raw,
        "root_contains_digit": any(ch.isdigit() for ch in root),
    }


def root_looks_adjusted(root: str | None) -> bool:
    """Conservative: digit-in-root OSI codes often mark adjusted/nonstandard classes.

    Flat files do not carry shares_per_contract. This is an exclusion heuristic,
    not a positive identification of standard 100-share contracts.
    """
    if not root:
        return True
    return any(ch.isdigit() for ch in root)
