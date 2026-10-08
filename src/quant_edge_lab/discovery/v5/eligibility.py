"""V5 listed-universe filter. Listing type/exchange come from instruments.parquet (hashed).

prev_close and 20d median RTH dollar volume are causal from completed prior sessions.
instruments.parquet is the same last-available CS table V4 uses; it is not session-PIT listing
history. That limitation is explicit. Execute refuses if the file is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.hashing import sha256_file

EXCHANGES = ("NASDAQ", "NYSE", "NYSE_AMERICAN")
SECURITY_TYPE = "COMMON_STOCK"


@dataclass
class EligibilityConfig:
    exchanges: tuple[str, ...] = EXCHANGES
    security_type: str = SECURITY_TYPE
    min_prev_close: float = 1.0
    min_median_rth_dvol: float = 1_000_000.0
    min_dvol_days: int = 20
    lookback: int = 20


def instruments_path(root: Path) -> Path:
    return Paths(root).normalized / "massive-flat" / "instruments.parquet"


def instruments_identity(root: Path) -> str:
    p = instruments_path(root)
    if not p.exists():
        return "MISSING"
    return sha256_file(p)


def listed_ids(instruments: pl.DataFrame | None, cfg: EligibilityConfig) -> set[str] | None:
    """None means no listing table (unit-test helper). Execute always supplies instruments."""
    if instruments is None:
        return None
    need = {"instrument_id", "exchange", "security_type"}
    if not need <= set(instruments.columns):
        raise ValueError("instruments must include instrument_id, exchange, security_type")
    ok = instruments.filter(
        (pl.col("security_type") == cfg.security_type)
        & pl.col("exchange").is_in(list(cfg.exchanges))
    )
    return {str(x) for x in ok["instrument_id"].to_list()}


def is_listed(iid: str, listed: set[str] | None) -> bool:
    if listed is None:
        return True
    return iid in listed


def causal_ok(
    iid: str,
    day: str,
    hist: dict[str, dict[str, list[tuple[str, float]]]],
    cfg: EligibilityConfig,
) -> bool:
    rec = hist.get(iid) or {"closes": [], "dvols": []}
    closes = [
        (d, v) for d, v in rec.get("closes") or [] if d < day and v is not None and np.isfinite(v)
    ]
    dvols = [
        (d, v)
        for d, v in rec.get("dvols") or []
        if d < day and v is not None and np.isfinite(v) and v > 0
    ]
    if cfg.min_prev_close > 0:
        if not closes:
            return False
        if float(closes[-1][1]) < cfg.min_prev_close:
            return False
    if cfg.min_dvol_days > 0 or cfg.min_median_rth_dvol > 0:
        if len(dvols) < cfg.min_dvol_days:
            return False
        med = float(np.median([v for _d, v in dvols[-cfg.lookback :]]))
        if med < cfg.min_median_rth_dvol:
            return False
    return True


def update_elig_hist(
    hist: dict[str, dict[str, list[tuple[str, float]]]],
    *,
    iid: str,
    day: str,
    close_1559: float | None,
    rth_dvol: float | None,
    lookback: int,
) -> None:
    rec = hist.setdefault(iid, {"closes": [], "dvols": []})
    if close_1559 is not None and np.isfinite(close_1559):
        rec["closes"].append((day, float(close_1559)))
        rec["closes"] = rec["closes"][-lookback:]
    if rth_dvol is not None and np.isfinite(rth_dvol) and rth_dvol > 0:
        rec["dvols"].append((day, float(rth_dvol)))
        rec["dvols"] = rec["dvols"][-lookback:]


def rth_dollar_volume(g: pl.DataFrame) -> float | None:
    if g.height == 0 or "close" not in g.columns or "volume" not in g.columns:
        return None
    v = float((g["close"] * g["volume"]).sum())
    return v if np.isfinite(v) else None


def cfg_from_manifest(man: dict[str, Any]) -> EligibilityConfig:
    el = man.get("eligibility") or {}
    return EligibilityConfig(
        exchanges=tuple(el.get("exchanges") or EXCHANGES),
        security_type=str(el.get("security_type") or SECURITY_TYPE),
        min_prev_close=float(el.get("price_prev_close_min") or 1.0),
        min_median_rth_dvol=float(
            el.get("trailing_20d_median_rth_dollar_volume_min") or 1_000_000.0
        ),
        min_dvol_days=int(el.get("warmup_trading_days") or 20),
        lookback=int(el.get("warmup_trading_days") or 20),
    )
