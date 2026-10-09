"""Exact prior-20 research-session residual maps. No last-20-available reachback."""

from __future__ import annotations

from collections import deque

import numpy as np

from quant_edge_lab.discovery.v6.residual import SCALE_LOOKBACK, shock_z

Key = tuple[str, str]
HistoryKey = Key


class SessionResidualHistory:
    """Rolling deque of sparse day maps: (instrument_id, clock_key) -> residual.

    Lookup uses the exact previous len(maps) sessions. Scale is available only
    when there are exactly 20 maps and the key is finite in every one.
    """

    def __init__(self, *, lookback: int = SCALE_LOOKBACK) -> None:
        self.lookback = int(lookback)
        self._maps: deque[dict[Key, float]] = deque()

    def __len__(self) -> int:
        return len(self._maps)

    def push(self, day_map: dict[Key, float]) -> None:
        self._maps.append(dict(day_map))
        while len(self._maps) > self.lookback:
            self._maps.popleft()

    def lookup(self, iid: str, clock_key: str) -> list[float] | None:
        if len(self._maps) != self.lookback:
            return None
        key: Key = (iid, clock_key)
        out: list[float] = []
        for m in self._maps:
            if key not in m:
                return None
            v = m[key]
            if v is None or not np.isfinite(float(v)):
                return None
            out.append(float(v))
        return out

    def shock_z_for(self, iid: str, clock_key: str, impulse: float | None) -> float | None:
        return shock_z(impulse, self.lookup(iid, clock_key))
