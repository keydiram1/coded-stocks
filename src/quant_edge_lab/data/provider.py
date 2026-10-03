from __future__ import annotations

from typing import Protocol

import polars as pl


class MarketDataProvider(Protocol):
    """Provider adapters load immutable raw-ish frames. Sample works without secrets."""

    name: str

    def bars_1m(self) -> pl.DataFrame: ...

    def instruments(self) -> pl.DataFrame: ...


class SampleDataProvider:
    name = "sample"

    def __init__(self, bars: pl.DataFrame, instruments: pl.DataFrame) -> None:
        self._bars = bars
        self._instruments = instruments

    def bars_1m(self) -> pl.DataFrame:
        return self._bars

    def instruments(self) -> pl.DataFrame:
        return self._instruments


class MassiveProviderStub:
    """Config-only placeholder. No fake API. Requires credentials later."""

    name = "massive"

    def bars_1m(self) -> pl.DataFrame:
        raise RuntimeError("Massive adapter is not implemented. Use SampleDataProvider.")

    def instruments(self) -> pl.DataFrame:
        raise RuntimeError("Massive adapter is not implemented. Use SampleDataProvider.")
