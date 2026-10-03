from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InstrumentHistory(BaseModel):
    instrument_id: str
    ticker: str
    effective_from: datetime
    effective_to: datetime | None = None
    exchange: str
    security_type: str = "COMMON_STOCK"
    cik: str | None = None
    active: bool = True
    list_date: datetime | None = None
    delist_date: datetime | None = None


class Bar1m(BaseModel):
    instrument_id: str
    ts_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    transactions: int | None = None
    source: str = "sample"


class Trade(BaseModel):
    instrument_id: str
    ts_event: datetime
    ts_sip: datetime | None = None
    sequence: int | None = None
    price: float
    size: float
    exchange: str | None = None
    conditions: str | None = None


class Quote(BaseModel):
    instrument_id: str
    ts_event: datetime
    ts_sip: datetime | None = None
    sequence: int | None = None
    bid_px: float
    bid_sz: float
    bid_ex: str | None = None
    ask_px: float
    ask_sz: float
    ask_ex: str | None = None
    conditions: str | None = None


class FundamentalPointInTime(BaseModel):
    instrument_id: str
    accepted_at: datetime
    period_end: datetime | None = None
    shares_out: float | None = None
    market_cap: float | None = None
    source: str


class FloatObservation(BaseModel):
    instrument_id: str
    effective_at: datetime
    observed_at: datetime
    float_shares: float | None = None
    float_pct: float | None = None
    method: str | None = None
    source: str
    quality: str | None = None


class NewsEvent(BaseModel):
    news_id: str
    published_at: datetime
    first_seen_at: datetime
    provider: str
    instruments: list[str]
    headline: str
    catalyst_class: str | None = None
    raw_hash: str | None = None


class CorporateAction(BaseModel):
    instrument_id: str
    type: str
    effective_date: datetime
    ratio: float | None = None
    source: str


class ResearchEvent(BaseModel):
    event_id: str
    experiment_id: str
    instrument_id: str
    ticker: str
    event_ts: datetime
    decision_ts: datetime
    feature_version: str
    side: Literal["long", "short"]
    features: dict[str, float | None] = Field(default_factory=dict)


class Outcome(BaseModel):
    event_id: str
    horizon: str
    forward_return: float | None = None
    mfe: float | None = None
    mae: float | None = None
    target_first: bool | None = None
    stop_first: bool | None = None
    ambiguous: bool = False
    entry_price: float | None = None
    execution_status: str = "SIGNAL_ONLY"


class ExperimentRecord(BaseModel):
    experiment_id: str
    hypothesis_id: str
    hypothesis_hash: str
    family_id: str
    git_sha: str
    data_manifest_hash: str
    universe_version: str
    feature_version: str
    execution_model_version: str
    random_seed: int
    created_at: datetime
    stage: str
    number_of_variants_tested: int
    hypothesis_path: str
    events_path: str
    outcomes_path: str
    summary_path: str
    report_path: str
    events_hash: str
    outcomes_hash: str
    dataset: str = "sample"
    data_source: str = "sample"


class LiveSetup(BaseModel):
    event_id: str
    detected_at: datetime
    market_ts: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    feed_lag_ms: float | None = None
    scanner_latency_ms: float | None = None


class ExecutionObservation(BaseModel):
    event_id: str
    decision_ts: datetime
    order_ts: datetime | None = None
    ack_ts: datetime | None = None
    fill_ts: datetime | None = None
    qty: float | None = None
    filled_qty: float | None = None
    price: float | None = None
    fees: float | None = None
    reject_reason: str | None = None
