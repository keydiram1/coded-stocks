"""Explicit V5 scientific objects. Asset-class agnostic at the core."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Direction = Literal["LONG", "SHORT"]
DecisionLabel = Literal["KILL", "VALIDATED_SUBTHRESHOLD_PHENOMENON", "RESEARCH_PASS", "ACTIVE"]
StageName = Literal["READINESS", "EVENTS", "D1", "D2", "D3", "REPORT"]
HypothesisStatus = Literal["ACTIVE", "KILL", "PENDING"]


class MechanismSpec(BaseModel):
    mechanism_id: str
    description: str
    who_or_what: str
    asset_class: str | None = None
    campaign_id: str | None = None


class ExpectedResponse(BaseModel):
    name: str
    value: float | None
    definition: str
    available_at: datetime | None = None


class ObservedResponse(BaseModel):
    name: str
    value: float | None
    definition: str
    available_at: datetime | None = None


class Discrepancy(BaseModel):
    name: str
    value: float | None
    definition: str = "observed_t - expected_t"


class ResolutionOutcome(BaseModel):
    name: str
    horizon: str
    value: float | None = None
    diagnostic: bool = False
    executable_claim: bool = False


class CandidateRule(BaseModel):
    hypothesis_id: str
    mechanism_id: str
    role: Literal["primary", "robustness"]
    dislocation_abs_min: float | None = None
    rvol_min: float | None = None
    require_rvol: bool = True
    direction_policy: Literal["reversal_no_flip"] = "reversal_no_flip"
    primary_outcome: str
    frozen_from_split: str = "D1"
    frozen: bool = False


class HypothesisEvent(BaseModel):
    """Minimum V5 event contract. Extra campaign fields may ride in extras."""

    instrument_id: str
    ticker: str | None = None
    trading_date: str
    decision_ts: datetime
    first_available_at: datetime | None = None
    observed_value: float | None
    expected_value: float | None
    discrepancy: float | None
    expected_resolution_direction: Direction | None
    mechanism_id: str
    hypothesis_id: str
    extras: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    stage: StageName
    split: str | None = None
    n_events: int = 0
    n_hypotheses: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ScientificDecision(BaseModel):
    hypothesis_id: str
    split: str
    label: DecisionLabel
    reason: str
    mean_primary: float | None = None
    trial_index: int | None = None


class TelemetrySnapshot(BaseModel):
    stage: str
    substage: str | None = None
    current_unit: str | None = None
    completed: int = 0
    total: int = 0
    percent_complete: float | None = None
    elapsed_s: float | None = None
    eta_s: float | None = None
    rows_processed: int = 0
    events_generated: int = 0
    hypotheses_active: int = 0
    hypotheses_killed: int = 0
    checkpoint_state: str = "none"
    last_successful_checkpoint: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    sample_size: int | None = None
    ticker_days: int | None = None
    trading_days: int | None = None
    tickers: int | None = None
    allowed_split_mean_effect_bp: float | None = None
    se: float | None = None
    ci: list[float] | None = None
    direction_agreement: float | None = None
    pass_kill_reason: str | None = None
    hypothesis_lines: list[str] = Field(default_factory=list)
    sealed_oos: str = "inaccessible"


class RunStatus(BaseModel):
    run_id: str
    campaign_id: str
    identity: dict[str, str]
    snapshot: TelemetrySnapshot
    stage_complete: dict[str, bool] = Field(default_factory=dict)
    sealed_oos: str = "inaccessible"
