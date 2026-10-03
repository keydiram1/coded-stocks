from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HypothesisState = Literal[
    "REGISTERED",
    "RUNNING_STAGE_1",
    "KILLED_STAGE_1",
    "PASSED_STAGE_1",
    "RUNNING_STAGE_2",
    "KILLED_STAGE_2",
    "PASSED_STAGE_2",
    "RUNNING_FULL_DISCOVERY",
    "KILLED_DISCOVERY",
    "PASSED_DISCOVERY",
    "FALSIFICATION",
    "PROMISING",
    "ERROR",
    "PAUSED",
    "IDEA_ONLY",
]

EventPolicy = Literal["all_clustered", "first_ticker_day", "cooldown"]


class StageCriteria(BaseModel):
    version: str = "stage_criteria_v1"
    min_ticker_days: int = 25
    min_trading_days: int = 8
    kill_if_sign_disagrees: bool = True
    min_abs_mean_to_kill_as_too_small: float | None = None
    max_top_ticker_share: float = 0.50
    primary_horizon: str = "15m"


class Taxonomy(BaseModel):
    time_scale: list[str] = Field(default_factory=list)
    mechanism: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    data_dependencies: list[str] = Field(default_factory=lambda: ["OHLCV"])
    origin: str = "independent"


class FamilyIdea(BaseModel):
    id: str
    title: str
    executable: bool = False
    idea_only_reason: str | None = None
    mechanism: str
    rationale: str
    expected_direction: Literal["long", "short", "either"]
    event_policy: EventPolicy = "first_ticker_day"
    cooldown_minutes: int | None = None
    max_events_per_ticker_day: int | None = 1
    hypothesis_yaml: str | None = None
    engine: str = "unimplemented"
    parameter_neighborhood: list[dict] = Field(default_factory=list)
    stage_criteria: StageCriteria = Field(default_factory=StageCriteria)
    taxonomy: Taxonomy = Field(default_factory=Taxonomy)
    limitations: list[str] = Field(default_factory=list)
    parent_family_id: str | None = None
    derived_from_experiment: str | None = None
    known_weaknesses: list[str] = Field(default_factory=list)


class Checkpoint(BaseModel):
    family_id: str
    variant_id: str = "default"
    state: HypothesisState
    stage: str
    days_done: int = 0
    days_total: int = 0
    events: int = 0
    ticker_days: int = 0
    unique_tickers: int = 0
    trading_days_with_events: int = 0
    interim_means: dict[str, float | None] = Field(default_factory=dict)
    concentration_top_ticker_share: float | None = None
    decision: str | None = None
    decision_reason: str | None = None
    error: str | None = None
    updated_at: str | None = None
    interim: bool = True
