from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PriceRange(BaseModel):
    min: float = 1.0
    max: float = 20.0


class UniverseSpec(BaseModel):
    exchanges: list[str]
    security_type: str = "COMMON_STOCK"
    price_prev_close: PriceRange = Field(default_factory=PriceRange)


class TimeWindowET(BaseModel):
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def hhmm(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError("time_et must be HH:MM")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError("invalid HH:MM")
        return f"{h:02d}:{m:02d}"


class Condition(BaseModel):
    feature: str
    op: Literal[">=", "<=", ">", "<", "=="]
    value: float


class TriggerSpec(BaseModel):
    session: Literal["RTH", "PREMARKET", "ANY"] = "RTH"
    time_et: TimeWindowET
    conditions: list[Condition]


class FeatureRef(BaseModel):
    asof: Literal["event_time"] = "event_time"
    window: str | None = None


class BarrierSpec(BaseModel):
    take_profit: float = 0.03
    stop_loss: float = -0.02


class OutcomeSpec(BaseModel):
    forward_returns: list[str]
    barriers: BarrierSpec = Field(default_factory=BarrierSpec)
    calculate: list[str] = Field(default_factory=lambda: ["MFE", "MAE"])


class ExecutionSpec(BaseModel):
    discovery_model: Literal["next_bar_open"] = "next_bar_open"
    status: Literal["SIGNAL_ONLY"] = "SIGNAL_ONLY"


class StatisticsSpec(BaseModel):
    cluster_by: Literal["symbol_day", "trading_day"] = "symbol_day"
    bootstrap_unit: Literal["trading_day", "symbol_day"] = "trading_day"
    multiple_testing: Literal["BH"] = "BH"
    q_target: float = 0.05


class ValidationSpec(BaseModel):
    chronological: bool = True
    freeze_before_oos: bool = True


class HypothesisSpec(BaseModel):
    id: str
    family: str
    description: str
    side: Literal["long", "short"] = "long"
    universe: UniverseSpec
    trigger: TriggerSpec
    features: dict[str, FeatureRef]
    outcomes: OutcomeSpec
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    statistics: StatisticsSpec = Field(default_factory=StatisticsSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    falsification: list[str] = Field(default_factory=list)
    model_config = {"extra": "forbid"}

    @field_validator("features")
    @classmethod
    def required_features_present(cls, v: dict[str, FeatureRef], info: Any) -> dict[str, FeatureRef]:
        return v
