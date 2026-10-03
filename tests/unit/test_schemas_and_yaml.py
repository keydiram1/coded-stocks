from pathlib import Path

from quant_edge_lab.hypotheses.loader import hypothesis_hash, load_hypothesis
from quant_edge_lab.models.schemas import Bar1m, ExperimentRecord, FloatObservation, ResearchEvent


ROOT = Path(__file__).resolve().parents[2]


def test_yaml_hypotheses_validate():
    for name in (
        "gap_rvol_continuation_v1.yaml",
        "extreme_extension_fade_v1.yaml",
        "premarket_high_behavior_v1.yaml",
    ):
        spec = load_hypothesis(ROOT / "hypotheses" / name)
        assert spec.id
        assert spec.features
        assert spec.trigger.conditions
        assert spec.execution.status == "SIGNAL_ONLY"
        h1 = hypothesis_hash(spec)
        h2 = hypothesis_hash(load_hypothesis(ROOT / "hypotheses" / name))
        assert h1 == h2


def test_schema_models_construct():
    from datetime import UTC, datetime

    Bar1m(
        instrument_id="x",
        ts_utc=datetime.now(UTC),
        open=1,
        high=1.1,
        low=0.9,
        close=1.0,
        volume=10,
    )
    FloatObservation(
        instrument_id="x",
        effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        float_shares=1e6,
        source="test",
    )
    ResearchEvent(
        event_id="e",
        experiment_id="x",
        instrument_id="x",
        ticker="T",
        event_ts=datetime.now(UTC),
        decision_ts=datetime.now(UTC),
        feature_version="v",
        side="long",
        features={"gap_pct": 0.2},
    )
