from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from quant_edge_lab.data.massive.news import (
    CANONICAL_UTC,
    PRIMARY_LAG_MIN,
    attach_news_state,
    flatten_article,
    insights_with_lag,
    load_insights,
    news_root,
    validate_news_cache,
    with_utc_us,
)
from quant_edge_lab.discovery.campaign_v3_runner import EXPECTED_GATES, EXPECTED_MANIFEST, freeze_campaign
from quant_edge_lab.discovery.dir_gates import decide_directional, load_gates_from


def test_v3_freeze_stable():
    root = Path(".")
    a = freeze_campaign(root)
    b = freeze_campaign(root)
    assert a["manifest"] == EXPECTED_MANIFEST
    assert a["gates"] == EXPECTED_GATES
    man = yaml.safe_load((root / "knowledge/campaigns/directional_v3_manifest.yaml").read_text(encoding="utf-8"))
    assert man["n_jobs"] == len(man["jobs"]) == 57


def test_news_available_at_is_published_plus_60():
    df = pl.DataFrame(
        {
            "ticker": ["AAA"],
            "published_ts": [datetime(2026, 9, 1, 14, 0, tzinfo=UTC)],
            "sentiment_score": [1],
            "article_id": ["x"],
            "publisher": ["Z"],
            "article_ticker_count": [1],
        }
    )
    out = insights_with_lag(df, PRIMARY_LAG_MIN)
    assert out["available_at"][0] == datetime(2026, 9, 1, 15, 0, tzinfo=UTC)


def test_future_news_does_not_change_past_decision():
    ins = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "published_ts": [
                datetime(2026, 9, 1, 13, 0, tzinfo=UTC),
                datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
            ],
            "sentiment_score": [1, -1],
            "article_id": ["a", "b"],
            "publisher": ["Z", "Z"],
            "article_ticker_count": [1, 1],
        }
    )
    ev = pl.DataFrame(
        {
            "event_id": ["e1"],
            "ticker": ["AAA"],
            "decision_ts": [datetime(2026, 9, 1, 14, 30, tzinfo=UTC)],
            "instrument_id": ["ticker:AAA"],
        }
    )
    out = attach_news_state(ev, ins, lag_min=60)
    assert "sentiment_reasoning" not in out.columns
    assert out["latest_news_sentiment"][0] == 1


def test_v3_horizon_floor_stricter_than_15m():
    root = Path(".")
    gates = load_gates_from(root, Path("knowledge/campaigns/directional_v3_gates.yaml"))
    raw = yaml.safe_load((root / "knowledge/campaigns/directional_v3_gates.yaml").read_text(encoding="utf-8"))
    g = gates.stages["fast"]
    d15, _ = decide_directional(
        mean_primary=0.0013,
        median_primary=0.001,
        win_rate=0.55,
        ticker_days=40,
        trading_days=10,
        top_ticker_share=0.1,
        frac_days_agree=0.6,
        ci=[0.001, 0.002],
        gate=g,
        stage_complete=True,
        extras={"economic_floor": raw["economic_floors"]["15m"]},
    )
    d60, reason = decide_directional(
        mean_primary=0.0013,
        median_primary=0.001,
        win_rate=0.55,
        ticker_days=40,
        trading_days=10,
        top_ticker_share=0.1,
        frac_days_agree=0.6,
        ci=[0.001, 0.002],
        gate=g,
        stage_complete=True,
        extras={"economic_floor": raw["economic_floors"]["60m"]},
    )
    assert d15 == "PASS"
    assert d60 == "KILL"
    assert "0.0024" in reason


def test_flatten_skips_non_enum_sentiment():
    rows = flatten_article(
        {
            "id": "1",
            "published_utc": "2026-09-01T00:00:00Z",
            "tickers": ["A"],
            "insights": [
                {"ticker": "A", "sentiment": "positive", "sentiment_reasoning": "x"},
                {"ticker": "A", "sentiment": "very positive", "sentiment_reasoning": "y"},
            ],
            "publisher": {"name": "Z"},
        }
    )
    assert len(rows) == 1
    assert rows[0]["sentiment_score"] == 1
    assert rows[0]["sentiment_reasoning"] == "x"


def _insights(published, score=1, ticker="AAA", aid="a"):
    return pl.DataFrame(
        {
            "ticker": [ticker],
            "published_ts": [published],
            "sentiment_score": [score],
            "article_id": [aid],
            "publisher": ["Z"],
            "article_ticker_count": [1],
        }
    )


def _events(decision, ticker="AAA", eid="e1"):
    return pl.DataFrame(
        {
            "event_id": [eid],
            "ticker": [ticker],
            "decision_ts": [decision],
            "instrument_id": [f"ticker:{ticker}"],
        }
    )


def test_join_key_dtypes_identical_before_asof():
    ins = insights_with_lag(_insights(datetime(2026, 9, 1, 14, 0, tzinfo=UTC)), 60)
    ev = with_utc_us(_events(datetime(2026, 9, 1, 15, 30)), "decision_ts")
    assert ins.schema["available_at"] == CANONICAL_UTC
    assert ev.schema["decision_ts"] == CANONICAL_UTC
    assert ins.schema["available_at"] == ev.schema["decision_ts"]
    out = attach_news_state(_events(datetime(2026, 9, 1, 15, 30)), _insights(datetime(2026, 9, 1, 14, 0, tzinfo=UTC)))
    assert out["latest_news_sentiment"][0] == 1
    assert out.schema["decision_ts"] == CANONICAL_UTC
    assert out.schema["latest_news_available_at"] == CANONICAL_UTC


def test_known_timestamp_joins_expected_article():
    ins = pl.concat(
        [
            _insights(datetime(2026, 9, 1, 13, 0, tzinfo=UTC), score=1, aid="old"),
            _insights(datetime(2026, 9, 1, 14, 0, tzinfo=UTC), score=-1, aid="target"),
        ]
    )
    out = attach_news_state(_events(datetime(2026, 9, 1, 15, 30, tzinfo=UTC)), ins)
    assert out["latest_article_id"][0] == "target"
    assert out["latest_news_sentiment"][0] == -1


def test_pit_boundary_plus_60_minutes():
    ins = _insights(datetime(2026, 9, 1, 14, 0, tzinfo=UTC), aid="pit")
    hidden = attach_news_state(_events(datetime(2026, 9, 1, 14, 59, tzinfo=UTC)), ins)
    at = attach_news_state(_events(datetime(2026, 9, 1, 15, 0, tzinfo=UTC)), ins)
    after = attach_news_state(_events(datetime(2026, 9, 1, 15, 1, tzinfo=UTC)), ins)
    assert hidden["latest_article_id"][0] is None or hidden["latest_news_sentiment"][0] is None
    assert after["latest_article_id"][0] == "pit"
    # Equality at available_at is allowed (MAY); we include it.
    assert at["latest_article_id"][0] == "pit"


def test_future_news_poison_unobservable():
    ins = pl.concat(
        [
            _insights(datetime(2026, 9, 1, 12, 0, tzinfo=UTC), score=1, aid="past"),
            _insights(datetime(2026, 9, 1, 18, 0, tzinfo=UTC), score=-1, aid="FUTURE_POISON"),
        ]
    )
    out = attach_news_state(_events(datetime(2026, 9, 1, 14, 30, tzinfo=UTC)), ins)
    assert out["latest_article_id"][0] == "past"
    assert out["latest_news_sentiment"][0] == 1
    assert "FUTURE_POISON" not in str(out["latest_article_id"][0])


def test_utc_normalization_preserves_instant():
    instant = datetime(2025, 1, 15, 15, 30, 0, tzinfo=UTC)
    naive = datetime(2025, 1, 15, 15, 30, 0)
    df = pl.DataFrame({"naive": [naive], "aware": [instant]})
    out = with_utc_us(with_utc_us(df, "naive"), "aware")
    assert out.schema["naive"] == CANONICAL_UTC
    assert out.schema["aware"] == CANONICAL_UTC
    assert out["naive"][0] == instant
    assert out["aware"][0] == instant
    assert int(out["naive"].dt.timestamp("us")[0]) == int(instant.timestamp() * 1_000_000)
    assert out["naive"].dt.timestamp("us")[0] == out["aware"].dt.timestamp("us")[0]


def test_real_news_cache_join_smoke():
    import pytest

    root = Path(".")
    if not (news_root(root) / "insights_all.parquet").exists():
        pytest.skip("news cache not present")
    stats = validate_news_cache(root)
    assert stats["n_insight_rows"] > 0
    assert stats["null_sentiment"] == 0
    ins = load_insights(root).head(5000)
    from quant_edge_lab.features.causal_store import store_dir

    feat_dir = store_dir(root)
    days = sorted(feat_dir.glob("date=*/part.parquet"))
    if not days:
        # Fall back to real insight tickers + real published instants as market decision times.
        sample = ins.filter(pl.col("ticker").is_not_null()).head(200)
        ev = sample.select(
            pl.col("ticker"),
            (pl.col("published_ts") + pl.duration(minutes=90)).alias("decision_ts"),
            (pl.col("ticker") + pl.lit("|") + pl.col("published_ts").cast(pl.Utf8)).alias("event_id"),
            (pl.lit("ticker:") + pl.col("ticker")).alias("instrument_id"),
        )
        out = attach_news_state(ev, sample)
    else:
        feat = pl.read_parquet(days[-1]).head(300)
        if "ticker" not in feat.columns:
            feat = feat.with_columns(pl.col("instrument_id").str.replace("^ticker:", "").alias("ticker"))
        feat = feat.with_columns((pl.col("instrument_id") + pl.lit("|") + pl.col("ts_utc").cast(pl.Utf8)).alias("event_id"))
        tickers = set(feat["ticker"].unique().to_list())
        day_ins = ins.filter(pl.col("ticker").is_in(list(tickers))).head(8000)
        if day_ins.height == 0:
            pytest.skip("no overlapping tickers in sample")
        out = attach_news_state(feat, day_ins)
    assert out.height > 0
    joined = out.filter(pl.col("latest_news_available_at").is_not_null())
    assert joined.height > 0
    labels = set(joined["latest_news_sentiment"].drop_nulls().to_list())
    assert labels & {1, 0, -1}
    late = joined.filter(pl.col("latest_news_available_at") > pl.col("decision_ts"))
    assert late.height == 0
    assert "sentiment_reasoning" not in out.columns

