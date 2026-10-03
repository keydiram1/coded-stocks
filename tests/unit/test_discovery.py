from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from quant_edge_lab.discovery.catalog import all_families, family_by_id, write_registry
from quant_edge_lab.discovery.event_policy import apply_event_policy
from quant_edge_lab.discovery.funnel import decide_stage
from quant_edge_lab.discovery.knowledge import import_experiment_001, search_knowledge
from quant_edge_lab.discovery.lineage import classify_pair
from quant_edge_lab.discovery.models import StageCriteria
from quant_edge_lab.discovery.runner import run_batch
from quant_edge_lab.discovery.stages import select_stage_days


def test_stage_panels_deterministic():
    days = [f"2021-10-{d:02d}" for d in range(1, 21)] + [f"2024-03-{d:02d}" for d in range(1, 21)]
    a = select_stage_days(days, "fast", n_fast=10)
    b = select_stage_days(days, "fast", n_fast=10)
    assert a == b
    assert len(a) == 10
    full = select_stage_days(days, "full")
    assert full == sorted(days)


def test_kill_and_promote_preregistered():
    c = StageCriteria(min_ticker_days=5, min_trading_days=3, primary_horizon="15m")
    d, _ = decide_stage(
        expected_direction="short",
        ticker_days=10,
        trading_days=5,
        mean_primary=-0.01,
        top_ticker_share=0.1,
        criteria=c,
        stage_complete=True,
    )
    assert d == "PASS"
    d, reason = decide_stage(
        expected_direction="short",
        ticker_days=10,
        trading_days=5,
        mean_primary=0.02,
        top_ticker_share=0.1,
        criteria=c,
        stage_complete=True,
    )
    assert d == "KILL" and "sign disagrees" in reason
    d, reason = decide_stage(
        expected_direction="long",
        ticker_days=1,
        trading_days=1,
        mean_primary=0.05,
        top_ticker_share=0.0,
        criteria=c,
        stage_complete=True,
    )
    assert d == "KILL" and "insufficient" in reason
    d, _ = decide_stage(
        expected_direction="long",
        ticker_days=10,
        trading_days=5,
        mean_primary=0.01,
        top_ticker_share=0.1,
        criteria=c,
        stage_complete=False,
    )
    assert d == "CONTINUE"


def test_first_event_policy():
    ev = pl.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "session_date": ["2021-10-04", "2021-10-04", "2021-10-04"],
            "decision_ts": [
                datetime(2021, 10, 4, 13, 35),
                datetime(2021, 10, 4, 14, 0),
                datetime(2021, 10, 4, 13, 40),
            ],
        }
    ).with_columns(pl.col("decision_ts").cast(pl.Datetime("us")))
    out = apply_event_policy(ev, "first_ticker_day")
    assert out.height == 2
    assert out.filter(pl.col("ticker") == "A").height == 1


def test_lineage_and_registry():
    fams = all_families()
    assert len(fams) == 41  # H000–H040
    assert family_by_id("H001").executable
    assert not family_by_id("H006").executable
    assert classify_pair(family_by_id("H000"), family_by_id("H001")) in {
        "DERIVED",
        "DERIVED_OPPOSITE_DIRECTION",
        "RELATED_OR_UNCLEAR",
    }
    assert classify_pair(family_by_id("H001"), family_by_id("H040")) == "DIFFERENT_MECHANISM"


def test_exp1_frozen_hash():
    freeze = Path("experiments/exp-gap_rvol_continuation_v1-20261002075236/hypothesis_freeze.json")
    assert freeze.exists()
    text = freeze.read_text(encoding="utf-8")
    assert "783aeaac9d848b1368613720f3a2f6aeccd8c4e3fd553071b07159b8acb4bd20" in text
    yaml = Path("hypotheses/gap_rvol_continuation_v1.yaml").read_text(encoding="utf-8")
    assert "side: long" in yaml


def _minute_day(day: datetime, start_et_hour: int, n: int, px: float, vol: float, ticker: str = "AAA"):
    # day is UTC timestamp of 09:30 ET: EDT = UTC+4 in October 2021 → 13:30 UTC
    rows = []
    for i in range(n):
        ts = day + timedelta(minutes=i)
        rows.append(
            {
                "instrument_id": f"ticker:{ticker}",
                "ticker": ticker,
                "ts_utc": ts,
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": vol,
                "transactions": 10,
                "source": "test",
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))


def test_batch_one_error_does_not_kill_other(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hypotheses" / "discovery").mkdir(parents=True)
    src = Path(__file__).resolve().parents[2]
    (tmp_path / "hypotheses" / "discovery" / "H001_gap_fade.yaml").write_text(
        (src / "hypotheses" / "discovery" / "H001_gap_fade.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "hypotheses" / "discovery" / "H040_monday_gap.yaml").write_text(
        (src / "hypotheses" / "discovery" / "H040_monday_gap.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    inst = pl.DataFrame(
        {
            "instrument_id": ["ticker:AAA"],
            "ticker": ["AAA"],
            "exchange": ["NASDAQ"],
            "security_type": ["COMMON_STOCK"],
        }
    )
    # Fri 2021-10-01 09:30 ET = 13:30 UTC
    d1 = datetime(2021, 10, 1, 13, 30)
    d2 = datetime(2021, 10, 4, 13, 30)  # Monday
    bars1 = _minute_day(d1, 9, 390, 10.0, 1_000.0)
    bars2 = _minute_day(d2, 9, 180, 12.0, 500_000.0)
    store = {"2021-10-01": bars1, "2021-10-04": bars2}

    def loader(day: str) -> pl.DataFrame:
        return store[day]

    state = run_batch(
        tmp_path,
        ["H001", "H040"],
        batch_id="t1",
        stage="fast",
        day_loader=loader,
        all_days=["2021-10-01", "2021-10-04"],
        instruments=inst,
        print_every=1,
    )
    assert "H001" in state["jobs"]
    assert "H040" in state["jobs"]
    assert state["jobs"]["H001"]["state"] != "ERROR" or state["jobs"]["H040"]["state"] != "ERROR"
    # resume does not crash
    state2 = run_batch(
        tmp_path,
        ["H001", "H040"],
        batch_id="t1",
        stage="fast",
        resume=True,
        day_loader=loader,
        all_days=["2021-10-01", "2021-10-04"],
        instruments=inst,
    )
    assert state2["jobs"]["H001"]["days_done"] >= 1


def test_knowledge_import(tmp_path: Path):
    p = import_experiment_001(tmp_path)
    assert p.exists()
    hits = search_knowledge(tmp_path, "rvol")
    assert hits
