from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_edge_lab.discovery.v5.continuation import (
    SOURCE_IDENTITY,
    CalendarIdentityError,
    require_continuation_calendar,
)
from quant_edge_lab.discovery.v5.continuation_runner import (
    execute_v5_continuation,
    readiness_v5_continuation,
)
from quant_edge_lab.discovery.v5.preflight import CalendarError
from quant_edge_lab.hashing import sha256_json


def _frozen(parquet_exists=lambda _d: True) -> dict:
    return require_continuation_calendar(Path("."), parquet_exists=parquet_exists)


def test_exact_frozen_calendar_passes():
    rec = _frozen()
    assert rec["calendar_identity_match"] is True
    assert rec["calendar_hash_actual"] == SOURCE_IDENTITY["calendar"]
    assert rec["calendar_hash_expected"] == SOURCE_IDENTITY["calendar"]
    assert rec["calendar_hash"] == sha256_json(rec["days"])
    assert rec["first"] == "2021-10-01"
    assert rec["last"] == "2026-10-01"
    assert rec["counts"]["D1"] == 494
    assert rec["counts"]["D2"] == 247
    assert rec["counts"]["D3"] == 494


def test_remove_one_middle_session_refuses():
    rec = _frozen()
    days = list(rec["days"])
    mid = days[len(days) // 2]
    days.remove(mid)
    with pytest.raises((CalendarIdentityError, CalendarError)):
        require_continuation_calendar(Path("."), days=days, parquet_exists=lambda _d: True)


def test_add_extra_in_panel_session_refuses():
    rec = _frozen()
    days = list(rec["days"])
    extra = "2022-07-03"
    assert extra not in days
    days.append(extra)
    with pytest.raises((CalendarIdentityError, CalendarError)):
        require_continuation_calendar(Path("."), days=days, parquet_exists=lambda _d: True)


def test_reordered_input_still_matches_frozen_calendar():
    rec = _frozen()
    shuffled = list(reversed(rec["days"]))
    again = require_continuation_calendar(
        Path("."), days=shuffled, parquet_exists=lambda _d: True
    )
    assert again["days"] == rec["days"]
    assert again["calendar_hash_actual"] == SOURCE_IDENTITY["calendar"]
    assert again["calendar_hash"] == sha256_json(again["days"])


def test_execute_uses_validated_calendar_object(tmp_path: Path, monkeypatch):
    from quant_edge_lab.discovery.v5.continuation import GATES_REL, MANIFEST_REL

    rec = _frozen()
    dest = tmp_path / "knowledge" / "campaigns"
    dest.mkdir(parents=True)
    man_text = (Path(".") / MANIFEST_REL).read_text(encoding="utf-8")
    gates_text = (Path(".") / GATES_REL).read_text(encoding="utf-8")
    (dest / "v5_continuation_manifest.yaml").write_text(
        man_text.replace("execution_status: NOT_APPROVED", "execution_status: FROZEN", 1),
        encoding="utf-8",
    )
    (dest / "v5_continuation_gates.yaml").write_text(
        gates_text.replace("execution_status: NOT_APPROVED", "execution_status: FROZEN", 1),
        encoding="utf-8",
    )
    ck = (
        tmp_path
        / "data"
        / "derived"
        / "discovery"
        / "v5"
        / "v5-close-dislocation"
        / "checkpoints"
        / "progress.json"
    )
    ck.parent.mkdir(parents=True)
    import json

    ck.write_text(json.dumps({"identity": SOURCE_IDENTITY}), encoding="utf-8")

    captured: dict = {}

    def fake_require(root, *, days=None, parquet_exists=None, expected=None):
        captured["rec"] = rec
        return rec

    def fake_attach(events, calendar):
        captured["used"] = calendar
        return events

    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation_runner.require_continuation_calendar",
        fake_require,
    )
    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation_runner.attach_normalized_dislocation",
        fake_attach,
    )
    ev = pl.DataFrame(
        {
            "instrument_id": ["a"],
            "trading_date": ["2024-11-01"],
            "discrepancy": [0.01],
            "close_volume_rvol": [2.0],
            "next_open_to_15m": [0.001],
        }
    )
    execute_v5_continuation(tmp_path, events=ev)
    assert captured["used"] is captured["rec"]["days"]


def test_calendar_validation_does_not_read_parquet_payload(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("must not read parquet payload")

    monkeypatch.setattr("polars.read_parquet", boom)
    rec = require_continuation_calendar(Path("."), parquet_exists=lambda _d: True)
    assert rec["calendar_identity_match"] is True
    state = readiness_v5_continuation(Path("."))
    assert state["calendar_identity_match"] is True
    assert state["calendar_hash_actual"] == SOURCE_IDENTITY["calendar"]
