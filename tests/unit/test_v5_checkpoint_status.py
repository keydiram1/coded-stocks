from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule
from quant_edge_lab.discovery.v5.checkpoint import load_ckpt, require_resume_identity, write_ckpt
from quant_edge_lab.discovery.v5.evaluation import trial_count
from quant_edge_lab.discovery.v5.identity import ResumeIdentityError, identity_blob
from quant_edge_lab.discovery.v5.manifest import GATES_REL, MANIFEST_REL, freeze_v5, load_v5
from quant_edge_lab.discovery.v5.models import CandidateRule, TelemetrySnapshot
from quant_edge_lab.discovery.v5.partitions import SPLIT_ORDER, assert_no_future_split_in_estimation
from quant_edge_lab.discovery.v5.runner import readiness_v5, run_campaign_v5
from quant_edge_lab.discovery.v5.status import format_status, read_status
from quant_edge_lab.discovery.v5.telemetry import persist_snapshot
from quant_edge_lab.hashing import sha256_file


def _ident(**kw) -> dict[str, str]:
    base = identity_blob(manifest_hash="m", gates_hash="g", git="abc", data_manifest_hash="data1")
    base.update(kw)
    return base


def test_readiness_does_not_execute():
    # Uses repo YAML; must not require --execute and must not open OOS.
    root = Path(".")
    out = run_campaign_v5(root, execute=False)
    assert out["mode"] == "READINESS"
    assert out["trial_count"] == 3
    assert out["sealed_oos"] in {"inaccessible", "closed"}
    assert "--execute" in out["launch_command"]


def test_freeze_includes_data_manifest():
    ident = freeze_v5(Path("."))
    assert "data_manifest" in ident
    assert ident["manifest"] == sha256_file(Path(".") / MANIFEST_REL)
    assert ident["gates"] == sha256_file(Path(".") / GATES_REL)


def test_trial_count_visible():
    man, _gates = load_v5(Path("."))
    camp = man[man["primary_campaign"]]
    assert trial_count(camp) == len(camp["hypotheses"]) == 3


def test_status_read_does_not_mutate(tmp_path: Path):
    snap = TelemetrySnapshot(
        stage="D1", completed=1, total=10, events_generated=3, checkpoint_state="OK"
    )
    persist_snapshot(
        tmp_path, "run-a", snap, extra={"campaign_id": "CLOSE_DISLOCATION_REVERSAL_V1"}
    )
    path = tmp_path / "data" / "derived" / "discovery" / "v5" / "run-a" / "status.json"
    # Paths() redirects only when pyproject exists; tmp has no pyproject so data is under tmp_path.
    if not path.exists():
        from quant_edge_lab.discovery.v5.checkpoint import v5_run_dir

        path = v5_run_dir(tmp_path, "run-a") / "status.json"
    mtime = path.stat().st_mtime_ns
    body = path.read_text(encoding="utf-8")
    rec = read_status(tmp_path, "run-a")
    text = format_status(rec)
    assert "D1" in text
    assert path.stat().st_mtime_ns == mtime
    assert path.read_text(encoding="utf-8") == body


def test_data_manifest_mismatch_refuses_resume(tmp_path: Path):
    ident = _ident(data_manifest="data1")
    write_ckpt(tmp_path, "run-b", "run_identity", {"complete": True}, ident)
    other = _ident(data_manifest="data2")
    try:
        require_resume_identity(tmp_path, "run-b", other)
        raise AssertionError("expected ResumeIdentityError")
    except ResumeIdentityError as exc:
        assert "data-manifest" in str(exc)


def test_checkpoint_resume_deterministic(tmp_path: Path):
    ident = _ident()
    write_ckpt(
        tmp_path,
        "run-c",
        "d1_rules",
        {"complete": True, "rules": [{"hypothesis_id": "H1_PRIMARY"}]},
        ident,
    )
    a = load_ckpt(tmp_path, "run-c", "d1_rules", ident)
    b = load_ckpt(tmp_path, "run-c", "d1_rules", ident)
    assert a == b
    assert a["complete"] is True


def test_frozen_rule_unchanged_on_later_split():
    rule = CandidateRule(
        hypothesis_id="H1_PRIMARY",
        mechanism_id="forced_eod_preclose_flow",
        role="primary",
        dislocation_abs_min=0.02,
        rvol_min=1.5,
        require_rvol=True,
        primary_outcome="next_open_to_15m",
        frozen=True,
    )
    ev = pl.DataFrame(
        {
            "discrepancy": [0.03, 0.01],
            "close_volume_rvol": [2.0, 2.0],
            "instrument_id": ["a", "b"],
            "trading_date": ["2024-01-01", "2024-01-01"],
        }
    )
    a = apply_rule(ev, rule)
    b = apply_rule(ev, rule)
    assert a.height == b.height == 1
    assert rule.dislocation_abs_min == 0.02


def test_d1_estimation_cannot_use_d2_days():
    man = yaml.safe_load((Path(".") / MANIFEST_REL).read_text(encoding="utf-8"))
    assert_no_future_split_in_estimation("D1", ["2022-01-03"], man)
    try:
        assert_no_future_split_in_estimation("D1", ["2024-11-01"], man)
        raise AssertionError("expected later-split rejection")
    except AssertionError as exc:
        assert "later split" in str(exc)


def test_readiness_helper_matches_cli_contract():
    r = readiness_v5(Path("."))
    for k in (
        "manifest_hash",
        "gates_hash",
        "data_manifest_hash",
        "git",
        "trial_count",
        "primary_outcome",
        "sealed_oos",
        "launch_command",
        "execution_status",
    ):
        assert k in r
    assert r["partitions"]["D1"]["start"] == "2021-10-29"
    assert set(SPLIT_ORDER) == {"D1", "D2", "D3"}
    assert r["execution_status"] == "FROZEN"


def test_execute_refused_until_approved():
    from quant_edge_lab.discovery.v5.runner import assert_execution_approved

    man, gates = load_v5(Path("."))
    assert_execution_approved(man, gates)
    both = {**man, "execution_status": "APPROVED"}, {**gates, "execution_status": "FROZEN"}
    assert_execution_approved(*both)
    try:
        assert_execution_approved(
            {**man, "execution_status": "APPROVED"},
            {**gates, "execution_status": "NOT_APPROVED"},
        )
        raise AssertionError("gates not approved must refuse")
    except RuntimeError:
        pass
    try:
        assert_execution_approved(
            {**man, "execution_status": "NOT_APPROVED"},
            {**gates, "execution_status": "APPROVED"},
        )
        raise AssertionError("manifest not approved must refuse")
    except RuntimeError:
        pass
    try:
        assert_execution_approved(
            {**man, "execution_status": "NOT_APPROVED"},
            {**gates, "execution_status": "NOT_APPROVED"},
        )
        raise AssertionError("both not approved must refuse")
    except RuntimeError:
        pass


def test_interrupted_resume_matches_uninterrupted(tmp_path: Path):
    from datetime import date, timedelta

    import polars as pl
    import pytest

    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days
    from quant_edge_lab.discovery.v5.checkpoint import DayStore, InterruptAfter
    from tests.unit.test_v5_close_dislocation import _close_day

    start = date(2024, 2, 1)
    bars = {}
    for i in range(8):
        d = start + timedelta(days=i)
        bars[d.isoformat()] = pl.concat(
            [
                _close_day(d, iid="a", ticker="A", last_step=0.01),
                _close_day(d, iid="b", ticker="B", last_step=-0.01),
            ]
        )
    keys = sorted(bars)

    def load_day(day: str):
        return bars[day]

    def load_next(day: str):
        i = keys.index(day)
        return bars[keys[i + 1]] if i + 1 < len(keys) else None

    ident = _ident()
    store_u = DayStore(tmp_path, "unint", ident)
    full, r_full = build_events_from_days(
        keys, load_day, load_next, min_rth_minutes=6, rvol_lookback=20, store=store_u
    )
    store_i = DayStore(tmp_path, "intr", ident)
    with pytest.raises(InterruptAfter):
        build_events_from_days(
            keys,
            load_day,
            load_next,
            min_rth_minutes=6,
            rvol_lookback=20,
            store=store_i,
            interrupt_after=3,
        )
    resumed, r_res = build_events_from_days(
        keys, load_day, load_next, min_rth_minutes=6, rvol_lookback=20, store=store_i
    )
    assert store_i.last_completed_day() == store_u.last_completed_day() == keys[-1]
    bu = store_u.load_beta_history()
    br = store_i.load_beta_history()
    assert len(bu) == len(br)
    for a, b in zip(bu, br, strict=True):
        assert a.sort("instrument_id").to_dicts() == b.sort("instrument_id").to_dicts()
    assert r_full == r_res
    if full.height or resumed.height:
        assert (
            full.sort(["instrument_id", "trading_date"]).to_dicts()
            == resumed.sort(["instrument_id", "trading_date"]).to_dicts()
        )


def test_crash_before_progress_is_harmless(tmp_path: Path):
    from datetime import date, timedelta

    import pytest

    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days
    from quant_edge_lab.discovery.v5.checkpoint import CrashAfter, DayStore
    from tests.unit.test_v5_close_dislocation import _close_day

    start = date(2024, 3, 1)
    bars = {}
    for i in range(5):
        d = start + timedelta(days=i)
        bars[d.isoformat()] = pl.concat(
            [
                _close_day(d, iid="a", ticker="A", last_step=0.01),
                _close_day(d, iid="b", ticker="B", last_step=-0.01),
            ]
        )
    keys = sorted(bars)

    def load_day(day: str):
        return bars[day]

    def load_next(day: str):
        i = keys.index(day)
        return bars[keys[i + 1]] if i + 1 < len(keys) else None

    ident = _ident()
    store_u = DayStore(tmp_path, "clean", ident)
    full, r_full = build_events_from_days(
        keys, load_day, load_next, min_rth_minutes=6, rvol_lookback=20, store=store_u
    )
    crash_day = keys[1]
    for point in ("beta", "rvol", "events", "rotate", "progress"):
        store = DayStore(tmp_path, f"crash-{point}", ident)
        with pytest.raises(CrashAfter):
            build_events_from_days(
                keys,
                load_day,
                load_next,
                min_rth_minutes=6,
                rvol_lookback=20,
                store=store,
                crash_after=point,
                crash_day=crash_day,
            )
        if point == "progress":
            assert store.last_completed_day() == crash_day
        else:
            assert store.last_completed_day() == keys[0]
        resumed, r_res = build_events_from_days(
            keys, load_day, load_next, min_rth_minutes=6, rvol_lookback=20, store=store
        )
        assert store.last_completed_day() == store_u.last_completed_day()
        bu = (
            pl.concat(store_u.load_beta_history())
            if store_u.load_beta_history()
            else pl.DataFrame()
        )
        br = pl.concat(store.load_beta_history()) if store.load_beta_history() else pl.DataFrame()
        if bu.height:
            assert (
                bu.sort(["trading_date", "instrument_id"]).to_dicts()
                == br.sort(["trading_date", "instrument_id"]).to_dicts()
            )
        assert r_full == r_res
        if full.height:
            assert (
                full.sort(["instrument_id", "trading_date"]).to_dicts()
                == resumed.sort(["instrument_id", "trading_date"]).to_dicts()
            )
        assert not any(n.startswith("day=") for n in store.generation_names())
        assert set(store_u.generation_names()) <= {"current", "previous"}


def test_state_generations_are_bounded(tmp_path: Path):
    from datetime import date, timedelta

    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days
    from quant_edge_lab.discovery.v5.checkpoint import DayStore
    from tests.unit.test_v5_close_dislocation import _close_day

    start = date(2024, 5, 1)
    bars = {}
    for i in range(6):
        d = start + timedelta(days=i)
        bars[d.isoformat()] = pl.concat(
            [
                _close_day(d, iid="a", ticker="A", last_step=0.01),
                _close_day(d, iid="b", ticker="B", last_step=-0.01),
            ]
        )
    keys = sorted(bars)

    def load_day(day: str):
        return bars[day]

    def load_next(day: str):
        i = keys.index(day)
        return bars[keys[i + 1]] if i + 1 < len(keys) else None

    store = DayStore(tmp_path, "gens", _ident())
    build_events_from_days(
        keys, load_day, load_next, min_rth_minutes=6, rvol_lookback=20, store=store
    )
    names = store.generation_names()
    assert set(names) <= {"current", "previous"}
    assert "current" in names
    assert len(names) <= 2
    assert not any(n.startswith("day=") for n in names)


def test_missing_committed_state_refuses_resume(tmp_path: Path):
    from datetime import date, timedelta

    import pytest

    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days
    from quant_edge_lab.discovery.v5.checkpoint import DayStore
    from quant_edge_lab.discovery.v5.identity import ResumeIdentityError
    from tests.unit.test_v5_close_dislocation import _close_day

    start = date(2024, 7, 1)
    bars = {}
    for i in range(3):
        d = start + timedelta(days=i)
        bars[d.isoformat()] = pl.concat(
            [
                _close_day(d, iid="a", ticker="A", last_step=0.01),
                _close_day(d, iid="b", ticker="B", last_step=-0.01),
            ]
        )
    keys = sorted(bars)

    def load_day(day: str):
        return bars[day]

    def load_next(day: str):
        i = keys.index(day)
        return bars[keys[i + 1]] if i + 1 < len(keys) else None

    for missing in ("beta_history.parquet", "rvol_state.json", "elig_state.json"):
        store = DayStore(tmp_path, f"miss-{missing}", _ident())
        build_events_from_days(
            keys, load_day, load_next, min_rth_minutes=6, rvol_lookback=20, store=store
        )
        path = store._gen("current") / missing
        path.unlink()
        with pytest.raises(ResumeIdentityError, match="missing committed"):
            if missing.startswith("beta"):
                store.load_beta_history()
            elif missing.startswith("rvol"):
                store.load_rvol()
            else:
                store.load_elig()
