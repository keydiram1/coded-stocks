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
    ident = _ident()
    snap = TelemetrySnapshot(stage="D1", completed=1, total=10, events_generated=3, checkpoint_state="OK")
    persist_snapshot(tmp_path, "run-a", snap, extra={"campaign_id": "CLOSE_DISLOCATION_REVERSAL_V1"})
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
    ident = _ident(data_manifest_hash="data1")
    write_ckpt(tmp_path, "run-b", "run_identity", {"complete": True}, ident)
    other = _ident(data_manifest_hash="data2")
    try:
        require_resume_identity(tmp_path, "run-b", other)
        raise AssertionError("expected ResumeIdentityError")
    except ResumeIdentityError as exc:
        assert "data-manifest" in str(exc)


def test_checkpoint_resume_deterministic(tmp_path: Path):
    ident = _ident()
    write_ckpt(tmp_path, "run-c", "d1_rules", {"complete": True, "rules": [{"hypothesis_id": "H1_PRIMARY"}]}, ident)
    a = load_ckpt(tmp_path, "run-c", "d1_rules", ident)
    b = load_ckpt(tmp_path, "run-c", "d1_rules", ident)
    assert a == b
    assert a["complete"] is True


def test_frozen_rule_unchanged_on_later_split():
    rule = CandidateRule(
        hypothesis_id="H1_PRIMARY",
        mechanism_id="forced_eod_auction_flow",
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
    for k in ("manifest_hash", "gates_hash", "data_manifest_hash", "git", "trial_count", "primary_outcome", "sealed_oos", "launch_command"):
        assert k in r
    assert r["partitions"]["D1"]["start"] == "2021-10-29"
    assert set(SPLIT_ORDER) == {"D1", "D2", "D3"}
