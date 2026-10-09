from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import apply_rule
from quant_edge_lab.discovery.v5.continuation import (
    CAMPAIGN_ID,
    GATES_REL,
    H1_DISLOC,
    H1_RVOL,
    H2_DISLOC,
    H2_RVOL,
    H3_DISLOC,
    MANIFEST_REL,
    SCIENCE_ID,
    SOURCE_IDENTITY,
    SOURCE_RUN_ID,
    attach_continuation,
    continuation_direction,
    continuation_signed,
    freeze_continuation,
    frozen_rules,
    load_continuation,
    refuse_discovery_split,
    reversal_signed,
)
from quant_edge_lab.discovery.v5.continuation_runner import (
    evaluate_continuation_d3,
    execute_v5_continuation,
    readiness_v5_continuation,
    run_campaign_v5_continuation,
)
from quant_edge_lab.discovery.v5.evaluation import decide
from quant_edge_lab.discovery.v5.manifest import GATES_REL as REVERSAL_GATES
from quant_edge_lab.discovery.v5.manifest import MANIFEST_REL as REVERSAL_MANIFEST
from quant_edge_lab.hashing import sha256_file


def test_positive_discrepancy_positive_forward_is_positive_continuation():
    assert continuation_direction(0.01) == "LONG"
    assert continuation_signed(0.02, 0.01) == pytest.approx(0.02)


def test_positive_discrepancy_negative_forward_is_negative_continuation():
    assert continuation_signed(-0.02, 0.01) == pytest.approx(-0.02)


def test_negative_discrepancy_negative_forward_is_positive_continuation():
    assert continuation_direction(-0.01) == "SHORT"
    assert continuation_signed(-0.02, -0.01) == pytest.approx(0.02)


def test_continuation_signed_negates_reversal_primary_signed():
    pairs = [(0.03, 0.01), (-0.04, 0.02), (0.05, -0.03), (-0.01, -0.02)]
    for fwd, disc in pairs:
        c = continuation_signed(fwd, disc)
        r = reversal_signed(fwd, disc)
        assert c is not None and r is not None
        assert c == pytest.approx(-r)


def test_absolute_family_unchanged_and_normalized_family_separate():
    man, _g = load_continuation(Path("."))
    camp = man[man["primary_campaign"]]
    assert camp["absolute_family_trial_count"] == 3
    assert camp["normalized_family_trial_count"] == 2
    assert camp["total_trial_count"] == 5
    assert camp["normalized_scale"]["threshold_abs_z"] == 2.0
    assert camp["normalized_scale"]["lookback_sessions"] == 20
    assert camp["normalized_scale"]["scale_lookback_sessions"] == 20
    assert camp["normalized_scale"]["scale_requires_all_sessions"] is True
    assert camp["primary_estimand"] == "equal_weight_trading_day_mean"
    vn = camp.get("volatility_normalization") or {}
    assert vn.get("in_this_campaign") is True
    assert man["execution_status"] == "NOT_APPROVED"


def test_frozen_thresholds_cannot_drift():
    man, _gates = load_continuation(Path("."))
    rules = frozen_rules(man[man["primary_campaign"]])
    by = {r.hypothesis_id: r for r in rules}
    assert by["H1_PRIMARY"].dislocation_abs_min == H1_DISLOC
    assert by["H1_PRIMARY"].rvol_min == H1_RVOL
    assert by["H2_ABS_Q95"].dislocation_abs_min == H2_DISLOC
    assert by["H2_ABS_Q95"].rvol_min == H2_RVOL
    assert by["H3_DISLOCATION_ONLY"].dislocation_abs_min == H3_DISLOC
    assert by["H3_DISLOCATION_ONLY"].rvol_min is None
    assert by["H3_DISLOCATION_ONLY"].require_rvol is False
    assert by["H1_PRIMARY"].z_abs_min is None
    assert by["H4_Z2_RVOL"].z_abs_min == 2.0
    assert by["H5_Z2_ONLY"].z_abs_min == 2.0
    assert by["H4_Z2_RVOL"].dislocation_abs_min is None
    assert by["H5_Z2_ONLY"].dislocation_abs_min is None
    assert by["H4_Z2_RVOL"].rvol_min == H1_RVOL
    assert by["H5_Z2_ONLY"].require_rvol is False


def test_h1_remains_primary():
    man, _g = load_continuation(Path("."))
    camp = man[man["primary_campaign"]]
    roles = {h["hypothesis_id"]: h["role"] for h in camp["hypotheses"]}
    assert roles["H1_PRIMARY"] == "primary"
    assert roles["H2_ABS_Q95"] == "robustness"
    rules = frozen_rules(camp)
    assert rules[0].hypothesis_id == "H1_PRIMARY"
    assert rules[0].role == "primary"
    assert sum(1 for r in rules if r.role == "primary") == 1
    ids = [r.hypothesis_id for r in rules]
    assert ids == [
        "H1_PRIMARY",
        "H2_ABS_Q95",
        "H3_DISLOCATION_ONLY",
        "H4_Z2_RVOL",
        "H5_Z2_ONLY",
    ]


def test_only_d3_is_evaluable():
    with pytest.raises(RuntimeError, match="D1/D2"):
        refuse_discovery_split("D1")
    with pytest.raises(RuntimeError, match="D1/D2"):
        refuse_discovery_split("D2")
    refuse_discovery_split("D3")


def test_d1_d2_empirical_evaluation_refused():
    ev = _synth_events(0.002)
    rules = frozen_rules()
    gates = _loose_gates()
    with pytest.raises(RuntimeError, match="refuses D1/D2"):
        evaluate_continuation_d3(ev, rules, gates, split="D1")
    with pytest.raises(RuntimeError, match="refuses D1/D2"):
        evaluate_continuation_d3(ev, rules, gates, split="D2")


def test_execute_refuses_not_approved():
    from quant_edge_lab.discovery.v5.continuation import assert_execution_approved

    man, gates = load_continuation(Path("."))
    assert man["execution_status"] == "NOT_APPROVED"
    assert gates["execution_status"] == "NOT_APPROVED"
    with pytest.raises(RuntimeError, match="refused"):
        assert_execution_approved(man, gates)
    with pytest.raises(RuntimeError, match="refused"):
        run_campaign_v5_continuation(Path("."), execute=True)


def test_source_identity_mismatch_refuses(tmp_path: Path):
    root = _approved_tree(tmp_path)
    ck = (
        root
        / "data"
        / "derived"
        / "discovery"
        / "v5"
        / SOURCE_RUN_ID
        / "checkpoints"
        / "progress.json"
    )
    ck.parent.mkdir(parents=True)
    bad = dict(SOURCE_IDENTITY)
    bad["git"] = "deadbeef"
    ck.write_text(json.dumps({"identity": bad}), encoding="utf-8")
    with pytest.raises(Exception, match="identity mismatch"):
        execute_v5_continuation(root, events=_synth_events(0.002))


def test_d3_bh_across_exactly_five_hypotheses():
    ev = _synth_events(0.002)
    rows = evaluate_continuation_d3(ev, frozen_rules(), _loose_gates(), split="D3")
    assert len(rows) == 5
    assert {r["rule"]["hypothesis_id"] for r in rows} == {
        "H1_PRIMARY",
        "H2_ABS_Q95",
        "H3_DISLOCATION_ONLY",
        "H4_Z2_RVOL",
        "H5_Z2_ONLY",
    }
    assert all(r["tested_this_stage"] == 5 for r in rows)
    assert all(r["trial_count"] == 5 for r in rows)
    assert all("bh_rejected" in r["stats"] for r in rows)


def test_significant_five_bp_is_subthreshold():
    st = _ok_stats(0.0005)
    d3 = decide(
        split="D3",
        hypothesis_id="H1_PRIMARY",
        stats=st,
        gates=_loose_gates(),
        bh_survivor=True,
    )
    assert d3.label == "VALIDATED_SUBTHRESHOLD_PHENOMENON"


def test_significant_fifteen_bp_is_research_pass():
    st = _ok_stats(0.0015)
    d3 = decide(
        split="D3",
        hypothesis_id="H1_PRIMARY",
        stats=st,
        gates=_loose_gates(),
        bh_survivor=True,
    )
    assert d3.label == "RESEARCH_PASS"


def test_positive_nonsignificant_is_kill():
    st = _ok_stats(0.002)
    d3 = decide(
        split="D3",
        hypothesis_id="H1_PRIMARY",
        stats=st,
        gates=_loose_gates(),
        bh_survivor=False,
    )
    assert d3.label == "KILL"
    assert d3.reason == "d3_not_significant"


def test_negative_mean_is_kill():
    st = _ok_stats(-0.001)
    d3 = decide(
        split="D3",
        hypothesis_id="H1_PRIMARY",
        stats=st,
        gates=_loose_gates(),
        bh_survivor=True,
    )
    assert d3.label == "KILL"
    assert d3.reason == "direction_failed"


def test_secondary_horizons_cannot_alter_decision():
    ev = _synth_events(0.002)
    gates = _loose_gates()
    rules = frozen_rules()
    a = evaluate_continuation_d3(ev, rules, gates)
    mutated = ev.with_columns(
        pl.lit(-0.5).alias("next_open_to_5m"),
        pl.lit(0.5).alias("next_open_to_30m"),
    )
    b = evaluate_continuation_d3(mutated, rules, gates)
    assert [r["decision"] for r in a] == [r["decision"] for r in b]


def test_synthetic_d3_does_not_touch_real_d3_files(tmp_path: Path, monkeypatch):
    root = _approved_tree(tmp_path)
    ck = (
        root
        / "data"
        / "derived"
        / "discovery"
        / "v5"
        / SOURCE_RUN_ID
        / "checkpoints"
        / "progress.json"
    )
    ck.parent.mkdir(parents=True)
    ck.write_text(json.dumps({"identity": SOURCE_IDENTITY}), encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("real D3 payload must not be loaded")

    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation_runner.load_continuation_source_events", boom
    )
    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation.load_continuation_source_events", boom
    )
    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation.load_pre_d3_event_partitions", boom
    )
    fake_days = ["2024-11-01"]
    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation_runner.require_continuation_calendar",
        lambda *_a, **_k: {"days": fake_days},
    )
    out = execute_v5_continuation(root, events=_synth_events(0.002))
    assert out["campaign_id"] == "v5_close_dislocation_continuation"
    assert out["science_id"] == SCIENCE_ID
    assert out["trial_count"] == 5
    assert out["tested_this_stage"] == 5
    assert "D1" not in out
    assert "D2" not in out
    assert out["sealed_oos"] == "inaccessible"
    written = (
        root / "data" / "derived" / "discovery" / "v5" / "v5-close-continuation" / "final.json"
    )
    assert written.exists()
    reversal_final = (
        root / "data" / "derived" / "discovery" / "v5" / SOURCE_RUN_ID / "final.json"
    )
    assert not reversal_final.exists()


def test_readiness_does_not_load_d3(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("readiness must not load D3")

    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation.load_d3_event_partitions", boom
    )
    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation.load_pre_d3_event_partitions", boom
    )
    monkeypatch.setattr(
        "quant_edge_lab.discovery.v5.continuation.load_continuation_source_events", boom
    )
    state = readiness_v5_continuation(Path("."))
    assert state["mode"] == "READINESS"
    assert state["execution_status"] == "NOT_APPROVED"
    assert state["evaluable_split"] == "D3"
    assert state["h1_role"] == "primary"
    assert state["sealed_oos"] == "inaccessible"
    assert state["trial_count"] == 5
    assert state["z_threshold_abs"] == 2.0
    assert state["scale_lookback_sessions"] == 20
    assert state["scale_requires_all_sessions"] is True
    assert state["primary_estimand"] == "equal_weight_trading_day_mean"
    assert state["bh_q"] == 0.10
    assert state["calendar_identity_match"] is True
    assert state["calendar_hash_actual"] == SOURCE_IDENTITY["calendar"]
    assert CAMPAIGN_ID in state["campaigns"]


def test_original_reversal_campaign_yaml_unmodified():
    assert sha256_file(Path(".") / REVERSAL_MANIFEST) == SOURCE_IDENTITY["manifest"]
    assert sha256_file(Path(".") / REVERSAL_GATES) == SOURCE_IDENTITY["gates"]


def test_not_approved_refuses_real_d3_and_pre_d3_loads():
    from quant_edge_lab.discovery.v5.continuation import (
        load_d3_event_partitions,
        load_pre_d3_event_partitions,
    )

    with pytest.raises(RuntimeError, match="refused"):
        load_d3_event_partitions(Path("."))
    with pytest.raises(RuntimeError, match="refused"):
        load_pre_d3_event_partitions(Path("."))


def test_normalized_rules_do_not_change_absolute_h1_h2_h3():
    ev = _synth_events(0.002).with_columns(pl.lit(0.1).alias("normalized_dislocation"))
    rules = frozen_rules()
    abs_rules = [r for r in rules if r.hypothesis_id.startswith("H") and r.z_abs_min is None]
    a = [apply_rule(ev, r).height for r in abs_rules]
    b = [
        apply_rule(ev.with_columns(pl.lit(9.0).alias("normalized_dislocation")), r).height
        for r in abs_rules
    ]
    assert a == b
    assert len(abs_rules) == 3


def test_continuation_science_id_and_status():
    man, gates = load_continuation(Path("."))
    ident = freeze_continuation(Path("."))
    assert man["science_id"] == SCIENCE_ID
    assert ident["science"] == SCIENCE_ID
    assert man["status"] == "SIGNAL_ONLY"
    assert man["campaign_id"] == "v5_close_dislocation_continuation"
    assert (Path(".") / MANIFEST_REL).exists()
    assert (Path(".") / GATES_REL).exists()


def _ok_stats(mean: float) -> dict:
    return {
        "ticker_days": 10,
        "trading_days": 10,
        "tickers": 5,
        "top_ticker_share": 0.2,
        "mean": mean,
        "win_rate": 0.6,
    }


def _loose_gates() -> dict:
    return yaml.safe_load((Path(".") / GATES_REL).read_text(encoding="utf-8")) | {
        "sample": {
            "min_ticker_days": 1,
            "min_trading_days": 1,
            "min_tickers": 1,
            "max_top_ticker_share": 1.0,
            "min_win_rate": 0.0,
        }
    }


def _synth_events(mean_forward: float, n: int = 40) -> pl.DataFrame:
    disc = 0.01
    raw = pl.DataFrame(
        {
            "instrument_id": [f"i{i % 5}" for i in range(n)],
            "trading_date": [f"2024-11-{(i % 20) + 1:02d}" for i in range(n)],
            "discrepancy": [disc] * n,
            "close_volume_rvol": [2.0] * n,
            "next_open_to_15m": [mean_forward] * n,
            "next_open_to_5m": [mean_forward] * n,
            "next_open_to_30m": [mean_forward] * n,
            "close_to_next_open": [0.0] * n,
        }
    )
    return attach_continuation(raw)


def _approved_tree(tmp: Path) -> Path:
    dest = tmp / "knowledge" / "campaigns"
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
    return tmp
