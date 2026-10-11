import json
from datetime import datetime, time
from pathlib import Path

import polars as pl
import pytest

from quant_edge_lab.discovery.v6.calendar import (
    assert_no_d3_days,
    calendar_manifest,
    d2_state_and_eval_days,
)
from quant_edge_lab.discovery.v6.d1_pin import (
    FROZEN_CELL,
    FROZEN_D1_IDENTITY,
    FROZEN_SELECTED_ID,
    D1PinError,
    assert_d1_selection,
    assert_d1_sha256,
    assert_d2_identity,
    load_d1_pin,
)
from quant_edge_lab.discovery.v6.design import load_v6
from quant_edge_lab.discovery.v6.pipeline import walk_d2
from quant_edge_lab.discovery.v6.promotion import d2_failure_reasons, d2_h1_decision, h1_survives_d2
from quant_edge_lab.discovery.v6.runner import run_campaign_v6
from quant_edge_lab.hashing import sha256_file, sha256_text

SAMPLE = {
    "min_ticker_days": 750,
    "min_trading_days": 100,
    "min_tickers": 75,
    "max_top_ticker_share": 0.50,
    "min_win_rate": 0.48,
}

H1_OK = {
    "mean": 0.0004,
    "ticker_days": 800,
    "trading_days": 120,
    "tickers": 80,
    "top_ticker_share": 0.1,
    "win_rate": 0.55,
}
INC_OK = {"mean_day_delta": 0.0004, "n_matched_days": 120, "p_one_sided": 0.01}

W15 = dict(FROZEN_CELL)


def _d1_payload(selected_id="W15_W15", cell=None):
    return {
        "selection": {
            "decision": "FREEZE",
            "selected_id": selected_id,
            "frozen_cell": cell or dict(FROZEN_CELL),
        }
    }


def _rth_day(day: str, names: dict[str, float]) -> pl.DataFrame:
    rows = []
    y, m, d = (int(x) for x in day.split("-"))
    for iid, px0 in names.items():
        for h in range(9, 16):
            for mm in range(60):
                t = time(h, mm)
                shock = 20.0 if iid == "AAA" and time(10, 0) <= t <= time(10, 14) else 0.0
                px = px0 + shock
                rows.append(
                    {
                        "instrument_id": iid,
                        "ticker": iid,
                        "session_date": day,
                        "time_et": t,
                        "is_rth": True,
                        "open": px,
                        "close": px,
                        "high": px,
                        "low": px,
                        "volume": 1_000_000.0,
                        "ts_utc": datetime(y, m, d, h, mm),
                    }
                )
    return pl.DataFrame(rows)


def test_d2_reads_exactly_w15_w15_from_d1_artifact(tmp_path: Path):
    d1 = _d1_payload()
    p = tmp_path / "d1.json"
    p.write_text(json.dumps(d1), encoding="utf-8")
    (tmp_path / "run_identity.json").write_text(
        json.dumps({"identity": FROZEN_D1_IDENTITY}), encoding="utf-8"
    )
    pin = load_d1_pin(
        tmp_path,
        expected_sha256=sha256_file(p),
        expected_identity=FROZEN_D1_IDENTITY,
        current_identity=FROZEN_D1_IDENTITY,
    )
    assert pin["selected_id"] == "W15_W15"
    assert pin["frozen_cell"]["impulse_window_minutes"] == 15
    assert pin["frozen_cell"]["wait_window_minutes"] == 15


def test_tampered_d1_selection_refuses():
    with pytest.raises(D1PinError, match="selected_id"):
        assert_d1_selection(_d1_payload(selected_id="W5_W5", cell={**FROZEN_CELL, "id": "W5_W5"}))
    with pytest.raises(D1PinError, match="decision"):
        bad = _d1_payload()
        bad["selection"]["decision"] = "KILL_AT_D1"
        assert_d1_selection(bad)


def test_identity_mismatch_refuses():
    stored = {"identity": dict(FROZEN_D1_IDENTITY)}
    current = dict(FROZEN_D1_IDENTITY)
    assert_d2_identity(stored, current)
    wrong = {**FROZEN_D1_IDENTITY, "data_manifest": "0" * 64}
    with pytest.raises(D1PinError, match="identity"):
        assert_d2_identity({"identity": wrong}, current)
    with pytest.raises(D1PinError, match="identity"):
        assert_d2_identity(stored, wrong)


def test_d1_sha256_mismatch_refuses(tmp_path: Path):
    p = tmp_path / "d1.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(D1PinError, match="sha256"):
        assert_d1_sha256(p, expected="abc")


def test_other_three_cells_never_evaluated_in_d2():
    des = load_v6(Path("."))
    man = calendar_manifest(des)
    days = ["2021-10-01", "2021-10-29", "2023-10-18", "2024-10-11"]
    names = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    bars = {d: _rth_day(d, names) for d in days[:3]}
    seen: list[str] = []

    def load(day: str):
        if day == "2024-10-11":
            raise AssertionError("D3 must not be read")
        return bars[day]

    import quant_edge_lab.discovery.v6.pipeline as pipe

    real_eval = pipe.evaluate_hypotheses

    def wrap_eval(events, **kw):
        out = real_eval(events, **kw)
        seen.append("eval")
        return out

    pipe.evaluate_hypotheses = wrap_eval
    try:
        with pytest.raises(RuntimeError, match="D2 refuses cell"):
            walk_d2(
                days,
                load,
                cal_man=man,
                des=des,
                instruments=None,
                cell={"id": "W5_W5", "impulse_window_minutes": 5, "wait_window_minutes": 5,
                      "min_abs_shock_z": 2.0, "high_retention_min": 0.5, "low_retention_max": 0.0},
            )
        out = walk_d2(
            days,
            load,
            cal_man=man,
            des=des,
            instruments=None,
            cell=W15,
        )
    finally:
        pipe.evaluate_hypotheses = real_eval
    assert out["evaluated_cell_ids"] == [FROZEN_SELECTED_ID]
    assert list(out["cells"]) == ["W15_W15"]
    assert seen == ["eval"]
    assert "W5_W5" not in out["cells"]
    assert "W5_W15" not in out["cells"]
    assert "W15_W5" not in out["cells"]


def test_d2_date_reader_cannot_access_d3():
    des = load_v6(Path("."))
    man = calendar_manifest(des)
    days = [
        "2021-10-01",
        man["splits"]["D1"]["end"],
        man["splits"]["D2"]["start"],
        man["splits"]["D2"]["end"],
        man["splits"]["D3"]["start"],
    ]
    pre, d2 = d2_state_and_eval_days(days, man)
    assert man["splits"]["D3"]["start"] not in pre
    assert man["splits"]["D3"]["start"] not in d2
    assert d2[0] == "2023-10-18"
    assert d2[-1] <= "2024-10-10"
    with pytest.raises(RuntimeError, match="D2 execute refuses"):
        assert_no_d3_days([man["splits"]["D3"]["start"]], man)
    loaded: list[str] = []

    def load(day: str):
        loaded.append(day)
        if day >= man["splits"]["D3"]["start"]:
            raise AssertionError("D3 payload")
        return None

    walk_d2(days, load, cal_man=man, des=des, instruments=None, cell=W15)
    assert man["splits"]["D3"]["start"] not in loaded
    assert all(d <= man["splits"]["D2"]["end"] for d in loaded)


def test_d2_floor_2_99_bp_fails_3_00_bp_floor_component():
    inc = INC_OK
    fail = d2_failure_reasons(
        {**H1_OK, "mean": 0.000299}, inc, sample=SAMPLE, bh_survivor=True
    )
    assert "below_d2_floor" in fail
    ok, why = h1_survives_d2(
        {**H1_OK, "mean": 0.000299}, inc, sample=SAMPLE, bh_survivor=True
    )
    assert ok is False and why == "below_d2_floor"
    floor_pass = d2_failure_reasons(
        {**H1_OK, "mean": 0.0003}, inc, sample=SAMPLE, bh_survivor=True
    )
    assert "below_d2_floor" not in floor_pass
    assert h1_survives_d2({**H1_OK, "mean": 0.0003}, inc, sample=SAMPLE, bh_survivor=True)[0]


def test_non_bh_h1_fails_incremental_gates_and_h2_cannot_rescue():
    reasons = d2_failure_reasons(H1_OK, INC_OK, sample=SAMPLE, bh_survivor=False)
    assert "bh_not_survivor" in reasons
    pfail = d2_failure_reasons(
        H1_OK,
        {**INC_OK, "p_one_sided": 0.05},
        sample=SAMPLE,
        bh_survivor=True,
    )
    assert "incremental_p" in pfail
    nfail = d2_failure_reasons(
        H1_OK,
        {**INC_OK, "n_matched_days": 99},
        sample=SAMPLE,
        bh_survivor=True,
    )
    assert "incremental_n_matched_days" in nfail
    killed = d2_h1_decision(H1_OK, INC_OK, sample=SAMPLE, bh_survivor=False)
    assert killed["decision"] == "KILL_AT_D2"
    assert killed["d3_opened"] is False
    assert "bh_not_survivor" in killed["failure_reasons"]


def test_successful_d2_does_not_open_d3():
    out = d2_h1_decision(H1_OK, INC_OK, sample=SAMPLE, bh_survivor=True)
    assert out["decision"] == "ACTIVE_FOR_D3"
    assert out["d2_opened"] is True
    assert out["d3_opened"] is False
    assert out["next_action"] == "STOP_FOR_INDEPENDENT_REVIEW"
    assert out["failure_reasons"] == []


def test_readiness_and_execute_d2_flag_do_not_read_d2_payloads(monkeypatch):
    called: list[str] = []

    def boom_d1(*_a, **_k):
        called.append("d1")
        raise AssertionError("d1")

    def boom_d2(*_a, **_k):
        called.append("d2")
        return {"mode": "D2_ONLY", "d3_opened": False, "evaluated_cell_ids": ["W15_W15"]}

    monkeypatch.setattr("quant_edge_lab.discovery.v6.runner.run_d1_execute", boom_d1)
    monkeypatch.setattr("quant_edge_lab.discovery.v6.runner.run_d2_execute", boom_d2)
    ready = run_campaign_v6(Path("."), execute=False)
    assert ready["mode"] == "READINESS"
    assert ready["d2_payloads_read"] is False
    assert ready["d2_frozen_cell"]["id"] == "W15_W15"
    assert ready["d2_start"] == "2023-10-18"
    assert ready["d2_end"] == "2024-10-10"
    assert called == []
    out = run_campaign_v6(Path("."), execute_d2=True)
    assert out["d3_opened"] is False
    assert called == ["d2"]
    with pytest.raises(RuntimeError, match="cannot combine"):
        run_campaign_v6(Path("."), execute=True, execute_d2=True)


def test_sha256_helper_stable():
    assert sha256_text("x") == sha256_text("x")
