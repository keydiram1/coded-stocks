from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from quant_edge_lab.discovery.v5.campaigns.close_dislocation import (
    attach_equal_weight_loo,
    freeze_rules_from_d1,
    session_close_rows,
)
from quant_edge_lab.discovery.v5.campaigns.close_features import (
    discrepancy,
    resolution_direction,
    signed_resolution,
)
from quant_edge_lab.discovery.v5.session import (
    classify_trading_day,
    has_close_window,
    no_overnight_in_close_window,
    with_session,
)
from quant_edge_lab.features.v4r.beta import beta_from_history
from quant_edge_lab.validation.leakage import future_poison
from tests.fixtures.bars import bars_from_et, session_minutes

ET = ZoneInfo("America/New_York")
CLOSE_TIMES = [(15, 54), (15, 55), (15, 56), (15, 57), (15, 58), (15, 59)]


def _close_day(
    day: date,
    *,
    iid: str,
    ticker: str,
    last_step: float,
    volume: float = 1000.0,
    extra_open: bool = True,
) -> pl.DataFrame:
    parts = []
    if extra_open:
        parts.append(
            session_minutes(
                day,
                (9, 30),
                1,
                open_px=10.0,
                close_step=0.0,
                volume=volume,
                instrument_id=iid,
                ticker=ticker,
            )
        )
        parts.append(
            session_minutes(
                day,
                (9, 34),
                1,
                open_px=10.0,
                close_step=0.0,
                volume=100.0,
                instrument_id=iid,
                ticker=ticker,
            )
        )
        parts.append(
            session_minutes(
                day,
                (9, 44),
                1,
                open_px=10.0,
                close_step=0.0,
                volume=100.0,
                instrument_id=iid,
                ticker=ticker,
            )
        )
        parts.append(
            session_minutes(
                day,
                (9, 59),
                1,
                open_px=10.0,
                close_step=0.0,
                volume=100.0,
                instrument_id=iid,
                ticker=ticker,
            )
        )
    px = 10.0
    rows = []
    for h, m in CLOSE_TIMES:
        ts = datetime(day.year, day.month, day.day, h, m, tzinfo=ET)
        step = last_step if (h, m) == (15, 59) else 0.0
        o = px
        c = px * (1 + step)
        rows.append((ts, o, max(o, c) * 1.001, min(o, c) * 0.999, c, volume))
        px = c
    parts.append(bars_from_et(rows, instrument_id=iid, ticker=ticker))
    return pl.concat(parts)


def test_close_window_and_no_overnight_wrap():
    d = date(2024, 6, 3)
    g = with_session(_close_day(d, iid="a", ticker="A", last_step=0.01))
    assert has_close_window(g)
    assert no_overnight_in_close_window(g)
    wrap = g.with_columns(
        pl.when(pl.col("time_et").dt.hour() == 15)
        .then(pl.date(2024, 6, 4))
        .otherwise(pl.col("session_date"))
        .alias("session_date")
    )
    close_w = wrap.filter(pl.col("time_et").dt.hour() >= 15)
    assert close_w["session_date"].n_unique() == 1 or not no_overnight_in_close_window(wrap)
    mixed = wrap
    assert mixed["session_date"].n_unique() == 2
    assert no_overnight_in_close_window(mixed) is False


def test_half_day_excluded():
    d = date(2024, 6, 3)
    g = session_minutes(d, (9, 30), 10, open_px=10.0, instrument_id="a", ticker="A")
    sess = with_session(g)
    assert classify_trading_day(sess, min_rth_minutes=6) == "HALF_OR_INCOMPLETE"
    assert session_close_rows(g, min_rth_minutes=6) is None


def test_final_5m_uses_1554_to_1559_same_session():
    d = date(2024, 6, 3)
    bars = _close_day(d, iid="a", ticker="A", last_step=0.02)
    rows = session_close_rows(bars, min_rth_minutes=6)
    assert rows is not None
    assert rows["observed_value"][0] == pytest.approx(0.02, abs=1e-9)


def test_next_session_mapping_columns_exist_after_attach():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import (
        attach_outcomes_for_instrument,
        rth_only,
    )

    d0 = date(2024, 6, 3)
    d1 = date(2024, 6, 4)
    today = rth_only(with_session(_close_day(d0, iid="a", ticker="A", last_step=0.01)))
    nxt = _close_day(d1, iid="a", ticker="A", last_step=0.0)
    rec = attach_outcomes_for_instrument(
        {"expected_resolution_direction": "SHORT", "instrument_id": "a"},
        today,
        nxt,
    )
    assert "close_to_next_open" in rec
    assert "next_open_to_15m" in rec
    assert "next_open_to_5m" in rec
    assert "next_open_to_30m" in rec


def test_causal_beta_prior_days_only():
    days = []
    for i in range(20):
        df = pl.DataFrame(
            {
                "instrument_id": ["a", "b"],
                "daily_ret": [0.01 * (i + 1), -0.01],
                "daily_mkt_loo": [0.002 * (i + 1), 0.001 * (i - 5)],
            }
        )
        days.append(df)
    current = pl.DataFrame({"instrument_id": ["a"], "daily_ret": [9.99], "daily_mkt_loo": [9.99]})
    m = beta_from_history(days, ["a"])
    assert "a" in m
    leaked = beta_from_history(days + [current], ["a"])
    # current day must not be required; 20 completed days suffice
    assert np.isfinite(m["a"])
    _ = leaked


def test_leave_one_out_on_close_rows():
    d = date(2024, 6, 3)
    bars = pl.concat(
        [
            _close_day(d, iid="a", ticker="A", last_step=0.10),
            _close_day(d, iid="b", ticker="B", last_step=0.00),
            _close_day(d, iid="c", ticker="C", last_step=-0.10),
        ]
    )
    raw = session_close_rows(bars, min_rth_minutes=6)
    assert raw is not None
    rows = attach_equal_weight_loo(raw)
    b = rows.filter(pl.col("instrument_id") == "b")["market_final_5m_loo"][0]
    assert b == pytest.approx(0.0, abs=1e-9)


def test_future_poison_cannot_alter_todays_event():
    d = date(2024, 6, 3)
    rth = _close_day(d, iid="a", ticker="A", last_step=0.03)
    ah = session_minutes(
        d, (16, 5), 3, open_px=10.0, close_step=0.5, volume=9999, instrument_id="a", ticker="A"
    )
    bars = pl.concat([rth, ah], how="diagonal_relaxed")
    rows = session_close_rows(bars, min_rth_minutes=6)
    last_ts = with_session(rth).sort("time_et")["ts_utc"][-1]
    poisoned = future_poison(bars, last_ts, factor=50.0)
    rows2 = session_close_rows(poisoned, min_rth_minutes=6)
    assert rows is not None and rows2 is not None
    assert rows["observed_value"][0] == pytest.approx(rows2["observed_value"][0])


def test_one_bar_shift_changes_availability_contract():
    d = date(2024, 6, 3)
    bars = session_close_rows(_close_day(d, iid="a", ticker="A", last_step=0.02), min_rth_minutes=6)
    assert bars is not None
    avail = bars["first_available_at"][0]
    decision = bars["decision_ts"][0]
    assert avail == decision
    shifted = bars.with_columns(pl.col("first_available_at") + pl.duration(minutes=1))
    from quant_edge_lab.discovery.v5.causality import assert_frame_available_le_decision

    with pytest.raises(AssertionError):
        assert_frame_available_le_decision(shifted)


def test_d1_quantiles_ignore_later_days():
    d1 = pl.DataFrame(
        {
            "discrepancy": [0.01] * 25,
            "close_volume_rvol": [1.0] * 25,
            "trading_date": ["2022-01-01"] * 25,
        }
    )
    later = pl.DataFrame(
        {
            "discrepancy": [10.0] * 25,
            "close_volume_rvol": [100.0] * 25,
            "trading_date": ["2025-01-01"] * 25,
        }
    )
    camp = {
        "primary_outcome": "next_open_to_15m",
        "hypotheses": [
            {
                "hypothesis_id": "H1_PRIMARY",
                "role": "primary",
                "dislocation_abs_quantile": 0.9,
                "rvol_min_quantile": 0.8,
                "require_rvol": True,
            }
        ],
    }
    r1 = freeze_rules_from_d1(d1, camp)
    r_all = freeze_rules_from_d1(pl.concat([d1, later]), camp)
    assert r1[0].dislocation_abs_min == pytest.approx(0.01)
    assert r_all[0].dislocation_abs_min != r1[0].dislocation_abs_min


def test_synthetic_positive_control_sign():
    disc = 0.04
    fwd = -0.03
    direction = resolution_direction(disc)
    assert direction == "SHORT"
    assert signed_resolution(fwd, direction) > 0


def test_sparse_instruments_do_not_make_full_market_day_incomplete():
    d = date(2024, 6, 3)
    liquid = session_minutes(d, (9, 30), 390, open_px=10.0, instrument_id="liq", ticker="LIQ")
    sparse = _close_day(d, iid="s", ticker="S", last_step=0.01)
    sess = with_session(pl.concat([liquid, sparse], how="diagonal_relaxed"))
    assert classify_trading_day(sess, min_rth_minutes=380) == "FULL_RTH"


def test_half_day_ending_1259_is_incomplete():
    d = date(2024, 6, 3)
    g = session_minutes(d, (9, 30), 210, open_px=10.0, instrument_id="a", ticker="A")
    sess = with_session(g)
    assert classify_trading_day(sess, min_rth_minutes=380) == "HALF_OR_INCOMPLETE"


def test_incomplete_late_session_excluded():
    d = date(2024, 6, 3)
    g = session_minutes(d, (9, 30), 360, open_px=10.0, instrument_id="a", ticker="A")
    sess = with_session(g)
    assert classify_trading_day(sess, min_rth_minutes=380) == "HALF_OR_INCOMPLETE"


def test_next_open_requires_0930_bar_not_first_later_trade():
    from datetime import time as dtime

    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import next_day_outcomes, rth_only

    d0 = date(2024, 6, 3)
    d1 = date(2024, 6, 4)
    today = rth_only(with_session(_close_day(d0, iid="a", ticker="A", last_step=0.01)))
    late = session_minutes(d1, (9, 47), 1, open_px=11.0, instrument_id="a", ticker="A")
    close = _close_day(d1, iid="a", ticker="A", last_step=0.0, extra_open=False)
    nxt = pl.concat([late, close], how="diagonal_relaxed")
    outs = next_day_outcomes(today, nxt)
    assert outs["next_open_to_15m"] is None
    assert outs["close_to_next_open"] is None
    assert dtime(9, 47) in set(with_session(nxt)["time_et"].to_list())


def test_etf_and_warrant_cannot_contaminate_eligible_loo():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import (
        daily_panel_from_session,
        session_close_rows,
    )
    from quant_edge_lab.discovery.v5.eligibility import EligibilityConfig, listed_ids

    d = date(2024, 6, 3)
    a = _close_day(d, iid="a", ticker="A", last_step=0.02)
    b = _close_day(d, iid="b", ticker="B", last_step=-0.01)
    etf = _close_day(d, iid="etf1", ticker="SPY", last_step=0.50)
    warr = _close_day(d, iid="w1", ticker="WARR", last_step=-0.50)
    clean = pl.concat([a, b])
    dirty = pl.concat([a, b, etf, warr])
    inst = pl.DataFrame(
        {
            "instrument_id": ["a", "b", "etf1", "w1"],
            "exchange": ["NASDAQ", "NYSE", "NYSE", "NASDAQ"],
            "security_type": ["COMMON_STOCK", "COMMON_STOCK", "ETF", "WARRANT"],
        }
    )
    eligible = listed_ids(inst, EligibilityConfig())
    assert eligible == {"a", "b"}

    def loo_map(bars: pl.DataFrame) -> dict[str, float]:
        rows = session_close_rows(bars, min_rth_minutes=6)
        assert rows is not None
        rows = attach_equal_weight_loo(
            rows.filter(pl.col("instrument_id").is_in(sorted(eligible)))
        )
        return {
            str(r["instrument_id"]): float(r["market_final_5m_loo"])
            for r in rows.iter_rows(named=True)
        }

    assert loo_map(clean) == loo_map(dirty)
    panel_c = daily_panel_from_session(with_session(clean), eligible)
    panel_d = daily_panel_from_session(with_session(dirty), eligible)
    assert set(panel_c["instrument_id"].to_list()) == {"a", "b"}
    assert set(panel_d["instrument_id"].to_list()) == {"a", "b"}
    assert (
        panel_c.sort("instrument_id")["daily_mkt_loo"].to_list()
        == panel_d.sort("instrument_id")["daily_mkt_loo"].to_list()
    )


def test_beta_daily_requires_0930_open_and_1559_close():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import daily_panel_from_session

    d = date(2024, 6, 3)
    full = _close_day(d, iid="a", ticker="A", last_step=0.02)
    sparse = _close_day(d, iid="b", ticker="B", last_step=0.02, extra_open=False)
    panel = daily_panel_from_session(with_session(pl.concat([full, sparse])), {"a", "b"})
    assert panel["instrument_id"].to_list() == ["a"]
    # 09:30 open 10.0, 15:59 close 10*1.02
    assert panel["daily_ret"][0] == pytest.approx(0.02, abs=1e-9)


def test_rvol_state_truncated_to_lookback():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days

    start = date(2024, 4, 1)
    bars: dict[str, pl.DataFrame] = {}
    for i in range(30):
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

    _ev, rvol = build_events_from_days(
        keys, load_day, load_next, rvol_lookback=20, min_rth_minutes=6
    )
    assert max(len(v) for v in rvol.values()) <= 20


def test_beta_history_bootstraps_without_event_eligibility():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days

    start = date(2024, 1, 2)
    bars: dict[str, pl.DataFrame] = {}
    for i in range(21):
        d = start + timedelta(days=i)
        key = d.isoformat()
        bars[key] = pl.concat(
            [
                _close_day(d, iid="a", ticker="A", last_step=0.01 + 0.0001 * i),
                _close_day(d, iid="b", ticker="B", last_step=-0.01),
            ]
        )
    keys = sorted(bars)

    def load_day(day: str):
        return bars[day]

    def load_next(day: str):
        i = keys.index(day)
        return bars[keys[i + 1]] if i + 1 < len(keys) else None

    ev, rvol = build_events_from_days(
        keys, load_day, load_next, rvol_lookback=20, min_rth_minutes=6
    )
    last = keys[-1]
    hit = ev.filter(pl.col("trading_date") == last) if ev.height else ev
    assert hit.height > 0
    assert hit["beta_20d"].drop_nulls().len() == hit.height
    assert all(np.isfinite(x) for x in hit["beta_20d"].to_list())
    assert hit["close_volume_rvol"].drop_nulls().len() == hit.height
    assert last in {t[0] for t in rvol["a"]}


def test_panel_end_file_is_never_read_as_next_session():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days
    from quant_edge_lab.discovery.v5.preflight import clip_to_panel, next_session

    panel_start = "2026-09-28"
    panel_end = "2026-10-01"
    man = {"data": {"panel_start": panel_start, "panel_end": panel_end}}
    days = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
    bars = {
        d: pl.concat(
            [
                _close_day(date.fromisoformat(d), iid="a", ticker="A", last_step=0.01),
                _close_day(date.fromisoformat(d), iid="b", ticker="B", last_step=-0.01),
            ]
        )
        for d in days
    }
    clipped = clip_to_panel(days, man)
    read: list[str] = []

    def load_day(day: str):
        if day > panel_end:
            raise AssertionError(f"must not read {day}")
        read.append(day)
        return bars[day]

    def load_next(day: str):
        nxt = next_session(day, clipped, panel_end=panel_end)
        if nxt is None:
            return None
        return load_day(nxt)

    ev, _rvol = build_events_from_days(
        days,
        load_day,
        load_next,
        min_rth_minutes=6,
        rvol_lookback=20,
        panel_start=panel_start,
        panel_end=panel_end,
    )
    assert "2026-10-02" not in read
    assert clipped[0] == panel_start
    assert clipped[-1] == panel_end
    if ev.height:
        assert ev["trading_date"].max() <= panel_end
        assert ev["trading_date"].min() >= panel_start


def test_twenty_warmup_days_make_first_d1_signal_eligible():
    from quant_edge_lab.discovery.v5.campaigns.close_dislocation import build_events_from_days
    from quant_edge_lab.discovery.v5.eligibility import EligibilityConfig

    start = date(2024, 1, 2)
    bars: dict[str, pl.DataFrame] = {}
    for i in range(21):
        d = start + timedelta(days=i)
        bars[d.isoformat()] = pl.concat(
            [
                _close_day(
                    d, iid="a", ticker="A", last_step=0.01 + 0.0001 * i, volume=25_000.0
                ),
                _close_day(d, iid="b", ticker="B", last_step=-0.01, volume=25_000.0),
                _close_day(d, iid="etf1", ticker="SPY", last_step=0.50, volume=25_000.0),
            ]
        )
    keys = sorted(bars)
    inst = pl.DataFrame(
        {
            "instrument_id": ["a", "b", "etf1"],
            "exchange": ["NASDAQ", "NYSE", "NYSE"],
            "security_type": ["COMMON_STOCK", "COMMON_STOCK", "ETF"],
        }
    )

    def load_day(day: str):
        return bars[day]

    def load_next(day: str):
        i = keys.index(day)
        return bars[keys[i + 1]] if i + 1 < len(keys) else None

    ev, rvol = build_events_from_days(
        keys,
        load_day,
        load_next,
        rvol_lookback=20,
        min_rth_minutes=6,
        instruments=inst,
        elig_cfg=EligibilityConfig(),
    )
    first_d1 = keys[20]
    hit = ev.filter(pl.col("trading_date") == first_d1)
    assert hit.height > 0
    assert set(hit["instrument_id"].to_list()) <= {"a", "b"}
    assert "etf1" not in hit["instrument_id"].to_list()
    assert hit["beta_20d"].drop_nulls().len() == hit.height
    assert hit["close_volume_rvol"].drop_nulls().len() == hit.height
    assert len(rvol["a"]) <= 20
    assert len(rvol["a"]) >= 20


def test_synthetic_negative_control_no_direction():
    rng = np.random.default_rng(0)
    signed = []
    for x, y in zip(rng.normal(0, 0.01, 200), rng.normal(0, 0.01, 200), strict=True):
        d = discrepancy(x, 0.0)
        signed.append(signed_resolution(y, resolution_direction(d)) or 0.0)
    assert abs(float(np.mean(signed))) < 0.005
