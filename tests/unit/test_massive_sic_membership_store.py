from collections import Counter
from pathlib import Path

import pytest

from quant_edge_lab.data.groups.massive_sic_store import (
    FORBIDDEN_OUTPUT_KEYS,
    PIT_STATUS,
    SEMANTICS,
    STORE_MUST_NOT_CHOOSE,
    assert_no_forbidden_keys,
    audit_source,
    membership_from_tickers,
    normalize_sic,
    write_asof_part,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def paginate(self, path, params=None, max_items=None, page_sleep_s=0, results_key="results"):
        rows = list(self.get(path, params).get(results_key) or [])
        if max_items is not None:
            rows = rows[:max_items]
        yield from rows

    def get(self, path: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        self.calls.append((path, params))
        if path.startswith("/v3/reference/tickers/"):
            ticker = path.rsplit("/", 1)[-1]
            sic = "3571" if ticker == "AAPL" else "3674"
            return {"results": {"ticker": ticker, "sic_code": sic}}
        if params.get("active") == "false":
            return {
                "results": [
                    {
                        "ticker": "OLD",
                        "type": "CS",
                        "sic_code": "1000",
                        "active": False,
                        "delisted_utc": "2020-01-01",
                        "cik": "1",
                        "composite_figi": "BBG1",
                        "last_updated_utc": "2024-01-01T00:00:00Z",
                    }
                ]
            }
        return {
            "results": [
                {
                    "ticker": "AAA",
                    "type": "CS",
                    "sic_code": "3674",
                    "active": True,
                    "cik": "2",
                    "composite_figi": "BBG2",
                    "last_updated_utc": "2024-01-01T00:00:00Z",
                },
                {
                    "ticker": "AAPL",
                    "type": "CS",
                    "sic_code": "3571",
                    "active": True,
                    "cik": "3",
                    "composite_figi": "BBG3",
                    "last_updated_utc": "2024-01-01T00:00:00Z",
                },
            ]
        }


def stage0_event_group(eligible: list[dict], *, min_n: int = 8) -> dict[str, str | None]:
    """Documented Stage 0 rule only. Not a runner. Uses eligible counts at the event."""
    n4 = Counter(r["sic4"] for r in eligible if r.get("sic4"))
    out: dict[str, str | None] = {}
    for row in eligible:
        s4, s2 = row.get("sic4"), row.get("sic2")
        if s4 and n4[s4] >= min_n:
            out[str(row["ticker"])] = f"sic4:{s4}"
        elif s2:
            out[str(row["ticker"])] = f"sic2:{s2}"
        else:
            out[str(row["ticker"])] = None
    return out


def test_store_keeps_sic_hierarchy_and_does_not_choose_fallback():
    assert normalize_sic("3674") == ("3674", "36", "ok")
    assert normalize_sic("12345") == (None, None, "rejected_gt_4_digits")
    assert normalize_sic("36") == ("0036", "00", "ok")
    items = (
        [{"ticker": f"A{i}", "sic_code": "3674"} for i in range(20)]
        + [{"ticker": f"B{i}", "sic_code": "3679"} for i in range(3)]
    )
    rows = membership_from_tickers(items, "2021-10-01")
    by = {r["ticker"]: r for r in rows}
    assert by["A0"]["sic4_id"] == "sic4:3674"
    assert by["A0"]["sic2_id"] == "sic2:36"
    assert by["B0"]["sic4_id"] == "sic4:3679"
    assert by["B0"]["sic2_id"] == "sic2:36"
    assert by["B0"]["sic2_level"] == "sic_major_group"
    for row in rows:
        for key in STORE_MUST_NOT_CHOOSE:
            assert key not in row
        assert row["identity_scheme"] == "ticker"
    figi_row = membership_from_tickers(
        [{"ticker": "FB", "sic_code": "7370", "composite_figi": "BBG000", "cik": "9"}],
        "2021-10-01",
    )[0]
    assert figi_row["instrument_id"] == "figi:BBG000"
    assert figi_row["ticker"] == "FB"
    assert figi_row["identity_scheme"] == "figi"


def test_stage0_fallback_uses_all_eligible_sic2_members():
    eligible = (
        [{"ticker": f"A{i}", "sic4": "3674", "sic2": "36"} for i in range(20)]
        + [{"ticker": f"B{i}", "sic4": "3679", "sic2": "36"} for i in range(3)]
    )
    chosen = stage0_event_group(eligible)
    assert chosen["A0"] == "sic4:3674"
    assert chosen["B0"] == "sic2:36"
    sic2_members = [r["ticker"] for r in eligible if r["sic2"] == "36"]
    assert len(sic2_members) == 23


def test_membership_keeps_all_asof_rows_and_rejects_returns(tmp_path: Path):
    items = [
        {"ticker": "AAA", "sic_code": "3674", "type": "CS", "primary_exchange": "XNAS"},
        {"ticker": "BBB", "sic_code": "3674", "type": "CS", "primary_exchange": "XNAS"},
    ]
    d1 = membership_from_tickers(items, "2021-10-01")
    d2 = membership_from_tickers(items, "2023-01-03")
    p1 = write_asof_part(tmp_path, "2021-10-01", d1)
    p2 = write_asof_part(tmp_path, "2023-01-03", d2)
    assert p1 != p2
    assert p1.exists() and p2.exists()
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_keys({"future_5m_return": 1})
    with pytest.raises(ValueError, match="must not choose"):
        write_asof_part(tmp_path, "2021-10-02", [{**d1[0], "group_id": "sic4:3674"}])


def test_audit_writes_manifest_without_feasibility(tmp_path: Path):
    report = audit_source(FakeClient(), root=tmp_path, dates=("2021-10-01",), list_limit=2)
    assert report["pit_status"] == PIT_STATUS
    assert report["semantics"] == SEMANTICS
    assert report["call_it_pit_gics"] is False
    assert report["call_it_true_pit_sic"] is False
    assert report["stage0_feasibility_ran"] is False
    assert report["future_returns_computed"] is False
    assert report["sic_coverage"]["overview_has_sic_on_any_date"] is True
    assert "missing_sic_pct_by_asof" in report["missing_identifiers"]
    assert "n_disappeared" in report["ticker_changes_delistings"]
    assert "n_sic_changed" in report["sic_code_changes"]
    assert "identical_refetch_of_first_asof" in report["revision_behaviour"]
    assert Path(report["audit_path"]).exists()
    blob = Path(report["audit_path"]).read_text(encoding="utf-8")
    for key in FORBIDDEN_OUTPUT_KEYS:
        assert key not in blob
    assert "sample_assignments" not in blob
