from pathlib import Path

import pytest

from quant_edge_lab.data.groups.massive_sic_store import (
    FORBIDDEN_OUTPUT_KEYS,
    PIT_STATUS,
    SEMANTICS,
    assert_no_forbidden_keys,
    assign_primary_or_fallback,
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


def test_normalize_sic_and_fallback_only_when_industry_too_small():
    assert normalize_sic("3674") == ("3674", "36")
    fat = [{"sic4": "3674", "sic2": "36", "ticker": f"T{i}"} for i in range(8)]
    thin = [{"sic4": "3571", "sic2": "35", "ticker": "AAPL"}]
    out = assign_primary_or_fallback(fat + thin, min_primary_members=8)
    by = {r["ticker"]: r for r in out}
    assert by["T0"]["group_id"] == "sic4:3674"
    assert by["T0"]["assignment"] == "primary"
    assert by["AAPL"]["group_id"] == "sic2:35"
    assert by["AAPL"]["assignment"] == "fallback_insufficient_primary_members"
    assert by["AAPL"]["group_level"] == "industry_group"


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
