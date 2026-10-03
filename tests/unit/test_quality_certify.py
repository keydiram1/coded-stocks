from quant_edge_lab.data.massive.quality import _cat, _weekday_gaps, decide_verdict


def test_verdict_fail_if_any_fail():
    assert decide_verdict({"a": _cat("PASS", "ok"), "b": _cat("WARNING", "w")}) == (
        "READY_FOR_EXPLORATORY_RESEARCH"
    )
    assert decide_verdict({"a": _cat("FAIL", "bad")}) == "NOT_READY"


def test_weekday_gaps_skips_weekends():
    gaps = _weekday_gaps("2021-10-01", "2021-10-11", {"2021-10-01", "2021-10-04"})
    assert "2021-10-02" not in gaps  # Saturday
    assert "2021-10-05" in gaps
