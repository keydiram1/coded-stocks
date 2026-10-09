from quant_edge_lab.data.massive.options_audit import pit_field_table


def test_pit_marks_missing_quote_fields_unknown():
    rows = {r["field"]: r for r in pit_field_table(["ticker", "volume", "window_start", "close"])}
    assert rows["bid/ask/NBBO"]["class"] == "D"
    assert rows["open_interest"]["class"] == "D"
    assert rows["implied_volatility"]["class"] == "D"
    assert rows["greeks"]["class"] == "D"
    assert rows["close"]["class"] == "A"
    assert rows["volume"]["class"] == "A"


def test_readiness_without_quotes_is_limited():
    from quant_edge_lab.data.massive.options_audit import _readiness

    inv = {
        "n_minute_days": 100,
        "products_present": {"quotes_v1": False, "minute_aggs_v1": True},
    }
    assert _readiness(inv, {"ok": True}) == "READY_WITH_LIMITATIONS"
    assert _readiness({"n_minute_days": 0}, {"ok": False}) == "NOT_READY"
