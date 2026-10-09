from quant_edge_lab.data.massive.options_occ import parse_occ_ticker


def test_parse_occ_call():
    rec = parse_occ_ticker("O:AAPL241018C00225000")
    assert rec is not None
    assert rec["underlying_root"] == "AAPL"
    assert rec["expiration"] == "2024-10-18"
    assert rec["call_put"] == "call"
    assert rec["strike"] == 225.0


def test_parse_occ_put():
    rec = parse_occ_ticker("O:SPY260116P00500000")
    assert rec is not None
    assert rec["underlying_root"] == "SPY"
    assert rec["call_put"] == "put"
    assert rec["strike"] == 500.0


def test_parse_occ_rejects_garbage():
    assert parse_occ_ticker("AAPL") is None
    assert parse_occ_ticker("O:AAPL241018X00225000") is None
