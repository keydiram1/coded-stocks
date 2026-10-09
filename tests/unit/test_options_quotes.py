from pathlib import Path

import yaml

from quant_edge_lab.data.massive.options_occ import parse_occ_ticker, root_looks_adjusted
from quant_edge_lab.data.massive.options_quotes import quotes_object_key, size_distribution
from quant_edge_lab.data.massive.options_quotes_feasibility import (
    representation_recommendation,
)


def test_quotes_key_convention():
    assert quotes_object_key("2025-10-08") == (
        "us_options_opra/quotes_v1/2025/10/2025-10-08.csv.gz"
    )


def test_size_distribution():
    d = size_distribution([10, 20, 30, 40, 100])
    assert d["min"] == 10
    assert d["max"] == 100
    assert d["n"] == 5
    assert d["median"] == 30


def test_root_looks_adjusted_is_conservative():
    assert root_looks_adjusted("AAPL") is False
    assert root_looks_adjusted("AAPL1") is True
    rec = parse_occ_ticker("O:AAPL241018C00225000")
    assert rec is not None
    assert rec["root_contains_digit"] is False


def test_design_rejects_trade_close_primary():
    rec = yaml.safe_load(
        (Path(".") / "knowledge/campaigns/v5_options_local_rv_design.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert rec["rejected_primary_path"]["status"] == "REJECTED"
    assert rec["V5_OPTIONS_LOCAL_RV_V1"]["primary_space"] != (
        "price_mid_unavailable_use_trade_aggregate_close"
    )
    assert rec["execution_status"] == "NOT_APPROVED"
    assert representation_recommendation()["PRIMARY_for_V1"] == "B"
    assert representation_recommendation()["not_tested_on_forward_returns"] is True
