from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from email.message import EmailMessage
from io import BytesIO
from urllib.error import HTTPError

import polars as pl
import pytest

from quant_edge_lab.data.catalog import load_bars, load_instruments
from quant_edge_lab.data.massive.client import MassiveClient, MassiveHTTPError
from quant_edge_lab.data.massive.ingest import _checkpoint_path, ingest_tickers
from quant_edge_lab.data.massive.normalize import normalize_aggs, unix_ms_to_utc_naive
from quant_edge_lab.hypotheses.loader import load_hypothesis
from quant_edge_lab.pipeline import run_hypothesis


def test_unix_ms_utc():
    ts = unix_ms_to_utc_naive(1_717_502_400_000)
    assert ts.tzinfo is None
    assert ts.year == 2024


def test_normalize_tickers_mixed_list_dates():
    from quant_edge_lab.data.massive.normalize import normalize_tickers

    df = normalize_tickers(
        [
            {
                "ticker": "AAA",
                "type": "CS",
                "primary_exchange": "XNAS",
                "list_date": None,
                "active": True,
            },
            {
                "ticker": "BBB",
                "type": "CS",
                "primary_exchange": "XNYS",
                "list_date": "2019-10-07T04:00:00Z",
                "delisted_utc": None,
                "active": True,
            },
        ]
    )
    assert df.height == 2
    assert "2019-10-07" in str(df.filter(pl.col("ticker") == "BBB")["list_date"][0])


def test_normalize_aggs_dedupes_and_schema():
    raw = [
        {"o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "vw": 10.2, "n": 3, "t": 1_717_502_400_000},
        {"o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "vw": 10.2, "n": 3, "t": 1_717_502_400_000},
    ]
    df = normalize_aggs("AAPL", raw, instrument_id="figi:X")
    assert df.height == 1
    for col in ("instrument_id", "ticker", "ts_utc", "open", "high", "low", "close", "volume", "source"):
        assert col in df.columns
    assert df["source"][0] == "massive"
    assert df["adjusted"][0] is False


def test_pagination_follows_next_url_without_exposing_key():
    pages = {
        "/v2/aggs/ticker/AAPL/range/1/minute/2024-06-03/2024-06-03": {
            "results": [{"o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "t": 1_717_502_400_000}],
            "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/minute/2024-06-03/2024-06-03?cursor=abc",
        },
        "/v2/aggs/ticker/AAPL/range/1/minute/2024-06-03/2024-06-03": None,  # overwritten below
    }

    calls: list[str] = []

    def opener(req, timeout=0):
        path = req.full_url.split("?")[0].replace("https://api.massive.com", "")
        calls.append(path)
        if "cursor=abc" in req.full_url:
            return json.dumps({"results": [{"o": 2, "h": 2, "l": 2, "c": 2, "v": 2, "t": 1_717_502_460_000}]})
        return json.dumps(
            {
                "results": [{"o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "t": 1_717_502_400_000}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/minute/2024-06-03/2024-06-03?cursor=abc",
            }
        )

    client = MassiveClient(api_key="SECRETKEY", base_url="https://api.massive.com", opener=opener)
    rows = list(
        client.paginate(
            "/v2/aggs/ticker/AAPL/range/1/minute/2024-06-03/2024-06-03",
            {"adjusted": "false", "limit": 50},
        )
    )
    assert len(rows) == 2
    assert all("SECRETKEY" not in c for c in calls)


def test_retry_on_429(monkeypatch):
    hits = {"n": 0}

    def opener(req, timeout=0):
        hits["n"] += 1
        if hits["n"] == 1:
            hdrs = EmailMessage()
            hdrs["Retry-After"] = "0"
            raise HTTPError(req.full_url, 429, "rate", hdrs=hdrs, fp=BytesIO())
        return json.dumps({"results": []})

    client = MassiveClient(api_key="x", opener=opener, max_retries=3)
    monkeypatch.setattr("quant_edge_lab.data.massive.client.time.sleep", lambda s: None)
    assert client.get("/v3/reference/tickers", {"limit": 1})["results"] == []
    assert hits["n"] == 2


def test_resume_skips_completed(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_edge_lab.data.massive.ingest.time.sleep", lambda s: None)

    class Fake:
        def get(self, path, params=None):
            if "reference/tickers" in path:
                return {"results": [{"ticker": "AAA", "primary_exchange": "XNAS", "type": "CS"}]}
            return {"results": [{"o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "t": 1_717_502_400_000}]}

        def paginate(self, path, params=None, results_key="results"):
            yield {"o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "t": 1_717_502_400_000}

    ingest_tickers(Fake(), tmp_path, "massive-smoke", ["AAA"], date(2024, 6, 3), date(2024, 6, 3), sleep_s=0)
    first = json.loads(_checkpoint_path(tmp_path, "massive-smoke").read_text())
    ingest_tickers(Fake(), tmp_path, "massive-smoke", ["AAA"], date(2024, 6, 3), date(2024, 6, 3), sleep_s=0)
    bars = load_bars(tmp_path, "massive-smoke")
    assert bars.height == 1
    assert first["completed"]["AAA|2024-06-03|2024-06-03"]["ok"] is True


def test_same_hypothesis_sample_and_massive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    src = Path(__file__).resolve().parents[2]
    shutil.copytree(src / "hypotheses", tmp_path / "hypotheses")
    shutil.copytree(src / "config", tmp_path / "config")
    from quant_edge_lab.data.sample import write_sample_data

    write_sample_data(tmp_path, seed=42)
    rec_s = run_hypothesis(tmp_path / "hypotheses" / "gap_rvol_continuation_v1.yaml", root=tmp_path, dataset="sample")
    # Tiny real-shaped Massive parquet (fixture, not live API).
    ts = datetime(2024, 6, 3, 13, 30)
    bars = pl.DataFrame(
        {
            "instrument_id": ["figi:X"],
            "ticker": ["AAA"],
            "ts_utc": [ts],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "volume": [1000.0],
            "vwap": [10.0],
            "transactions": [10],
            "source": ["massive"],
        }
    ).with_columns(pl.col("ts_utc").cast(pl.Datetime("us")))
    inst = pl.DataFrame(
        {
            "instrument_id": ["figi:X"],
            "ticker": ["AAA"],
            "exchange": ["NASDAQ"],
            "security_type": ["COMMON_STOCK"],
        }
    )
    bdir = tmp_path / "data" / "normalized" / "massive-smoke" / "bars_1m" / "date=2024-06-03"
    bdir.mkdir(parents=True)
    bars.write_parquet(bdir / "part.parquet")
    inst.write_parquet(tmp_path / "data" / "normalized" / "massive-smoke" / "instruments.parquet")
    rec_m = run_hypothesis(
        tmp_path / "hypotheses" / "gap_rvol_continuation_v1.yaml",
        root=tmp_path,
        dataset="massive-smoke",
    )
    assert rec_s.dataset == "sample"
    assert rec_m.dataset == "massive-smoke"
    assert rec_s.hypothesis_hash == rec_m.hypothesis_hash
    spec = load_hypothesis(tmp_path / "hypotheses" / "gap_rvol_continuation_v1.yaml")
    assert spec.id == rec_s.hypothesis_id == rec_m.hypothesis_id
