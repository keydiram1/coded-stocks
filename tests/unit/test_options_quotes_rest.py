from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

from quant_edge_lab.data.massive.options_quotes_rest import probe_quotes_rest_entitlement


def test_entitlement_401_stops_without_flatfiles(tmp_path: Path, monkeypatch):
    def opener(req, timeout=0):
        raise HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            hdrs=EmailMessage(),
            fp=BytesIO(b'{"status":"ERROR","error":"Unknown API Key"}'),
        )

    monkeypatch.setattr(
        "quant_edge_lab.data.massive.options_quotes_rest.urllib.request.urlopen",
        opener,
    )
    monkeypatch.setattr(
        "quant_edge_lab.data.massive.options_quotes_rest.massive_api_key",
        lambda _root=None: "dummy",
    )
    rec = probe_quotes_rest_entitlement(tmp_path)
    assert rec["entitled"] is False
    assert rec["http"] == 401
    assert rec["error"] == "Unknown API Key"
    assert rec["stop"] is True
    assert rec["used_flatfiles"] is False
    assert rec["readiness"] == "NOT_READY"
    assert "quotes_v1_flatfile_get" in rec["did_not"]
