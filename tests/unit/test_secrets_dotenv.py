from pathlib import Path

from quant_edge_lab.secrets import load_dotenv, massive_api_key, massive_key_fingerprint


def test_empty_os_environ_does_not_block_dotenv(tmp_path: Path, monkeypatch):
    env_file = tmp_path / "keys.env"
    env_file.write_text("MASSIVE_API_KEY=dotenvkey123\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY", "   ")
    monkeypatch.setenv("QUANT_EDGE_DOTENV", str(env_file))
    load_dotenv(tmp_path)
    assert massive_api_key(tmp_path) == "dotenvkey123"


def test_fingerprint_source_follows_load_order(tmp_path: Path, monkeypatch):
    project = tmp_path / ".env"
    extra = tmp_path / "extra.env"
    project.write_text("MASSIVE_API_KEY=fromproject\n", encoding="utf-8")
    extra.write_text("MASSIVE_API_KEY=fromextra\n", encoding="utf-8")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("QUANT_EDGE_DOTENV", str(extra))
    fp = massive_key_fingerprint(tmp_path)
    assert fp["source"] == "project_dotenv"
    monkeypatch.setenv("MASSIVE_API_KEY", "   ")
    project.write_text("MASSIVE_API_KEY=\n", encoding="utf-8")
    fp2 = massive_key_fingerprint(tmp_path)
    assert fp2["source"] == "QUANT_EDGE_DOTENV"
