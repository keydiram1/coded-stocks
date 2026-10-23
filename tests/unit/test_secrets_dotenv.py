from pathlib import Path

from quant_edge_lab.secrets import load_dotenv, massive_api_key


def test_empty_os_environ_does_not_block_dotenv(tmp_path: Path, monkeypatch):
    env_file = tmp_path / "keys.env"
    env_file.write_text("MASSIVE_API_KEY=dotenvkey123\n", encoding="utf-8")
    monkeypatch.setenv("MASSIVE_API_KEY", "   ")
    monkeypatch.setenv("QUANT_EDGE_DOTENV", str(env_file))
    load_dotenv(tmp_path)
    assert massive_api_key(tmp_path) == "dotenvkey123"
