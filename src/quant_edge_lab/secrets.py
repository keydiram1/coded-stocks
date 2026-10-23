from __future__ import annotations

import os
from pathlib import Path


def _apply_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        existing = os.environ.get(key)
        empty_existing = existing is None or str(existing).strip() == ""
        if key and empty_existing:
            os.environ[key] = value


def load_dotenv(root: Path | None = None) -> None:
    """Load KEY=VALUE from .env. Does not override non-empty os.environ values.
    Empty or whitespace env values do not block dotenv (they used to, which
    produced REST 401 Unknown API Key with a blank apiKey query param).
    """
    _apply_dotenv_file((root or Path.cwd()) / ".env")
    extra = os.environ.get("QUANT_EDGE_DOTENV", "").strip()
    if extra:
        _apply_dotenv_file(Path(extra))
    _apply_dotenv_file(Path.home() / "quant-edge-lab" / ".env")


def massive_api_key(root: Path | None = None) -> str:
    load_dotenv(root)
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MASSIVE_API_KEY is not set. Put it in the project-root .env or the environment. "
            "Do not pass the key on the command line."
        )
    return key


def massive_key_fingerprint(root: Path | None = None) -> dict[str, object]:
    """Presence/length/source of the REST key. Never returns the secret."""
    import hashlib

    env_before = os.environ.get("MASSIVE_API_KEY")
    env_before_empty = env_before is None or str(env_before).strip() == ""
    project = (root or Path.cwd()) / ".env"
    extra = os.environ.get("QUANT_EDGE_DOTENV", "").strip()
    home = Path.home() / "quant-edge-lab" / ".env"
    source = "os.environ"
    if env_before_empty:
        if extra and Path(extra).is_file():
            source = "QUANT_EDGE_DOTENV"
        elif home.is_file():
            source = "home_quant_edge_lab_dotenv"
        elif project.is_file():
            source = "project_dotenv"
        else:
            source = "missing"
    key = massive_api_key(root)
    return {
        "loaded": True,
        "source": source,
        "len": len(key),
        "sha256_8": hashlib.sha256(key.encode("utf-8")).hexdigest()[:8],
        "empty_os_environ_blocked_dotenv": False,
        "project_dotenv_exists": project.is_file(),
        "home_dotenv_exists": home.is_file(),
    }
