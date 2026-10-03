from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(root: Path | None = None) -> None:
    """Load KEY=VALUE from project .env into os.environ (does not override existing)."""
    path = (root or Path.cwd()) / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def massive_api_key(root: Path | None = None) -> str:
    load_dotenv(root)
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MASSIVE_API_KEY is not set. Put it in the project-root .env or the environment. "
            "Do not pass the key on the command line."
        )
    return key
