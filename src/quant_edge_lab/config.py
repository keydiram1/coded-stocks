from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from quant_edge_lab.secrets import load_dotenv


def storage_home(repo_root: Path) -> Path:
    """Bulk data directory. QUANT_EDGE_DATA_ROOT redirects Parquet off the repo disk.

    Test tmp dirs (no pyproject.toml) always stay local so pytest cannot write to D:.
    """
    load_dotenv(repo_root)
    if not (repo_root / "pyproject.toml").exists():
        return repo_root
    override = os.environ.get("QUANT_EDGE_DATA_ROOT", "").strip()
    if override:
        return Path(override)
    return repo_root


class Paths:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        home = storage_home(self.root)
        self.data = home / "data"
        self.sample = self.data / "sample"
        self.raw = self.data / "raw"
        self.normalized = self.data / "normalized"
        self.derived = self.data / "derived"
        self.features = self.derived / "features"
        self.events = self.derived / "events"
        self.outcomes = self.derived / "outcomes"
        self.experiments = self.root / "experiments"
        self.reports = self.root / "reports"
        self.manifests = home / "manifests"
        self.hypotheses = self.root / "hypotheses"
        self.config = self.root / "config"

    def ensure(self) -> None:
        for p in (
            self.sample,
            self.raw,
            self.normalized,
            self.features,
            self.events,
            self.outcomes,
            self.experiments,
            self.reports,
            self.manifests,
        ):
            p.mkdir(parents=True, exist_ok=True)


class ResearchConfig(BaseModel):
    timezone: str = "America/New_York"
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    premarket_start: str = "04:00"
    premarket_end: str = "09:30"
    sample_seed: int = 42
    bootstrap_draws: int = 1000
    bootstrap_seed: int = 42


def load_research_config(path: Path | None = None) -> ResearchConfig:
    import yaml

    cfg_path = path or Paths().config / "research.yaml"
    if not cfg_path.exists():
        return ResearchConfig()
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return ResearchConfig.model_validate(payload)


class UniverseDefaults(BaseModel):
    country: str = "US"
    exchanges: list[str] = Field(default_factory=lambda: ["NASDAQ", "NYSE", "NYSE_AMERICAN"])
    security_type: str = "COMMON_STOCK"
    price_min: float = 1.0
    price_max: float = 20.0
    mcap_min: float = 10_000_000
    mcap_max: float = 500_000_000
    min_median_20d_dollar_volume: float = 250_000
    exclude_otc: bool = True
