from __future__ import annotations

from pathlib import Path

import yaml

from quant_edge_lab.hashing import sha256_json
from quant_edge_lab.models.hypothesis import HypothesisSpec


def load_hypothesis(path: Path | str) -> HypothesisSpec:
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    return HypothesisSpec.model_validate(payload)


def hypothesis_hash(spec: HypothesisSpec) -> str:
    return sha256_json(spec.model_dump(mode="json"))
