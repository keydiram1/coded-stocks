from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from quant_edge_lab.discovery.v5.identity import SCIENCE_ID, build_identity, data_manifest_identity
from quant_edge_lab.hashing import sha256_file

MANIFEST_REL = Path("knowledge/campaigns/v5_mechanism_manifest.yaml")
GATES_REL = Path("knowledge/campaigns/v5_mechanism_gates.yaml")


def load_v5(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    man = yaml.safe_load((root / MANIFEST_REL).read_text(encoding="utf-8"))
    gates = yaml.safe_load((root / GATES_REL).read_text(encoding="utf-8"))
    return man, gates


def freeze_v5(root: Path) -> dict[str, str]:
    man_h = sha256_file(root / MANIFEST_REL)
    gates_h = sha256_file(root / GATES_REL)
    ident = build_identity(root, manifest_hash=man_h, gates_hash=gates_h)
    ident["science"] = SCIENCE_ID
    ident["data_manifest"] = data_manifest_identity(root)
    return ident


def campaign_block(man: dict[str, Any]) -> dict[str, Any]:
    name = man["primary_campaign"]
    return man[name]
