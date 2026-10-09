"""Load V6 DESIGN_ONLY YAML. Does not run research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from quant_edge_lab.discovery.v5.identity import build_identity, data_manifest_identity
from quant_edge_lab.hashing import sha256_file

DESIGN_REL = Path("knowledge/campaigns/v6_shock_retention_design.yaml")
SELECTION_REL = Path("knowledge/campaigns/v6_next_campaign_selection.yaml")
SCIENCE_ID = "v6_market_residual_shock_retention_v1"
CAMPAIGN_KEY = "MARKET_RESIDUAL_SHOCK_RETENTION_V1"


def load_v6(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / DESIGN_REL).read_text(encoding="utf-8"))


def campaign_block(des: dict[str, Any]) -> dict[str, Any]:
    return des[CAMPAIGN_KEY]


def frozen_cells(
    des: dict[str, Any] | None = None, *, root: Path | None = None
) -> list[dict[str, Any]]:
    if des is None:
        if root is None:
            raise ValueError("root or des required")
        des = load_v6(root)
    cells = list(campaign_block(des)["d1_grid"]["frozen_cells"])
    if len(cells) != 4:
        raise AssertionError(f"expected 4 frozen D1 cells, got {len(cells)}")
    return cells


def freeze_v6(root: Path) -> dict[str, str]:
    man_h = sha256_file(root / DESIGN_REL)
    sel_h = sha256_file(root / SELECTION_REL)
    ident = build_identity(root, manifest_hash=man_h, gates_hash=sel_h)
    ident["science"] = SCIENCE_ID
    ident["data_manifest"] = data_manifest_identity(root)
    ident["design"] = man_h
    ident["selection"] = sel_h
    return ident
