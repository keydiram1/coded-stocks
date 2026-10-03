"""Read-only frozen Batch-10 directions. Never writes the frozen YAML.

The frozen file uses YAML aliases (`*horizons_default`) that are not always
anchored; we parse direction fields from text so we do not rewrite that file.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from quant_edge_lab.discovery.catalog import family_by_id
from quant_edge_lab.discovery.eval_batch10 import BATCH10_PRIMARY, expand_batch10_jobs

FROZEN_REL = Path("knowledge/families/batch10_frozen_v1.yaml")

_FAMILY_DIR_TO_FUNNEL = {
    "short": "short",
    "long": "long",
    "unsigned_then_follow": "either",
    "conditional_on_expansion_bar": "long",
    "conditional_sign_of_AH": "long",
}

FunnelDir = Literal["long", "short", "either"]


def frozen_yaml_path(root: Path | None = None) -> Path:
    if root is not None:
        p = Path(root) / FROZEN_REL
        if p.exists():
            return p
    return Path(__file__).resolve().parents[3] / FROZEN_REL


@lru_cache(maxsize=8)
def load_frozen_families(path: str | None = None) -> dict:
    p = Path(path) if path else frozen_yaml_path()
    return _parse_frozen_directions(p.read_text(encoding="utf-8"))


def _parse_frozen_directions(text: str) -> dict:
    families: dict[str, dict] = {}
    current: str | None = None
    in_families = False
    for line in text.splitlines():
        if line.startswith("families:"):
            in_families = True
            continue
        if in_families and line and not line.startswith(" ") and not line.startswith("#"):
            break
        if not in_families:
            continue
        m = re.match(r"^  (H\d{3}):\s*$", line)
        if m:
            current = m.group(1)
            families[current] = {"direction": None, "neighborhood_variants": []}
            continue
        if current is None:
            continue
        dm = re.match(r"^    direction:\s+(\S+)", line)
        if dm and families[current]["direction"] is None:
            families[current]["direction"] = dm.group(1).strip()
            continue
        vid = re.search(r"id:\s*(H\d{3}\.v\d+)", line)
        if vid:
            rec: dict = {"id": vid.group(1)}
            d2 = re.search(r"direction:\s*(long|short|either)\b", line)
            if d2:
                rec["direction"] = d2.group(1)
            families[current]["neighborhood_variants"].append(rec)
    return families


def expected_direction_for_job(family_id: str, variant_id: str = "default", *, root: Path | None = None) -> FunnelDir:
    """Variant's own frozen direction; parent family direction only if the variant omits one."""
    path = frozen_yaml_path(root)
    fams = load_frozen_families(str(path))
    spec = fams.get(family_id) or {}
    family_raw = spec.get("direction")
    mapped = _FAMILY_DIR_TO_FUNNEL.get(str(family_raw), None)
    family_dir: FunnelDir = mapped if mapped in {"long", "short", "either"} else family_by_id(family_id).expected_direction  # type: ignore[assignment]
    if variant_id in {"default", family_id, ""}:
        return family_dir
    rows = spec.get("neighborhood_variants") or []
    for row in rows:
        if str(row.get("id")) != variant_id:
            continue
        if "direction" in row:
            raw = str(row["direction"])
            out = _FAMILY_DIR_TO_FUNNEL.get(raw, raw)
            if out not in {"long", "short", "either"}:
                raise ValueError(f"{variant_id} frozen direction {raw!r} is not a funnel direction")
            return out  # type: ignore[return-value]
        return family_dir
    raise KeyError(f"{variant_id} not in frozen neighborhood_variants of {family_id}")


def all_batch10_job_directions(root: Path | None = None) -> dict[str, FunnelDir]:
    out: dict[str, FunnelDir] = {}
    for fid, vid, _ in expand_batch10_jobs(list(BATCH10_PRIMARY), expand_variants=True):
        key = fid if vid == "default" else vid
        out[key] = expected_direction_for_job(fid, vid, root=root)
    return out
