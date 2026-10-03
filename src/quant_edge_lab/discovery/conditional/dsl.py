"""Declarative YAML for manually registered and machine-discovered algorithms."""

from __future__ import annotations

from typing import Any

import yaml

from quant_edge_lab.discovery.conditional import AlgorithmCandidate, Condition, make_candidate


def candidate_to_yaml_dict(c: AlgorithmCandidate, *, campaign: str = "V4") -> dict[str, Any]:
    conds = []
    for x in c.conditions:
        item: dict[str, Any] = {"feature": x.feature, "op": _op_name(x.operator), "value": x.threshold}
        if not isinstance(x.threshold, bool):
            item["threshold_source"] = c.discovery_sample
            item["frozen_value"] = x.threshold
        conds.append(item)
    return {
        "candidate_id": c.candidate_id,
        "target": c.target,
        "direction": c.direction,
        "horizon_minutes": c.horizon_minutes,
        "conditions": conds,
        "lineage": {
            "campaign": campaign,
            "selected_on": c.discovery_sample,
            "source_model": c.source_model,
            "parent_id": c.parent_id,
            "events": list(c.lineage),
        },
        "status": "SIGNAL_ONLY",
    }


def candidate_to_yaml(c: AlgorithmCandidate, **kwargs: Any) -> str:
    return yaml.safe_dump(candidate_to_yaml_dict(c, **kwargs), sort_keys=False)


def candidate_from_yaml_dict(d: dict[str, Any]) -> AlgorithmCandidate:
    conds = []
    for row in d.get("conditions") or []:
        op = _op_symbol(str(row.get("op") or row.get("operator")))
        val = row.get("frozen_value", row.get("value", row.get("threshold")))
        conds.append(Condition(str(row["feature"]), op, val))  # type: ignore[arg-type]
    return make_candidate(
        conds,
        d["direction"],
        d["target"],
        int(d.get("horizon_minutes") or 15),
        str((d.get("lineage") or {}).get("source_model") or "yaml"),
        str((d.get("lineage") or {}).get("selected_on") or "manual"),
        parent_id=(d.get("lineage") or {}).get("parent_id"),
    )


def _op_name(op: str) -> str:
    return {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "==": "eq"}.get(op, op)


def _op_symbol(name: str) -> str:
    return {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "==", ">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "=="}.get(name, name)
