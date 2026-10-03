"""Exact duplicate / variant / derived / different-mechanism classification."""

from __future__ import annotations

from quant_edge_lab.discovery.models import FamilyIdea


def classify_pair(a: FamilyIdea, b: FamilyIdea) -> str:
    if a.id == b.id:
        return "EXACT_DUPLICATE"
    if a.hypothesis_yaml and a.hypothesis_yaml == b.hypothesis_yaml and a.event_policy == b.event_policy:
        return "EXACT_DUPLICATE"
    if a.engine == b.engine and a.engine != "unimplemented" and a.expected_direction == b.expected_direction:
        if a.parameter_neighborhood and a.parameter_neighborhood != b.parameter_neighborhood:
            return "PARAMETER_VARIANT"
    if {a.expected_direction, b.expected_direction} == {"long", "short"} and _same_core(a, b):
        return "DERIVED_OPPOSITE_DIRECTION"
    if a.parent_family_id == b.id or b.parent_family_id == a.id:
        return "DERIVED"
    if a.derived_from_experiment and a.derived_from_experiment == b.derived_from_experiment:
        return "DERIVED"
    mech_a = set(a.taxonomy.mechanism)
    mech_b = set(b.taxonomy.mechanism)
    ctx_a = set(a.taxonomy.context)
    ctx_b = set(b.taxonomy.context)
    if mech_a and mech_a != mech_b:
        return "DIFFERENT_MECHANISM"
    if ctx_a and ctx_a.isdisjoint(ctx_b) and mech_a == mech_b:
        return "DIFFERENT_CONTEXT_SAME_MECHANISM"
    return "RELATED_OR_UNCLEAR"


def _same_core(a: FamilyIdea, b: FamilyIdea) -> bool:
    return a.engine == b.engine and a.hypothesis_yaml == b.hypothesis_yaml
