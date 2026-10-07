"""Graph overlay uniqueness and cardinality. Fail closed."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_edge_lab.discovery.campaign_v4 import CampaignStop
from quant_edge_lab.discovery.v4r.stage1 import assert_unique_keys, atomic_write_parquet, v4r_day_path, v4r_peer_day_path
from quant_edge_lab.peers import GRAPH_FEATURE_COLS, PeerGraph, attach_peer_features


def apply_overlay(root: Path, day: str, g: PeerGraph, *, identity: dict) -> dict:
    p = v4r_day_path(root, day)
    dest = v4r_peer_day_path(root, day)
    feat = pl.read_parquet(p)
    assert_unique_keys(feat, where=f"stage1 {day}")
    n0 = feat.height
    drop = [c for c in GRAPH_FEATURE_COLS if c in feat.columns]
    if drop:
        feat = feat.drop(drop)
    if g.history_end and g.history_end >= day:
        raise CampaignStop(f"PIT: graph history_end {g.history_end} not < day {day}")
    if g.as_of > day:
        raise CampaignStop(f"PIT: graph as_of {g.as_of} is after day {day}")
    feat = attach_peer_features(feat, g)
    if feat.height != n0:
        raise CampaignStop(f"overlay fan-out {day}: {n0} -> {feat.height}")
    keys = ["trading_date", "instrument_id", "decision_ts"]
    ov = feat.select([c for c in keys if c in feat.columns] + [c for c in GRAPH_FEATURE_COLS if c in feat.columns])
    assert_unique_keys(ov, where=f"overlay {day}")
    if ov.height != n0:
        raise CampaignStop(f"overlay cardinality {day}: {ov.height} != {n0}")
    atomic_write_parquet(ov, dest)
    from quant_edge_lab.discovery.knowledge import atomic_write_json

    rec = {"day": day, "n": n0, "as_of": g.as_of, "n_edges": len(g.edges), "identity": identity}
    atomic_write_json(dest.parent / "ckpt.json", rec)
    return rec


def join_overlay(stage1: pl.DataFrame, overlay: pl.DataFrame) -> pl.DataFrame:
    assert_unique_keys(stage1, where="join stage1")
    assert_unique_keys(overlay, where="join overlay")
    n0 = stage1.height
    keys = [c for c in ("trading_date", "instrument_id", "decision_ts") if c in stage1.columns and c in overlay.columns]
    gcols = [c for c in GRAPH_FEATURE_COLS if c in overlay.columns]
    drop = [c for c in gcols if c in stage1.columns]
    df = stage1.drop(drop) if drop else stage1
    out = df.join(overlay.select(keys + gcols), on=keys, how="left")
    if out.height != n0:
        raise CampaignStop(f"join fan-out {n0} -> {out.height}")
    return out
