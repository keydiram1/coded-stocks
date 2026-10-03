"""Campaign v3 evaluators. V2 state definitions imported unchanged. No reasoning features."""

from __future__ import annotations

from datetime import time

import polars as pl
import yaml
from pathlib import Path

from quant_edge_lab.data.massive.news import attach_cross_section, attach_news_state
from quant_edge_lab.discovery.campaign_v2_families import eval_job as eval_v2_job
from quant_edge_lab.discovery.eval_batch10 import _stamp

THR = {
    "n_observe_lo": 5,
    "n_observe_hi": 20,
    "n_ret": 0.003,
    "n_rvol": 2.0,
    "resid_n5": 0.003,
    "resid_xr": 0.008,
    "resid_nr": 0.005,
    "intensity_n": 2,
}


def load_v3_jobs(root: Path) -> tuple[dict, list[dict]]:
    raw = yaml.safe_load((root / "knowledge/campaigns/directional_v3_manifest.yaml").read_text(encoding="utf-8"))
    return raw, list(raw["jobs"])


def _prep(feat: pl.DataFrame) -> pl.DataFrame:
    f = attach_cross_section(feat)
    if "ticker" not in f.columns:
        f = f.with_columns(pl.col("instrument_id").str.replace("^ticker:", "").alias("ticker"))
    return f


def _news_on_feat(feat: pl.DataFrame, insights: pl.DataFrame, lag: int) -> pl.DataFrame:
    if feat.height == 0:
        return feat
    f = feat.with_columns((pl.col("instrument_id") + pl.lit("|") + pl.col("ts_utc").cast(pl.Utf8)).alias("event_id"))
    return attach_news_state(f, insights, lag_min=lag)


def _bucket(ev: pl.DataFrame, bucket: str) -> pl.DataFrame:
    if ev.height == 0:
        return ev
    sent = pl.col("latest_news_sentiment")
    mins = pl.col("minutes_since_latest_available_news")
    if bucket == "none":
        return ev.filter(sent.is_null() | (mins > 240) | mins.is_null())
    if bucket == "pos":
        return ev.filter((sent == 1) & (mins <= 240))
    if bucket == "neu":
        return ev.filter((sent == 0) & (mins <= 240))
    if bucket == "neg":
        return ev.filter((sent == -1) & (mins <= 240))
    if bucket == "agree":
        return ev.filter(
            ((pl.col("side") == "long") & (sent == 1) | (pl.col("side") == "short") & (sent == -1)) & (mins <= 240)
        )
    if bucket == "fade":
        return ev.filter(
            ((pl.col("side") == "long") & (sent == -1) | (pl.col("side") == "short") & (sent == 1)) & (mins <= 240)
        )
    return ev


def eval_va(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    base = eval_v2_job(spec["v2_state"], feat, exp_id)
    if base.height == 0:
        return base
    base = attach_news_state(base, insights, lag_min=lag)
    out = _bucket(base, spec.get("news") or "none")
    if spec.get("side_mode") == "reverse" and out.height:
        out = out.with_columns(
            pl.when(pl.col("side") == "long").then(pl.lit("short")).otherwise(pl.lit("long")).alias("side")
        )
    return out


def eval_news_px(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    f = _news_on_feat(_prep(feat), insights, lag)
    if f.height == 0 or "latest_news_sentiment" not in f.columns:
        return pl.DataFrame()
    lo, hi = THR["n_observe_lo"], THR["n_observe_hi"]
    r, rv = THR["n_ret"], THR["n_rvol"]
    mins = pl.col("minutes_since_latest_available_news")
    f = f.filter((pl.col("time_et") >= time(9, 45)) & (pl.col("time_et") <= time(14, 30)))
    jid = spec["id"]
    if jid.startswith("N1."):
        hits = f.filter((pl.col("latest_news_sentiment") == 1) & mins.is_between(lo, hi) & (pl.col("ret_5m") >= r) & (pl.col("rvol_5m") >= rv))
        hits = hits.with_columns(pl.lit("long").alias("side"))
    elif jid.startswith("N2."):
        hits = f.filter((pl.col("latest_news_sentiment") == 1) & mins.is_between(lo, hi) & (pl.col("ret_5m") < 0))
        hits = hits.with_columns(pl.lit("short").alias("side"))
    elif jid.startswith("N3."):
        hits = f.filter((pl.col("latest_news_sentiment") == -1) & mins.is_between(lo, hi) & (pl.col("ret_5m") <= -r) & (pl.col("rvol_5m") >= rv))
        hits = hits.with_columns(pl.lit("short").alias("side"))
    elif jid.startswith("N4."):
        hits = f.filter((pl.col("latest_news_sentiment") == -1) & mins.is_between(lo, hi) & (pl.col("ret_5m") > 0))
        hits = hits.with_columns(pl.lit("long").alias("side"))
    else:
        return pl.DataFrame()
    return _stamp(hits, exp_id)


def eval_n5(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    f = _news_on_feat(_prep(feat), insights, lag)
    if f.height == 0 or "resid_5m" not in f.columns:
        return pl.DataFrame()
    lo, hi = THR["n_observe_lo"], THR["n_observe_hi"]
    z = THR["resid_n5"]
    mins = pl.col("minutes_since_latest_available_news")
    f = f.filter((pl.col("time_et") >= time(9, 45)) & (pl.col("time_et") <= time(14, 30)))
    news, resid, side = spec["news"], spec["resid"], spec["side"]
    weak = pl.col("resid_5m") <= -z
    strong = pl.col("resid_5m") >= z
    cond = (pl.col("latest_news_sentiment") == news) & mins.is_between(lo, hi)
    cond = cond & (weak if resid == "weak" else strong)
    hits = f.filter(cond).with_columns(pl.lit(side).alias("side"))
    return _stamp(hits, exp_id)


def eval_ni(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    f = _news_on_feat(_prep(feat), insights, lag)
    if f.height == 0 or "positive_count_2h" not in f.columns:
        return pl.DataFrame()
    n = THR["intensity_n"]
    f = f.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    jid = spec["id"]
    if "multi_pos" in jid:
        hits = f.filter(pl.col("positive_count_2h") >= n).with_columns(pl.lit("long").alias("side"))
    elif "multi_neg" in jid:
        hits = f.filter(pl.col("negative_count_2h") >= n).with_columns(pl.lit("short").alias("side"))
    elif "pub_agree" in jid:
        hits = f.filter((pl.col("publisher_count_2h") >= n) & (pl.col("net_sentiment_2h") > 0)).with_columns(pl.lit("long").alias("side"))
    else:
        hits = f.filter((pl.col("single_ticker_news_count_2h") >= 1) & (pl.col("latest_news_sentiment") == 1)).with_columns(
            pl.lit("long").alias("side")
        )
    return _stamp(hits, exp_id)


def eval_xr(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    f = _prep(feat)
    z, rv = THR["resid_xr"], THR["n_rvol"]
    f = f.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 30)))
    if spec["kind"] == "xs_cond":
        base = eval_v2_job(spec["v2_state"], feat, exp_id)
        if base.height == 0:
            return base
        if "resid_5m" not in base.columns:
            base = attach_cross_section(base)
        if spec.get("xs") == "agree":
            hits = base.filter(
                ((pl.col("side") == "long") & (pl.col("resid_5m") > 0)) | ((pl.col("side") == "short") & (pl.col("resid_5m") < 0))
            )
        else:
            hits = base.filter(
                ((pl.col("side") == "long") & (pl.col("resid_5m") < 0)) | ((pl.col("side") == "short") & (pl.col("resid_5m") > 0))
            )
        return hits
    if spec["id"].startswith("XR.strong"):
        hits = f.filter((pl.col("resid_5m") >= z) & (pl.col("rvol_5m") >= rv) & (pl.col("xs_ret_rank") >= 0.8)).with_columns(
            pl.lit("long").alias("side")
        )
    else:
        hits = f.filter((pl.col("resid_5m") <= -z) & (pl.col("rvol_5m") >= rv) & (pl.col("xs_ret_rank") <= 0.2)).with_columns(
            pl.lit("short").alias("side")
        )
    return _stamp(hits, exp_id)


def eval_nr(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    f = _news_on_feat(_prep(feat), insights, lag)
    if f.height == 0 or "resid_5m" not in f.columns:
        return pl.DataFrame()
    z = THR["resid_nr"]
    lo, hi = THR["n_observe_lo"], THR["n_observe_hi"]
    mins = pl.col("minutes_since_latest_available_news")
    f = f.filter((pl.col("time_et") >= time(9, 45)) & (pl.col("time_et") <= time(14, 30)) & mins.is_between(lo, hi))
    jid = spec["id"]
    if "pos_strong" in jid:
        hits = f.filter((pl.col("latest_news_sentiment") == 1) & (pl.col("resid_5m") >= z)).with_columns(pl.lit("long").alias("side"))
    elif "pos_weak" in jid:
        hits = f.filter((pl.col("latest_news_sentiment") == 1) & (pl.col("resid_5m") <= -z)).with_columns(pl.lit("short").alias("side"))
    elif "neg_weak" in jid:
        hits = f.filter((pl.col("latest_news_sentiment") == -1) & (pl.col("resid_5m") <= -z)).with_columns(pl.lit("short").alias("side"))
    else:
        hits = f.filter((pl.col("latest_news_sentiment") == -1) & (pl.col("resid_5m") >= z)).with_columns(pl.lit("long").alias("side"))
    return _stamp(hits, exp_id)


def eval_bl(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    f = _news_on_feat(_prep(feat), insights, lag)
    f = f.filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 0)))
    jid = spec["id"]
    if "news_pos" in jid:
        hits = f.filter((pl.col("latest_news_sentiment") == 1) & (pl.col("minutes_since_latest_available_news") <= 120)).with_columns(
            pl.lit("long").alias("side")
        )
    elif "news_neg" in jid:
        hits = f.filter((pl.col("latest_news_sentiment") == -1) & (pl.col("minutes_since_latest_available_news") <= 120)).with_columns(
            pl.lit("short").alias("side")
        )
    else:
        f = _prep(feat).filter((pl.col("time_et") >= time(10, 0)) & (pl.col("time_et") <= time(14, 0)))
        hits = f.filter(pl.col("resid_5m") >= THR["resid_xr"]).with_columns(pl.lit("long").alias("side"))
    return _stamp(hits, exp_id)


def eval_v3_job(spec: dict, feat: pl.DataFrame, insights: pl.DataFrame, lag: int, exp_id: str) -> pl.DataFrame:
    k = spec.get("kind")
    if k == "cond":
        return eval_va(spec, feat, insights, lag, exp_id)
    if k == "news_px":
        return eval_news_px(spec, feat, insights, lag, exp_id)
    if k == "n5":
        return eval_n5(spec, feat, insights, lag, exp_id)
    if k == "intensity":
        return eval_ni(spec, feat, insights, lag, exp_id)
    if k in {"xs", "xs_cond"}:
        return eval_xr(spec, feat, insights, lag, exp_id)
    if k == "nr":
        return eval_nr(spec, feat, insights, lag, exp_id)
    if k == "baseline":
        return eval_bl(spec, feat, insights, lag, exp_id)
    return pl.DataFrame()
