"""V4R Stage-1 day builder. Own store. Does not write V4 paths."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.features.causal_store import day_path as causal_day_path
from quant_edge_lab.features.v4 import assert_available_le_decision, ensure_utc, on_decision_grid
from quant_edge_lab.features.v4.features import add_activity_proxies, add_legacy_flags, add_own_and_activity, enrich_grid
from quant_edge_lab.features.v4.residuals import attach_loo_residuals
from quant_edge_lab.features.v4r.beta import attach_beta, beta_from_history, daily_return_panel, loo_daily_market
from quant_edge_lab.features.v4r.target import attach_forward_holding_residuals
from quant_edge_lab.peers import GRAPH_FEATURE_COLS


def v4r_root_data(root: Path) -> Path:
    return Paths(root).features


def v4r_store(root: Path) -> Path:
    p = v4r_root_data(root) / "cross_sectional_v4r"
    p.mkdir(parents=True, exist_ok=True)
    return p


def v4r_day_path(root: Path, day: str) -> Path:
    return v4r_store(root) / f"date={day}" / "part.parquet"


def v4r_peer_store(root: Path) -> Path:
    p = v4r_root_data(root) / "cross_sectional_v4r_peers"
    p.mkdir(parents=True, exist_ok=True)
    return p


def v4r_peer_day_path(root: Path, day: str) -> Path:
    return v4r_peer_store(root) / f"date={day}" / "part.parquet"


def v4r_daily_store(root: Path) -> Path:
    p = v4r_root_data(root) / "v4r_daily"
    p.mkdir(parents=True, exist_ok=True)
    return p


def v4r_daily_path(root: Path, day: str) -> Path:
    return v4r_daily_store(root) / f"date={day}" / "part.parquet"


def v4r_graph_dir(root: Path) -> Path:
    p = Paths(root).derived / "features" / "peer_graph_v4r"
    p.mkdir(parents=True, exist_ok=True)
    return p


def v4r_ckpt_dir(root: Path) -> Path:
    p = Paths(root).derived / "discovery" / "batches" / "dir-campaign-v4r"
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_parquet(df: pl.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".new")
    df.write_parquet(tmp)
    tmp.replace(dest)


def assert_unique_keys(df: pl.DataFrame, *, where: str) -> None:
    keys = [c for c in ("trading_date", "instrument_id", "decision_ts") if c in df.columns]
    if len(keys) < 3:
        raise RuntimeError(f"{where}: missing join keys {keys}")
    n = df.height
    u = df.select(keys).n_unique()
    if n != u:
        raise RuntimeError(f"{where}: duplicate keys rows={n} unique={u}")


def add_xs(feat: pl.DataFrame) -> pl.DataFrame:
    if feat.height == 0 or "resid_ret_5m" not in feat.columns:
        return feat
    n = pl.len().over("decision_ts")
    feat = feat.with_columns(
        (pl.col("resid_ret_5m").rank(method="average").over("decision_ts") / n).alias("resid_rank_5m"),
        (pl.col("resid_ret_5m").abs().median().over("decision_ts")).alias("xs_dispersion_5m"),
        (pl.col("resid_ret_5m") > 0).mean().over("decision_ts").alias("xs_breadth_5m"),
    )
    return feat.with_columns(
        (pl.col("resid_rank_5m") - pl.col("resid_rank_5m").shift(1).over("instrument_id")).alias("resid_rank_change_5m")
    )


def apply_clock(feat: pl.DataFrame, clock_mean: dict[int, float]) -> pl.DataFrame:
    if "minutes_from_open" not in feat.columns and "time_et" in feat.columns:
        feat = feat.with_columns(
            (pl.col("time_et").dt.hour().cast(pl.Int32) * 60 + pl.col("time_et").dt.minute().cast(pl.Int32) - (9 * 60 + 30)).alias(
                "minutes_from_open"
            )
        )
    if not clock_mean:
        feat = feat.with_columns(pl.lit(None).cast(pl.Float64).alias("same_clock_mean_return"))
        if "ret_5m" in feat.columns:
            feat = feat.with_columns((pl.col("ret_5m") - pl.col("same_clock_mean_return").fill_null(0.0)).alias("same_clock_surprise"))
        return feat
    cmap = pl.DataFrame({"minutes_from_open": list(clock_mean.keys()), "same_clock_mean_return": list(clock_mean.values())})
    if "same_clock_mean_return" in feat.columns:
        feat = feat.drop("same_clock_mean_return")
    feat = feat.join(cmap, on="minutes_from_open", how="left")
    if "ret_5m" in feat.columns:
        feat = feat.with_columns((pl.col("ret_5m") - pl.col("same_clock_mean_return").fill_null(0.0)).alias("same_clock_surprise"))
    return feat


def build_v4r_day(
    root: Path,
    day: str,
    *,
    beta_map: dict[str, float],
    clock_mean: dict[int, float],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    p = causal_day_path(root, day)
    if not p.exists():
        raise FileNotFoundError(f"causal_v2 missing for {day}")
    feat = pl.read_parquet(p)
    feat = on_decision_grid(feat)
    if "trading_date" not in feat.columns:
        feat = feat.with_columns(pl.lit(day).alias("trading_date"))
    if "decision_ts" in feat.columns:
        feat = ensure_utc(feat, "decision_ts")
    if "available_at" in feat.columns:
        feat = ensure_utc(feat, "available_at")
        assert_available_le_decision(feat, "available_at", "decision_ts")
    if "prev_close" in feat.columns:
        feat = feat.filter(pl.col("prev_close") >= 1.0)
    if "typ_1m_vol" in feat.columns and "prev_close" in feat.columns:
        feat = feat.with_columns((pl.col("typ_1m_vol") * pl.col("prev_close") * 390.0).alias("trailing_rth_dvol_proxy"))
        feat = feat.filter(pl.col("trailing_rth_dvol_proxy") >= 1_000_000)
    # weights then LOO market; beta attached BEFORE residual so missing beta → null resid
    if "ticker" not in feat.columns and "instrument_id" in feat.columns:
        feat = feat.with_columns(pl.col("instrument_id").str.replace("^ticker:", "").alias("ticker"))
    if "weight_mkt" not in feat.columns:
        if "med20_dvol" in feat.columns:
            feat = feat.with_columns(pl.col("med20_dvol").sqrt().alias("weight_mkt"))
        elif "typ_1m_vol" in feat.columns and "prev_close" in feat.columns:
            feat = feat.with_columns((pl.col("typ_1m_vol") * pl.col("prev_close") * 390).sqrt().alias("weight_mkt"))
        else:
            feat = feat.with_columns(pl.lit(1.0).alias("weight_mkt"))
    if "beta_20d" in feat.columns:
        feat = feat.drop("beta_20d")
    feat = attach_beta(feat, beta_map)
    if "ret_5m" in feat.columns:
        feat = attach_loo_residuals(feat, ret_col="ret_5m", weight_col="weight_mkt", out_mkt="market_ret_5m", out_resid="resid_ret_5m_raw", beta_col=None)
        # override resid with causal beta; null beta → null resid (no beta=1)
        feat = feat.with_columns(
            pl.when(pl.col("beta_20d").is_not_null())
            .then(pl.col("ret_5m") - pl.col("beta_20d") * pl.col("market_ret_5m"))
            .otherwise(None)
            .alias("resid_ret_5m")
        )
        if "resid_ret_5m_raw" in feat.columns:
            feat = feat.drop("resid_ret_5m_raw")
    feat = add_own_and_activity(feat)
    feat = add_legacy_flags(feat)
    feat = add_xs(feat)
    feat = apply_clock(feat, clock_mean)
    feat = add_activity_proxies(feat)
    drop_g = [c for c in GRAPH_FEATURE_COLS if c in feat.columns]
    if drop_g:
        feat = feat.drop(drop_g)
    feat = attach_forward_holding_residuals(feat)
    assert_unique_keys(feat, where=f"v4r stage1 {day}")
    daily = loo_daily_market(daily_return_panel(feat))
    return feat, daily
