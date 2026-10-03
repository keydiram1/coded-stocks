"""Massive news cache, coverage, and PIT-safe ticker-insight features.

Primary availability: published_utc + 60 minutes.
sentiment_reasoning is stored for audit and MUST NOT be used as a model feature.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.client import MassiveClient, probe_bases
from quant_edge_lab.secrets import massive_api_key

console = Console()
NEWS_VERSION = "massive_news_v1"
CANONICAL_UTC = pl.Datetime("us", time_zone="UTC")
PRIMARY_LAG_MIN = 60
LAGS = (60, 90, 120)
SENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}
INSIGHTS_ONSET_MIN_PCT = 50.0
INSIGHTS_RELIABLE_PCT = 80.0


def news_root(root: Path) -> Path:
    p = Paths(root).derived / "news" / NEWS_VERSION
    p.mkdir(parents=True, exist_ok=True)
    return p


def _month_starts(start: str, end: str) -> list[tuple[str, str]]:
    cur = datetime.fromisoformat(start).replace(tzinfo=None)
    stop = datetime.fromisoformat(end).replace(tzinfo=None)
    out = []
    while cur < stop:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        gte = cur.strftime("%Y-%m-%d")
        lt = min(nxt, stop).strftime("%Y-%m-%d")
        out.append((gte, lt))
        cur = nxt
    return out


def _sent_score(label: str | None) -> int | None:
    if not label:
        return None
    if label in SENT_MAP:
        return SENT_MAP[label]
    return None


def with_utc_us(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Attach/convert `col` to datetime[μs, UTC] without shifting a UTC instant.

    Naive values are treated as UTC wall times (repo convention for bar timestamps).
    Already-aware values are converted to UTC.
    """
    if col not in df.columns or df.height == 0:
        return df
    dt = df.schema[col]
    if dt == CANONICAL_UTC:
        return df
    if dt in (pl.String, pl.Utf8):
        return df.with_columns(pl.col(col).str.to_datetime(time_zone="UTC").cast(CANONICAL_UTC).alias(col))
    if not isinstance(dt, pl.Datetime):
        raise TypeError(f"{col} must be datetime or str, got {dt}")
    if dt.time_zone in (None, ""):
        return df.with_columns(pl.col(col).cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias(col))
    if dt.time_zone == "UTC":
        return df.with_columns(pl.col(col).cast(CANONICAL_UTC).alias(col))
    return df.with_columns(pl.col(col).dt.convert_time_zone("UTC").cast(CANONICAL_UTC).alias(col))


def flatten_article(article: dict[str, Any]) -> list[dict[str, Any]]:
    tickers = list(article.get("tickers") or [])
    n_t = len(tickers)
    pub = (article.get("publisher") or {}).get("name")
    rows = []
    for ins in article.get("insights") or []:
        lab = ins.get("sentiment")
        score = _sent_score(lab)
        if score is None:
            continue
        tk = ins.get("ticker")
        if not tk:
            continue
        rows.append(
            {
                "article_id": article.get("id"),
                "ticker": tk,
                "published_utc": article.get("published_utc"),
                "publisher": pub,
                "title": article.get("title"),
                "description": article.get("description"),
                "article_ticker_count": n_t,
                "sentiment": lab,
                "sentiment_score": score,
                "sentiment_reasoning": ins.get("sentiment_reasoning"),
                "keywords": json.dumps(article.get("keywords") or []),
                "author": article.get("author"),
                "article_url": article.get("article_url"),
            }
        )
    return rows


def ingest_month(client: MassiveClient, gte: str, lt: str, dest: Path, *, max_items: int | None = None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return int(pl.read_parquet(dest).height)
    rows: list[dict[str, Any]] = []
    n_art = 0
    for article in client.paginate(
        "/v2/reference/news",
        {
            "published_utc.gte": gte,
            "published_utc.lt": lt,
            "limit": 1000,
            "order": "asc",
            "sort": "published_utc",
        },
        page_sleep_s=0.08,
        max_items=max_items,
    ):
        n_art += 1
        rows.extend(flatten_article(article))
    schema = {
        "article_id": pl.String,
        "ticker": pl.String,
        "published_utc": pl.String,
        "publisher": pl.String,
        "title": pl.String,
        "description": pl.String,
        "article_ticker_count": pl.Int64,
        "sentiment": pl.String,
        "sentiment_score": pl.Int64,
        "sentiment_reasoning": pl.String,
        "keywords": pl.String,
        "author": pl.String,
        "article_url": pl.String,
    }
    if not rows:
        pl.DataFrame(schema=schema).write_parquet(dest)
        return 0
    pl.DataFrame(rows).write_parquet(dest)
    console.print(f"news cache {gte} articles≈{n_art} insights={len(rows)}")
    return len(rows)


def coverage_from_frame(df: pl.DataFrame, month: str) -> dict[str, Any]:
    if df.height == 0:
        return {
            "month": month,
            "insight_rows": 0,
            "article_count": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "unique_tickers": 0,
            "unique_publishers": 0,
            "pct_articles_with_insights": None,
        }
    arts = df["article_id"].n_unique()
    return {
        "month": month,
        "insight_rows": df.height,
        "article_count": arts,
        "positive": int((df["sentiment_score"] == 1).sum()),
        "neutral": int((df["sentiment_score"] == 0).sum()),
        "negative": int((df["sentiment_score"] == -1).sum()),
        "unique_tickers": df["ticker"].n_unique(),
        "unique_publishers": df["publisher"].n_unique(),
        "pct_articles_with_insights": 100.0,
    }


def probe_month_article_insight_rate(client: MassiveClient, gte: str, lt: str, n: int = 400) -> dict[str, Any]:
    arts = []
    for article in client.paginate(
        "/v2/reference/news",
        {
            "published_utc.gte": gte,
            "published_utc.lt": lt,
            "limit": min(n, 1000),
            "order": "asc",
            "sort": "published_utc",
        },
        page_sleep_s=0.05,
        max_items=n,
    ):
        arts.append(article)
    if not arts:
        return {"month": gte[:7], "n_articles": 0, "n_with_insights": 0, "pct_with_insights": 0.0}
    n_ins = sum(1 for a in arts if a.get("insights"))
    return {
        "month": gte[:7],
        "n_articles": len(arts),
        "n_with_insights": n_ins,
        "pct_with_insights": round(100.0 * n_ins / len(arts), 2),
    }


def load_existing_news_meta(root: Path) -> dict[str, Any] | None:
    nr = news_root(root)
    cov = nr / "coverage.json"
    combined = nr / "insights_all.parquet"
    if not cov.exists() or not combined.exists():
        return None
    meta = json.loads(cov.read_text(encoding="utf-8"))
    if not meta.get("ok"):
        return None
    return meta


def validate_news_cache(root: Path) -> dict[str, Any]:
    """Mechanical cache checks. Does not modify sentiment labels."""
    ins = load_insights(root)
    if ins.height == 0:
        raise RuntimeError("news cache empty or missing")
    ins = insights_with_lag(ins, PRIMARY_LAG_MIN)
    sent = ins["sentiment_score"]
    dups = ins.select(["article_id", "ticker"]).is_duplicated().sum() if "article_id" in ins.columns else 0
    pub_dt = ins.schema["published_ts"]
    avail_dt = ins.schema["available_at"]
    if pub_dt != CANONICAL_UTC or avail_dt != CANONICAL_UTC:
        raise TypeError(f"cache timestamps must be {CANONICAL_UTC}; published_ts={pub_dt} available_at={avail_dt}")
    out = {
        "n_insight_rows": ins.height,
        "published_ts_min": str(ins["published_ts"].min()),
        "published_ts_max": str(ins["published_ts"].max()),
        "news_available_at_min": str(ins["available_at"].min()),
        "news_available_at_max": str(ins["available_at"].max()),
        "published_ts_dtype": str(pub_dt),
        "available_at_dtype": str(avail_dt),
        "positive": int((sent == 1).sum()),
        "neutral": int((sent == 0).sum()),
        "negative": int((sent == -1).sum()),
        "null_sentiment": int(sent.null_count()),
        "null_published": int(ins["published_ts"].null_count()),
        "duplicate_article_ticker_rows": int(dups),
        "canonical_dtype": str(CANONICAL_UTC),
    }
    console.print(f"news cache validate {out}")
    return out


def ensure_news_cache(root: Path, *, probe_start: str = "2023-01-01", panel_end: str = "2026-10-01") -> dict[str, Any]:
    existing = load_existing_news_meta(root)
    if existing is not None:
        console.print(
            f"news cache reuse n={existing.get('n_insight_rows')} onset={existing.get('onset_gte')} (no redownload)"
        )
        return existing
    key = massive_api_key(root)
    base = probe_bases(key)
    client = MassiveClient(api_key=key, base_url=base, max_retries=5)
    nr = news_root(root)
    probe_path = nr / "coverage_probe.json"
    months = _month_starts(probe_start, panel_end)
    if probe_path.exists():
        probe_rows = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        probe_rows = []
        for gte, lt in months:
            rec = probe_month_article_insight_rate(client, gte, lt, n=400)
            rec["gte"] = gte
            rec["lt"] = lt
            probe_rows.append(rec)
            console.print(f"probe {rec['month']} pct_insights={rec['pct_with_insights']} n={rec['n_articles']}")
        probe_path.write_text(json.dumps(probe_rows, indent=2), encoding="utf-8")

    reliable = [
        r
        for r in probe_rows
        if (r.get("pct_with_insights") or 0) >= INSIGHTS_RELIABLE_PCT and r.get("n_articles", 0) >= 50
    ]
    usable = [
        r
        for r in probe_rows
        if (r.get("pct_with_insights") or 0) >= INSIGHTS_ONSET_MIN_PCT and r.get("n_articles", 0) >= 50
    ]
    if not usable:
        return {
            "ok": False,
            "reason": "No month reached 50% articles-with-insights in the probe. Refusing expensive V3.",
            "probe": probe_rows,
            "base_url": base,
        }
    onset = (reliable or usable)[0]["gte"]
    cache_months = [m for m in months if m[0] >= onset]
    monthly_cov = []
    parts = []
    for gte, lt in cache_months:
        dest = nr / "insights" / f"month={gte[:7]}" / "part.parquet"
        ingest_month(client, gte, lt, dest, max_items=None)
        df = pl.read_parquet(dest)
        monthly_cov.append(coverage_from_frame(df, gte[:7]))
        if df.height:
            parts.append(df)
    all_ins = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    combined = nr / "insights_all.parquet"
    if all_ins.height:
        all_ins.write_parquet(combined)
    meta = {
        "ok": True,
        "base_url": base,
        "onset_gte": onset,
        "panel_end": panel_end,
        "primary_lag_min": PRIMARY_LAG_MIN,
        "lags": list(LAGS),
        "reliable_pct_threshold": INSIGHTS_RELIABLE_PCT,
        "onset_rule": "first month with >=80% insights (else first >=50%) on 400-article probe",
        "n_insight_rows": int(all_ins.height),
        "monthly": monthly_cov,
        "probe": probe_rows,
        "cache": str(combined),
        "sector_pit": False,
        "sector_note": "No PIT-safe sector/industry membership in this repo; sector features omitted.",
        "built_at": datetime.now(UTC).isoformat(),
    }
    (nr / "coverage.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_insights(root: Path) -> pl.DataFrame:
    p = news_root(root) / "insights_all.parquet"
    if not p.exists():
        return pl.DataFrame()
    df = pl.read_parquet(p)
    if df.height == 0:
        return df
    src = "published_ts" if "published_ts" in df.columns else "published_utc"
    if src != "published_ts":
        df = df.with_columns(pl.col(src).alias("published_ts"))
    return with_utc_us(df, "published_ts")


def insights_with_lag(df: pl.DataFrame, lag_min: int) -> pl.DataFrame:
    if df.height == 0:
        return df
    df = with_utc_us(df, "published_ts")
    out = df.with_columns((pl.col("published_ts") + pl.duration(minutes=lag_min)).alias("available_at"))
    return with_utc_us(out, "available_at")


def attach_news_state(events: pl.DataFrame, insights: pl.DataFrame, *, lag_min: int = PRIMARY_LAG_MIN) -> pl.DataFrame:
    """Join latest and window counts. Does not attach sentiment_reasoning."""
    if events.height == 0:
        return events
    if "ticker" not in events.columns:
        events = events.with_columns(pl.col("instrument_id").str.replace("^ticker:", "").alias("ticker"))
    empty_cols = {
        "latest_news_sentiment": pl.Int64,
        "minutes_since_latest_available_news": pl.Float64,
        "positive_count_2h": pl.Int64,
        "neutral_count_2h": pl.Int64,
        "negative_count_2h": pl.Int64,
        "positive_count_today": pl.Int64,
        "neutral_count_today": pl.Int64,
        "negative_count_today": pl.Int64,
        "article_count_2h": pl.Int64,
        "publisher_count_2h": pl.Int64,
        "single_ticker_news_count_2h": pl.Int64,
        "multi_ticker_news_count_2h": pl.Int64,
        "net_sentiment_2h": pl.Int64,
        "net_sentiment_today": pl.Int64,
        "latest_article_ticker_count": pl.Int64,
        "same_direction_news_count": pl.Int64,
        "sentiment_last_2h": pl.Int64,
        "sentiment_last_4h": pl.Int64,
        "sentiment_today": pl.Int64,
    }
    if insights.height == 0:
        return events.with_columns([pl.lit(None).cast(dt).alias(c) if "latest" in c else pl.lit(0).cast(dt).alias(c) for c, dt in empty_cols.items() if c not in events.columns])
    ins = insights_with_lag(insights, lag_min).filter(pl.col("available_at").is_not_null())
    ins = with_utc_us(ins, "available_at")
    ev = with_utc_us(events, "decision_ts")
    if ev.schema["decision_ts"] != CANONICAL_UTC or ins.schema["available_at"] != CANONICAL_UTC:
        raise TypeError(
            f"join keys must be {CANONICAL_UTC}; got decision_ts={ev.schema['decision_ts']} "
            f"available_at={ins.schema['available_at']}"
        )
    ins = ins.sort(["ticker", "available_at"])
    ev = ev.sort(["ticker", "decision_ts"])
    latest = ev.join_asof(
        ins.select(["ticker", "available_at", "sentiment_score", "article_ticker_count", "article_id", "publisher"]).rename(
            {
                "available_at": "latest_news_available_at",
                "sentiment_score": "latest_news_sentiment",
                "article_ticker_count": "latest_article_ticker_count",
                "article_id": "latest_article_id",
                "publisher": "latest_publisher",
            }
        ),
        left_on="decision_ts",
        right_on="latest_news_available_at",
        by="ticker",
        strategy="backward",
        check_sortedness=False,
    )
    latest = latest.with_columns(
        ((pl.col("decision_ts") - pl.col("latest_news_available_at")).dt.total_minutes()).alias(
            "minutes_since_latest_available_news"
        )
    )
    win = ev.select(["event_id", "ticker", "decision_ts"]).join(
        ins.select(["ticker", "available_at", "sentiment_score", "article_id", "publisher", "article_ticker_count"]),
        on="ticker",
        how="inner",
    )
    win = win.filter(pl.col("available_at") <= pl.col("decision_ts"))
    w2 = win.filter(pl.col("available_at") > pl.col("decision_ts") - pl.duration(hours=2))
    w4 = win.filter(pl.col("available_at") > pl.col("decision_ts") - pl.duration(hours=4))
    wt = win.filter(pl.col("available_at") >= pl.col("decision_ts") - pl.duration(hours=12))

    def _agg(w: pl.DataFrame, suffix: str) -> pl.DataFrame:
        if w.height == 0:
            return pl.DataFrame(schema={"event_id": pl.String})
        return w.group_by("event_id").agg(
            (pl.col("sentiment_score") == 1).sum().alias(f"positive_count_{suffix}"),
            (pl.col("sentiment_score") == 0).sum().alias(f"neutral_count_{suffix}"),
            (pl.col("sentiment_score") == -1).sum().alias(f"negative_count_{suffix}"),
            pl.col("article_id").n_unique().alias(f"article_count_{suffix}"),
            pl.col("publisher").n_unique().alias(f"publisher_count_{suffix}"),
            (pl.col("article_ticker_count") == 1).sum().alias(f"single_ticker_news_count_{suffix}"),
            (pl.col("article_ticker_count") > 1).sum().alias(f"multi_ticker_news_count_{suffix}"),
            pl.col("sentiment_score").sum().alias(f"net_sentiment_{suffix}"),
        )

    out = latest
    for a in (_agg(w2, "2h"), _agg(w4, "4h"), _agg(wt, "today")):
        if a.height and "event_id" in a.columns:
            out = out.join(a, on="event_id", how="left")
    for c in list(out.columns):
        if c.endswith("_2h") or c.endswith("_4h") or c.endswith("_today"):
            out = out.with_columns(pl.col(c).fill_null(0))
    if "net_sentiment_2h" in out.columns:
        out = out.with_columns(pl.col("net_sentiment_2h").alias("sentiment_last_2h"))
    if "net_sentiment_4h" in out.columns:
        out = out.with_columns(pl.col("net_sentiment_4h").alias("sentiment_last_4h"))
    if "net_sentiment_today" in out.columns:
        out = out.with_columns(pl.col("net_sentiment_today").alias("sentiment_today"))
    if "latest_news_sentiment" in out.columns and "positive_count_2h" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("latest_news_sentiment") == 1)
            .then(pl.col("positive_count_2h"))
            .when(pl.col("latest_news_sentiment") == -1)
            .then(pl.col("negative_count_2h"))
            .otherwise(pl.col("neutral_count_2h"))
            .alias("same_direction_news_count")
        )
    return out


def attach_cross_section(feat: pl.DataFrame) -> pl.DataFrame:
    if feat.height == 0:
        return feat
    extra = []
    if "ret_5m" in feat.columns and "mkt_ret_5m" in feat.columns:
        extra.append((pl.col("ret_5m") - pl.col("mkt_ret_5m")).alias("resid_5m"))
    if "ret_15m" in feat.columns and "mkt_ret_15m" in feat.columns:
        extra.append((pl.col("ret_15m") - pl.col("mkt_ret_15m")).alias("resid_15m"))
    if "rvol_5m" in feat.columns:
        extra.append(pl.col("rvol_5m").mean().over("time_et").alias("mkt_rvol_5m"))
    if extra:
        feat = feat.with_columns(extra)
    return feat
