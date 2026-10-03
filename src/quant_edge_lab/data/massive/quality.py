"""Streaming certification of converted minute Flat Files. Never loads the full panel."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_edge_lab.config import Paths
from quant_edge_lab.data.massive.flatfiles import DATASET, load_manifest, local_gz_path, local_parquet_path

OHLC_OK = (pl.col("high") >= pl.max_horizontal("open", "close", "low")) & (
    pl.col("low") <= pl.min_horizontal("open", "close", "high")
)
OVERNIGHT_EXTREME = 0.40
INTRABAR_EXTREME = 0.30


def _out_dir(root: Path) -> Path:
    p = Paths(root).reports / "data_quality"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _weekday_gaps(start: str, end: str, have: set[str]) -> list[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    gaps = []
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5:
            s = cur.isoformat()
            if s not in have:
                gaps.append(s)
        cur += timedelta(days=1)
    return gaps


def _quantiles(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"min": None, "median": None, "mean": None, "max": None}
    s = pl.Series(vals)
    return {
        "min": float(s.min()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "max": float(s.max()),
    }


def _cat(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def decide_verdict(categories: dict[str, dict[str, str]]) -> str:
    if any(c["status"] == "FAIL" for c in categories.values()):
        return "NOT_READY"
    return "READY_FOR_EXPLORATORY_RESEARCH"


def certify_massive_flat(root: Path) -> dict[str, Any]:
    """Full-panel quality audit. Day-by-day scans only."""
    man = load_manifest(root)
    meta = man.get("meta", {})
    start = str(meta.get("accessible_start") or "2021-10-01")
    end = str(meta.get("accessible_end") or "2026-10-01")
    files = man.get("files", {})
    expected = sorted(
        d for d, rec in files.items() if rec.get("status") in {"downloaded", "skipped_verified"}
    )
    missing_gz = [d for d in expected if not local_gz_path(root, d).exists()]
    missing_pq = [d for d in expected if not local_parquet_path(root, d).exists()]
    failed = [
        d
        for d, rec in files.items()
        if rec.get("status") == "failed" or rec.get("conversion_status") == "failed"
    ]
    have = set(expected)
    weekday_gaps = _weekday_gaps(start, end, have)

    paths = Paths(root)
    inst_path = paths.normalized / DATASET / "instruments.parquet"
    splits_path = paths.normalized / DATASET / "corporate_actions.parquet"
    instruments = pl.read_parquet(inst_path) if inst_path.exists() else pl.DataFrame()
    splits = pl.read_parquet(splits_path) if splits_path.exists() else pl.DataFrame()
    split_keys: set[tuple[str, str]] = set()
    if splits.height and "ticker" in splits.columns and "effective_date" in splits.columns:
        for row in splits.iter_rows(named=True):
            ed = str(row.get("effective_date") or "")[:10]
            if ed:
                split_keys.add((str(row["ticker"]), ed))

    tickers: set[str] = set()
    by_date: list[dict[str, Any]] = []
    dup_rows = 0
    dup_tickers: set[str] = set()
    dup_dates: list[str] = []
    ohlc_bad = 0
    price_bad = 0
    vol_bad = 0
    txn_bad = 0
    session_out = 0
    off_minute = 0
    n_pre = n_rth = n_ah = 0
    total_rows = 0
    ts_min = None
    ts_max = None
    ohlc_examples: list[dict[str, Any]] = []
    price_examples: list[dict[str, Any]] = []
    overnight_explained: list[dict[str, Any]] = []
    overnight_unexplained: list[dict[str, Any]] = []
    volume_extremes: list[dict[str, Any]] = []
    price_extremes: list[dict[str, Any]] = []
    move_extremes: list[dict[str, Any]] = []
    prev_rth_close: dict[str, float] = {}
    coverage_jumps: list[dict[str, Any]] = []
    prev_n_tickers = None
    out = _out_dir(root)
    progress_path = out / "_certify_progress.json"

    for i, day in enumerate(expected):
        pq = local_parquet_path(root, day)
        if not pq.exists():
            continue
        lf = pl.scan_parquet(pq)
        et = (
            pl.col("ts_utc")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("America/New_York")
        )
        stats = lf.select(
            pl.len().alias("n"),
            pl.col("ticker").n_unique().alias("n_tickers"),
            pl.col("ts_utc").min().alias("ts_min"),
            pl.col("ts_utc").max().alias("ts_max"),
            (~OHLC_OK).sum().alias("bad_ohlc"),
            (
                (pl.col("open") <= 0)
                | (pl.col("high") <= 0)
                | (pl.col("low") <= 0)
                | (pl.col("close") <= 0)
            )
            .sum()
            .alias("p_bad"),
            (pl.col("volume") < 0).sum().alias("v_bad"),
            pl.when(pl.col("transactions").is_not_null())
            .then(pl.col("transactions") < 0)
            .otherwise(False)
            .sum()
            .alias("txn_bad"),
            ((et.dt.time() < dtime(4, 0)) | (et.dt.time() >= dtime(20, 0))).sum().alias("odd"),
            ((et.dt.second() != 0) | (et.dt.microsecond() != 0)).sum().alias("off_min"),
            ((et.dt.time() >= dtime(4, 0)) & (et.dt.time() < dtime(9, 30))).sum().alias("n_pre"),
            ((et.dt.time() >= dtime(9, 30)) & (et.dt.time() < dtime(16, 0))).sum().alias("n_rth"),
            ((et.dt.time() >= dtime(16, 0)) & (et.dt.time() < dtime(20, 0))).sum().alias("n_ah"),
            pl.col("volume").max().alias("max_vol"),
            pl.col("close").max().alias("max_close"),
            pl.col("close").min().alias("min_close"),
        ).collect()
        n = int(stats["n"][0])
        n_tickers = int(stats["n_tickers"][0])
        total_rows += n
        dmin = stats["ts_min"][0]
        dmax = stats["ts_max"][0]
        if ts_min is None or (dmin is not None and dmin < ts_min):
            ts_min = dmin
        if ts_max is None or (dmax is not None and dmax > ts_max):
            ts_max = dmax
        day_ohlc = int(stats["bad_ohlc"][0])
        ohlc_bad += day_ohlc
        day_p = int(stats["p_bad"][0])
        price_bad += day_p
        vol_bad += int(stats["v_bad"][0])
        txn_bad += int(stats["txn_bad"][0])
        session_out += int(stats["odd"][0])
        off_minute += int(stats["off_min"][0])
        n_pre += int(stats["n_pre"][0])
        n_rth += int(stats["n_rth"][0])
        n_ah += int(stats["n_ah"][0])

        day_tickers = lf.select(pl.col("ticker").unique()).collect()["ticker"].to_list()
        tickers.update(day_tickers)

        dup_df = (
            lf.group_by(["ticker", "ts_utc"])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
            .collect()
        )
        day_dups = int(dup_df["n"].sum()) if dup_df.height else 0
        extra_dup_rows = int((dup_df["n"] - 1).sum()) if dup_df.height else 0
        dup_rows += extra_dup_rows
        if extra_dup_rows:
            dup_dates.append(day)
            dup_tickers.update(dup_df["ticker"].to_list())

        if day_ohlc and len(ohlc_examples) < 25:
            ex = lf.filter(~OHLC_OK).select(["ticker", "ts_utc", "open", "high", "low", "close"]).head(5).collect()
            ohlc_examples.extend({"date": day, **r} for r in ex.to_dicts())
        if day_p and len(price_examples) < 25:
            ex = (
                lf.filter(
                    (pl.col("open") <= 0)
                    | (pl.col("high") <= 0)
                    | (pl.col("low") <= 0)
                    | (pl.col("close") <= 0)
                )
                .select(["ticker", "ts_utc", "open", "high", "low", "close"])
                .head(5)
                .collect()
            )
            price_examples.extend({"date": day, **r} for r in ex.to_dicts())

        rth = (
            lf.with_columns(et.alias("ts_et"))
            .filter((pl.col("ts_et").dt.time() >= dtime(9, 30)) & (pl.col("ts_et").dt.time() < dtime(16, 0)))
            .sort(["ticker", "ts_utc"])
            .group_by("ticker")
            .agg(
                pl.col("open").first().alias("rth_open"),
                pl.col("close").last().alias("rth_close"),
            )
            .collect()
        )
        for row in rth.iter_rows(named=True):
            t = row["ticker"]
            px = row["rth_open"]
            prev = prev_rth_close.get(t)
            if prev and prev > 0 and px and px > 0:
                ret = px / prev - 1.0
                if abs(ret) >= OVERNIGHT_EXTREME:
                    rec = {
                        "date": day,
                        "ticker": t,
                        "prev_rth_close": prev,
                        "rth_open": px,
                        "overnight_return": ret,
                    }
                    if (t, day) in split_keys:
                        rec["split"] = True
                        if len(overnight_explained) < 200:
                            overnight_explained.append(rec)
                    else:
                        rec["split"] = False
                        if len(overnight_unexplained) < 200:
                            overnight_unexplained.append(rec)
            if row["rth_close"]:
                prev_rth_close[t] = float(row["rth_close"])

        max_vol = stats["max_vol"][0]
        if max_vol is not None and max_vol >= 20_000_000 and len(volume_extremes) < 80:
            topv = lf.filter(pl.col("volume") == max_vol).select(["ticker", "ts_utc", "volume", "close"]).head(1).collect()
            volume_extremes.extend({"date": day, **r} for r in topv.to_dicts())
        max_c = stats["max_close"][0]
        min_c = stats["min_close"][0]
        if max_c is not None and max_c >= 5_000 and len(price_extremes) < 80:
            topp = lf.filter(pl.col("close") == max_c).select(["ticker", "ts_utc", "close", "volume"]).head(1).collect()
            price_extremes.extend({"date": day, "kind": "high", **r} for r in topp.to_dicts())
        if min_c is not None and 0 < min_c <= 0.01 and len(price_extremes) < 80:
            topp = lf.filter(pl.col("close") == min_c).select(["ticker", "ts_utc", "close", "volume"]).head(1).collect()
            price_extremes.extend({"date": day, "kind": "low", **r} for r in topp.to_dicts())

        moves = (
            lf.sort(["ticker", "ts_utc"])
            .with_columns((pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias("ret_1m"))
            .filter(pl.col("ret_1m").abs() >= INTRABAR_EXTREME)
            .select(["ticker", "ts_utc", "close", "ret_1m", "volume"])
            .head(3)
            .collect()
        )
        if moves.height and len(move_extremes) < 100:
            move_extremes.extend({"date": day, **r} for r in moves.to_dicts())

        if prev_n_tickers and prev_n_tickers > 0:
            chg = abs(n_tickers - prev_n_tickers) / prev_n_tickers
            if chg > 0.25:
                coverage_jumps.append({"date": day, "tickers": n_tickers, "prev": prev_n_tickers})
        prev_n_tickers = n_tickers
        by_date.append(
            {
                "date": day,
                "rows": n,
                "tickers": n_tickers,
                "duplicate_extra_rows": extra_dup_rows,
                "duplicate_keys": day_dups,
                "invalid_ohlc": day_ohlc,
                "nonpositive_price": day_p,
                "outside_0400_2000_et": int(stats["odd"][0]),
            }
        )
        if i % 25 == 0:
            progress_path.write_text(
                json.dumps({"day": day, "i": i, "n_days": len(expected), "unique_tickers": len(tickers)}, indent=2),
                encoding="utf-8",
            )

    rows_q = _quantiles([r["rows"] for r in by_date])
    tick_q = _quantiles([float(r["tickers"]) for r in by_date])
    med_t = tick_q["median"] or 0
    med_r = rows_q["median"] or 0
    suspicious = [
        r
        for r in by_date
        if (med_t and (r["tickers"] < 0.5 * med_t or r["tickers"] > 1.5 * med_t))
        or (med_r and (r["rows"] < 0.5 * med_r or r["rows"] > 1.5 * med_r))
    ]

    bar_tickers = tickers
    ref_tickers = set(instruments["ticker"].to_list()) if instruments.height else set()
    missing_ref = sorted(bar_tickers - ref_tickers)
    extra_ref = sorted(ref_tickers - bar_tickers)
    figi_n = int(instruments.filter(pl.col("composite_figi").is_not_null()).height) if instruments.height else 0
    asofs = sorted({str(x) for x in instruments["asof_date"].to_list()}) if instruments.height and "asof_date" in instruments.columns else []
    pit_reconstructable = False
    pit_reason = (
        "instruments.parquet is one row per ticker (last snapshot among 6 asof dates). "
        "effective_from/effective_to are null. Daily listing eligibility cannot be reconstructed "
        "as a true PIT calendar from this file."
    )

    categories = {}
    if missing_gz or missing_pq or failed or not expected:
        categories["coverage"] = _cat(
            "FAIL",
            f"Incomplete files missing_gz={len(missing_gz)} missing_pq={len(missing_pq)} failed={len(failed)}",
        )
    elif not by_date:
        categories["coverage"] = _cat("FAIL", "No converted days scanned.")
    else:
        note = (
            f"{len(expected)} converted days {start}..{end}; unique tickers={len(bar_tickers)}. "
            f"{len(weekday_gaps)} weekdays have no file (typically NYSE holidays; not inferred as missing SIP days)."
        )
        st = "WARNING" if suspicious or coverage_jumps else "PASS"
        if suspicious or coverage_jumps:
            note += f" Suspicious coverage days={len(suspicious)}; >25% ticker jumps={len(coverage_jumps)}."
        categories["coverage"] = _cat(st, note)

    if dup_rows == 0:
        categories["uniqueness"] = _cat("PASS", "No extra (ticker, ts_utc) rows.")
    elif dup_rows < 100:
        categories["uniqueness"] = _cat(
            "WARNING",
            f"{dup_rows} extra duplicate (ticker, ts_utc) rows on {len(dup_dates)} dates; not dropped.",
        )
    else:
        categories["uniqueness"] = _cat("FAIL", f"{dup_rows} extra duplicate (ticker, ts_utc) rows; not dropped.")

    if total_rows and ohlc_bad / total_rows > 0.001:
        categories["ohlc"] = _cat("FAIL", f"{ohlc_bad} OHLC violations ({ohlc_bad/total_rows:.4%} of rows).")
    elif ohlc_bad or price_bad or vol_bad or txn_bad:
        categories["ohlc"] = _cat(
            "WARNING",
            f"OHLC violations={ohlc_bad}; nonpositive prices={price_bad}; neg volume={vol_bad}; neg transactions={txn_bad}. Not dropped.",
        )
    else:
        categories["ohlc"] = _cat("PASS", "No OHLC, nonpositive price, negative volume, or negative transaction rows.")

    if session_out / total_rows > 0.01 if total_rows else False:
        categories["timestamps"] = _cat("FAIL", f"{session_out} bars outside 04:00–20:00 ET.")
    elif session_out or off_minute:
        categories["timestamps"] = _cat(
            "WARNING",
            f"{session_out} bars outside 04:00–20:00 ET; {off_minute} not on whole-minute timestamps. DST via America/New_York.",
        )
    else:
        categories["timestamps"] = _cat(
            "PASS",
            f"All bars in 04:00–20:00 ET after DST conversion. Session mix pre={n_pre} rth={n_rth} ah={n_ah}.",
        )

    n_unex = len(overnight_unexplained)
    n_ex = len(overnight_explained)
    if n_unex > 500:
        categories["splits"] = _cat("FAIL", f">{n_unex} sampled unexplained |overnight|>=40% jumps (cap).")
    else:
        categories["splits"] = _cat(
            "WARNING" if (n_unex or n_ex) else "PASS",
            f"As-printed bars; prices not adjusted. Sampled |overnight RTH|>=40%: explained_by_split={n_ex} unexplained={n_unex} (lists capped). Splits in file={splits.height}.",
        )

    categories["extremes"] = _cat(
        "WARNING" if (volume_extremes or price_extremes or move_extremes) else "PASS",
        "Flagged high volume (>=20M), high/low prices, and |1m ret|>=30% examples. None deleted.",
    )

    if not instruments.height:
        categories["reference_pit"] = _cat("FAIL", "instruments.parquet missing or empty.")
    else:
        categories["reference_pit"] = _cat(
            "WARNING",
            pit_reason + f" Bar tickers={len(bar_tickers)}; reference tickers={len(ref_tickers)}; "
            f"bars without reference={len(missing_ref)}; reference without bars={len(extra_ref)}.",
        )

    categories["instrument_identity"] = _cat(
        "WARNING",
        "instrument_id is ticker:<symbol>. composite_figi is present on some reference rows but bars do not use it. "
        "Ticker reuse, changes, and mergers cannot be disambiguated. Safe enough for EXPLORATORY experiment #1 "
        "if results are not treated as issuer-level identity. Not safe to seal an edge.",
    )

    overall = decide_verdict(categories)
    for r in ohlc_examples + price_examples + overnight_explained + overnight_unexplained + volume_extremes + price_extremes + move_extremes:
        for k, v in list(r.items()):
            if hasattr(v, "isoformat"):
                r[k] = str(v)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": DATASET,
        "overall": overall,
        "categories": categories,
        "coverage": {
            "min_ts_utc": str(ts_min) if ts_min else None,
            "max_ts_utc": str(ts_max) if ts_max else None,
            "trading_dates": len(by_date),
            "unique_tickers": len(bar_tickers),
            "tickers_per_day": tick_q,
            "rows_per_day": rows_q,
            "total_rows": total_rows,
            "suspicious_coverage_days": suspicious[:50],
            "coverage_jumps_gt_25pct": coverage_jumps,
            "weekday_dates_without_file": weekday_gaps,
            "missing_gz": missing_gz,
            "missing_parquet": missing_pq,
            "failed_files": failed,
            "accessible_start": start,
            "accessible_end": end,
        },
        "uniqueness": {
            "extra_duplicate_rows": dup_rows,
            "affected_dates": dup_dates,
            "affected_ticker_count": len(dup_tickers),
            "affected_tickers_sample": sorted(dup_tickers)[:50],
        },
        "ohlc": {
            "invalid_ohlc_rows": ohlc_bad,
            "nonpositive_price_rows": price_bad,
            "negative_volume_rows": vol_bad,
            "negative_transaction_rows": txn_bad,
            "ohlc_examples": ohlc_examples[:25],
            "price_examples": price_examples[:25],
        },
        "timestamps": {
            "bars_outside_extended_hours": session_out,
            "off_minute_timestamps": off_minute,
            "premarket_bars": n_pre,
            "rth_bars": n_rth,
            "afterhours_bars": n_ah,
            "timezone": "America/New_York",
        },
        "splits": {
            "split_records": splits.height,
            "overnight_explained_sample": overnight_explained,
            "overnight_unexplained_sample": overnight_unexplained,
            "prices_adjusted": False,
        },
        "extremes": {
            "volume": volume_extremes,
            "price": price_extremes,
            "one_minute_moves": move_extremes,
        },
        "reference_pit": {
            "instrument_rows": instruments.height,
            "asof_dates_present": asofs,
            "figi_nonnull": figi_n,
            "bar_tickers_with_reference": len(bar_tickers & ref_tickers),
            "bar_tickers_without_reference": len(missing_ref),
            "reference_tickers_without_bars": len(extra_ref),
            "missing_reference_sample": missing_ref[:100],
            "pit_reconstructable_daily": pit_reconstructable,
            "pit_reason": pit_reason,
            "active_inactive": instruments.group_by("active").agg(pl.len().alias("n")).to_dicts()
            if instruments.height
            else [],
            "exchange": instruments.group_by("exchange").agg(pl.len().alias("n")).to_dicts()
            if instruments.height
            else [],
        },
        "instrument_identity": {
            "scheme": "ticker:<symbol>",
            "figi_migration": False,
            "safe_for_first_exploratory": True,
            "limitations": [
                "Same ticker may represent different issuers over time.",
                "Ticker changes/mergers/delistings are not mapped as a single instrument.",
                "Universe join uses last available CS snapshot per ticker, not the listing state on that session.",
            ],
        },
    }

    json_path = out / "massive-flat-5y.json"
    md_path = out / "massive-flat-5y.md"
    by_date_path = out / "by_date.parquet"
    pl.DataFrame(by_date).write_parquet(by_date_path)
    if overnight_unexplained:
        pl.DataFrame(overnight_unexplained).write_parquet(out / "overnight_unexplained.parquet")
    if overnight_explained:
        pl.DataFrame(overnight_explained).write_parquet(out / "overnight_explained.parquet")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_md(report, json_path, by_date_path), encoding="utf-8")
    report["report_md"] = str(md_path)
    report["report_json"] = str(json_path)
    report["by_date_parquet"] = str(by_date_path)
    if progress_path.exists():
        progress_path.unlink()
    return report


def _render_md(report: dict[str, Any], json_path: Path, by_date_path: Path) -> str:
    cats = report["categories"]
    cov = report["coverage"]
    lines = [
        "# massive-flat 5-year dataset quality",
        "",
        f"**OVERALL: {report['overall']}**",
        "",
        "This is certification of as-printed Massive US stocks 1-minute Flat Files. "
        "Nothing was deleted or price-adjusted.",
        "",
        "## Category status",
    ]
    for name, cat in cats.items():
        lines.append(f"- **{name.upper()} — {cat['status']}:** {cat['reason']}")
    lines += [
        "",
        "## Coverage",
        f"- Range: `{cov['accessible_start']}` → `{cov['accessible_end']}`",
        f"- Timestamps UTC: `{cov['min_ts_utc']}` → `{cov['max_ts_utc']}`",
        f"- Trading dates scanned: {cov['trading_dates']}",
        f"- Unique tickers (full panel): **{cov['unique_tickers']}**",
        f"- Total rows: {cov['total_rows']}",
        f"- Tickers/day: {cov['tickers_per_day']}",
        f"- Rows/day: {cov['rows_per_day']}",
        f"- Weekdays with no file (likely holidays): {len(cov['weekday_dates_without_file'])}",
        f"- Suspicious coverage days: {len(cov['suspicious_coverage_days'])}",
        "",
        "## Uniqueness",
        f"- Extra duplicate (ticker, ts_utc) rows: {report['uniqueness']['extra_duplicate_rows']}",
        f"- Affected dates: {len(report['uniqueness']['affected_dates'])}",
        "",
        "## OHLC / prices / volume",
        json.dumps({k: report["ohlc"][k] for k in report["ohlc"] if k != "ohlc_examples" and k != "price_examples"}, indent=2),
        "",
        "## Timestamps / sessions",
        json.dumps(report["timestamps"], indent=2),
        "",
        "## Splits / discontinuities",
        "Prices are as-printed. Extreme overnight moves are classified against `corporate_actions.parquet` only.",
        f"- Split records: {report['splits']['split_records']}",
        f"- Explained sample n={len(report['splits']['overnight_explained_sample'])}",
        f"- Unexplained sample n={len(report['splits']['overnight_unexplained_sample'])}",
        "",
        "## Reference / PIT",
        report["reference_pit"]["pit_reason"],
        json.dumps(
            {k: report["reference_pit"][k] for k in report["reference_pit"] if k != "missing_reference_sample" and k != "pit_reason"},
            indent=2,
            default=str,
        ),
        "",
        "## Instrument identity",
        json.dumps(report["instrument_identity"], indent=2),
        "",
        "## Artifacts",
        f"- JSON: `{json_path}`",
        f"- By-date parquet: `{by_date_path}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def validate_converted(root: Path) -> dict[str, Any]:
    return certify_massive_flat(root)
