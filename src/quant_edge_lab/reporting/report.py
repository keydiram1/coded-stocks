from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from quant_edge_lab.models.hypothesis import HypothesisSpec
from quant_edge_lab.models.schemas import ExperimentRecord


FAKE_REASONS = [
    "Synthetic sample data is not market evidence; any mean return is a generator artifact.",
    "Minute OHLC cannot prove fills; all results are SIGNAL_ONLY.",
    "Events clustered on a few symbol-days can dominate the mean; N_raw is not the effective sample size.",
    "Multiple horizons were tested; uncorrected p-values overstate significance.",
    "Same-bar high/low sequencing is unresolved when ambiguous=true.",
    "Costs, spread, borrow, and impact are not subtracted.",
    "Eligibility used only previous close; real research also needs PIT shares/cap and delisted names.",
]


def _status(summary: dict) -> str:
    # Never PROMISING on synthetic/sample exploratory data.
    if summary.get("labels", {}).get("sample_is_research_evidence") is False:
        return "CONTINUE"
    return "CONTINUE"


def write_charts(outcomes: pl.DataFrame, chart_dir: Path) -> list[str]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if outcomes.height == 0:
        return paths
    primary = outcomes.filter(pl.col("horizon") == outcomes["horizon"][0])
    rets = primary["forward_return"].drop_nulls().to_list()
    if rets:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rets, bins=min(30, max(5, len(rets))), color="#334155", edgecolor="white")
        ax.set_title("Forward return distribution (first horizon, SIGNAL_ONLY)")
        ax.set_xlabel("return")
        p = chart_dir / "return_distribution.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))
    if "trading_day" in primary.columns:
        by = primary.group_by("trading_day").agg(pl.col("forward_return").mean().alias("mean_ret"))
        if by.height > 60:
            by = by.with_columns(pl.col("trading_day").cast(pl.Utf8).str.slice(0, 7).alias("ym")).group_by("ym").agg(
                pl.col("mean_ret").mean()
            )
            labels = by["ym"].to_list()
            title = "Mean SIGNAL_ONLY return by month"
        else:
            labels = [str(x) for x in by["trading_day"].to_list()]
            title = "Mean SIGNAL_ONLY return by day"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, by["mean_ret"].to_list(), color="#1e293b")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        p = chart_dir / "by_day.png"
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(str(p))
    return paths


def render_report(
    spec: HypothesisSpec,
    record: ExperimentRecord,
    summary: dict,
    chart_paths: list[str],
    path: Path,
) -> Path:
    status = _status(summary)
    lines = [
        f"# Experiment {record.experiment_id}",
        "",
        f"**STATUS: {status}**  ",
        "This run is EXPLORATORY / SIGNAL_ONLY. Smoke and sample datasets are not evidence of an edge.",
        "",
        "## SUMMARY",
        f"- Hypothesis: `{spec.id}` family `{spec.family}`",
        f"- Stage: {record.stage}",
        f"- Raw events: {summary.get('n_raw_events')}",
        f"- Symbol-days: {summary.get('n_symbol_days')}",
        f"- Symbols: {summary.get('n_symbols')}",
        f"- Trading days: {summary.get('n_trading_days')}",
        f"- Variants counted this run (horizons as tests): {record.number_of_variants_tested}",
        "",
        "## HYPOTHESIS",
        spec.description.strip(),
        f"- Side: {spec.side}",
        f"- Trigger window ET: {spec.trigger.time_et.start}–{spec.trigger.time_et.end} ({spec.trigger.session})",
        "- Conditions:",
    ]
    for c in spec.trigger.conditions:
        lines.append(f"  - `{c.feature} {c.op} {c.value}`")
    lines += [
        "",
        "## DATA",
        f"- Dataset: `{record.dataset}` source=`{record.data_source}`",
        f"- Manifest hash: `{record.data_manifest_hash}`",
        f"- Bar range: {summary.get('labels', {}).get('bar_ts_min')} → {summary.get('labels', {}).get('bar_ts_max')}",
        f"- Instruments in file: {summary.get('labels', {}).get('n_instruments')}",
        "- Volume stats are VOLUME LIQUIDITY PROXIES, not executable liquidity (no NBBO/size).",
        (
            "- Source: sample synthetic bars (not research evidence)"
            if summary.get("labels", {}).get("is_sample")
            else (
                "- Source: real Massive 1-minute aggregates (unadjusted). "
                + (
                    "SMOKE SAMPLE — not evidence of an edge."
                    if summary.get("labels", {}).get("is_smoke")
                    else "EXPLORATORY historical data — not a validated strategy."
                )
            )
        ),
        "",
        "## UNIVERSE",
        f"- Exchanges: {spec.universe.exchanges}",
        f"- Security type: {spec.universe.security_type}",
        f"- Prev close: {spec.universe.price_prev_close.min}–{spec.universe.price_prev_close.max}",
        "",
        "## EVENT DEFINITION",
        "An event is an observation that conditions held at decision_ts. It is not a trade.",
        "",
        "## AVAILABILITY / CAUSALITY ASSUMPTIONS",
        "- Features on bar `ts` (bar start) include that bar and are available at `ts + 1 minute`.",
        "- Entry uses next-bar open = open of the bar starting at decision_ts.",
        "- Previous close is last RTH close of the prior session (PIT).",
        "",
        "## EXECUTION ASSUMPTIONS",
        "- Model: next_bar_open / SIGNAL_ONLY",
        "- Minute high/low are not fills.",
        "- Unlimited liquidity is NOT assumed.",
        "",
        "## SAMPLE SIZE",
        json.dumps(
            {
                "n_raw_events": summary.get("n_raw_events"),
                "n_symbol_days": summary.get("n_symbol_days"),
                "n_symbols": summary.get("n_symbols"),
                "n_trading_days": summary.get("n_trading_days"),
            },
            indent=2,
        ),
        "",
        "## RESULT DISTRIBUTION / FORWARD RETURNS / MFE / MAE / CI",
    ]
    for h in summary.get("horizons", []):
        lines.append(f"### Horizon {h['horizon']}")
        lines.append("```json")
        lines.append(json.dumps(h, indent=2, default=str))
        lines.append("```")
    lines += [
        "",
        "## MULTIPLE-TESTING STATUS",
        "Benjamini-Hochberg across horizons in this run. Other threshold variants must be counted in the family trial log.",
        "```json",
        json.dumps(summary.get("benjamini_hochberg"), indent=2),
        "```",
        "",
        "## CHRONOLOGICAL BREAKDOWN",
        "```json",
        json.dumps(summary.get("chronological"), indent=2),
        "```",
        "",
        "## SYMBOL BREAKDOWN / CONCENTRATION",
        "```json",
        json.dumps(summary.get("by_symbol"), indent=2),
        "```",
        "",
        "## AMBIGUOUS EVENTS",
        "If take-profit and stop would both be touched in the same minute, `ambiguous=true` and we do not pick a winner.",
        "",
        "## RAW STATISTICAL EFFECT vs EXECUTABLE EDGE",
        "- RAW STATISTICAL EFFECT: distribution of SIGNAL_ONLY next-bar-open to horizon close returns on events.",
        "- EXECUTABLE EDGE: **not measured**. No historical NBBO, size, spread, or slippage in this dataset.",
        "- Do not treat a positive mean as a tradable profit after costs.",
        "",
        "## ROBUSTNESS / CONCENTRATION",
        "```json",
        json.dumps(summary.get("robustness"), indent=2, default=str),
        "```",
        "",
        "## HOW THIS RESULT COULD BE FAKE",
    ]
    reasons = spec.falsification or FAKE_REASONS
    for r in reasons:
        lines.append(f"- {r}")
    for r in FAKE_REASONS:
        if r not in reasons:
            lines.append(f"- {r}")
    for r in summary.get("how_this_could_be_fake") or []:
        lines.append(f"- {r}")
    lines += [
        "",
        "## NEXT FALSIFICATION EXPERIMENT",
        "Shift entry one extra bar later; if the mean disappears, the 'edge' was same-bar artifact.",
        "On real data: pull NBBO around events and assume fill at ask (long) or bid (short) or worse.",
        "",
        f"## STATUS: {status}",
        "",
        "## CHARTS",
    ]
    for p in chart_paths:
        lines.append(f"- `{p}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = "<html><body><pre>" + path.read_text(encoding="utf-8").replace("<", "&lt;") + "</pre>"
    for p in chart_paths:
        html += f'<p><img src="{Path(p).as_posix()}" /></p>'
    html += "</body></html>"
    path.with_suffix(".html").write_text(html, encoding="utf-8")
    return path
