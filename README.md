# Quant Edge Lab

Local scientific engine for **discovering and falsifying** short-horizon statistical
edges in US equities. It is **not** a trading bot, broker client, or P&amp;L optimizer.

Most candidate hypotheses should fail. A correct rejection is a successful experiment.

## What this slice does

Synthetic minute bars → YAML hypothesis → point-in-time features → events →
outcomes (including **ambiguous** OHLC barrier sequencing) → clustered bootstrap +
Benjamini–Hochberg → hashed experiment registry → Markdown/HTML report →
deterministic reproduce.

Minute results are labeled **SIGNAL_ONLY**. They do not prove executable P&amp;L.

Sample data is **not research evidence**. Reports will not be labeled PROMISING.

## Setup (Python 3.12+)

```bash
cd quant-edge-lab
python -m venv .venv
# Windows:
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Commands

```bash
quant-edge init-sample-data
quant-edge validate-hypothesis hypotheses/gap_rvol_continuation_v1.yaml
quant-edge run hypotheses/gap_rvol_continuation_v1.yaml
quant-edge list-experiments
quant-edge show-experiment <experiment_id>
quant-edge reproduce <experiment_id>
```

Equivalent module form: `python -m quant_edge_lab <command>`.

Massive (real data). Put `MASSIVE_API_KEY` in project-root `.env` (gitignored). Never commit it.
Set `QUANT_EDGE_DATA_ROOT=D:\quant-edge-lab-data` so Parquet and ingest checkpoints go to D:, not C:.

```bash
python -m quant_edge_lab massive check-access
python -m quant_edge_lab massive download --dataset massive-smoke
python -m quant_edge_lab massive status --dataset massive-smoke
python -m quant_edge_lab run hypotheses/gap_rvol_continuation_v1.yaml --dataset massive-smoke
python -m quant_edge_lab massive download --dataset massive-pilot --start 2026-09-02 --end 2026-09-30 --max-tickers 40
python -m quant_edge_lab massive download --dataset massive-full --start 2024-10-02 --end 2026-10-01
python -m quant_edge_lab massive status --dataset massive-full
python -m quant_edge_lab run hypotheses/gap_rvol_continuation_v1.yaml --dataset massive-full
```

`--dataset massive` uses full, else pilot, else smoke if present.

Minute history older than ~2 years may 403 on this plan. Unadjusted bars (`adjusted=false`). Grouped-daily may 429 under rate limits.

```bash
python -m pytest
```

## Architecture

Raw-ish Parquet (immutable sample) → features (bar close / `available_at`) →
events (conditions held) → next-bar-open discovery model → outcomes → statistics
→ `experiments/<id>/`.

DuckDB is a declared dependency for later large scans; this slice uses Polars on
Parquet. `run --dataset sample|massive-smoke|massive-pilot|massive-full` uses the
same feature/event/outcome engine. Massive bars are stored unadjusted.

## Scientific rules (non-negotiable)

- Ticker is not identity; `instrument_id` is.
- Do not use current float/market cap on historical events.
- Completed bar features are not known at that bar's open.
- Same-minute take-profit and stop → `ambiguous=true` (no favorable ordering).
- Events are clustered; they are not i.i.d.
- Every threshold/horizon variant is a trial.
- AI must not silently tune sealed OOS (no sealed OOS in this slice).

See `MASTER_RESEARCH_CONTEXT.md` for why the repo looks like this.

## Implemented vs deferred

**Implemented:** sample generator, Massive 1m ingest (resumable), dataset switch
on `run`, three YAML families, PIT features, event/outcome engines, BH, day-block
bootstrap, registry, reports, leakage unit tests, CLI, reproduce.

**Deferred:** tick/NBBO execution, matched controls, capacity `EV(q)`, live
sockets, options, borrow, Romano-Wolf/SPA/DSR, parameter search.
