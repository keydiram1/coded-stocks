# MASTER RESEARCH CONTEXT

This file is for future engineers and coding agents. It records **why** the
repository exists, not just how to run it.

## Mission

Build a reproducible scientific experimentation platform that searches for,
measures, attacks, falsifies, validates, and later *observes* possible
short-horizon market inefficiencies in US equities.

We are not building a stock picker, TA dashboard, trading bot, generic
backtester-as-optimizer, or a system whose purpose is to maximize backtest P&amp;L.

Core principle: the next experiment should be the cheapest experiment capable of
proving that an apparent edge is not real.

## Labels

| Label | Meaning |
| --- | --- |
| FACT | Directly established |
| INFERENCE | Reasonable conclusion from evidence |
| HYPOTHESIS | Testable claim |
| RESULT | Empirical output of *our* data/code |
| UNKNOWN | Not established |

Do not promote HYPOTHESIS to FACT in docs or reports.

## Inspiration (not a strategy)

Anecdotes about an individual programmer/trader motivated research *directions*:
high turnover, small/micro-cap, low capacity, capital recycling, spread,
liquidity, shorts, events, microstructure, mechanical/forced flow.

FACT: we do not know that person's algorithm.
HYPOTHESIS: some opportunities for a small independent trader may sit below the
capacity of large firms.
UNKNOWN: whether any such edge exists in our data.

Do not reverse-engineer a secret strategy.

## Two families

1. Informational / statistical price edges (gap, RVOL, fade, news, etc.).
2. Structural / mechanical edges (auctions, LULD, forced flow, predictable algo
   responses). Both are first-class *research questions*, not assumed facts.

Monetization may later be long, short, options, pairs, ETFs, futures, vol, or
event contracts. Discover the **phenomenon** first. A predictable decline is
information even if borrow later makes a short impossible.

## Progressive resolution

Minute discovery → targeted trade/quote validation → L2/L3 only when justified →
live observation → tiny real execution calibration.

Minute OHLC cannot prove path. If one minute prints +5% and −3%, sequencing is
**AMBIGUOUS** until higher resolution exists. Never infer the favorable order.

## Capacity and recycling

Capacity is first-class: eventual `EV_net(q)` after markout, spread, slippage,
fees, impact, borrow. Do not invent sophisticated impact models from minute bars.

Do not confuse gross notional traded with required capital. Short holding periods
can recycle a small account. That is a measurement problem for later phases.

## Statistical standards

- Do not treat events as independent.
- Cluster by symbol-day / trading day.
- Block bootstrap by trading day.
- Count every variant (threshold, horizon, universe, feature definition).
- Benjamini–Hochberg in this slice; Romano-Wolf, White RC, DSR, SPA, PBO later.
- Reports must include **HOW THIS RESULT COULD BE FAKE**.
- Status: KILL / CONTINUE / PROMISING. PROMISING means “deserves a more expensive
  test,” never “proven strategy.” Sample data must not be PROMISING.

## Lifecycle

EXPLORATORY → VALIDATION → SEALED_OOS → LIVE_OBSERVATION → TINY_LIVE →
EXECUTION_VALIDATED → CAPACITY_VALIDATED.

Candidates must not quietly retune on sealed OOS. This slice is EXPLORATORY only.

## First real-data roadmap (not this build)

A historical US-equity minutes → B PIT instrument history → C corporate actions →
D exploratory event studies → E candidates → F targeted trades/NBBO around events →
G try to destroy the signal with realistic execution → H freeze rule → I sealed OOS →
J live no-trade observer → K tiny live → L empirical capacity curve.

## AI's role

May propose hypotheses, adversarial explanations, and experiment configs.
Must not choose favorable fills, hide failed variants, or declare profitability
from a pretty equity curve. The deterministic engine is the authority.

## What this repo implemented in v0.1

A working local vertical slice on **synthetic** data with the scientific labels
above encoded in code and tests. See README for commands and deferrals.
