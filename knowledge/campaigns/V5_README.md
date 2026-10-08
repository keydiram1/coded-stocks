# V5 mechanism-first

V4/V4R science is closed (0 RESEARCH_PASS). Do not treat leftover V4 features as
V5 hypotheses.

V5 unit of research: mechanism → expected → observed → discrepancy →
directional resolution → falsification.

YAML in this directory is a **preregistered proposal**. Thresholds and the
campaign-specific 12bp floor require reviewer approval before `--execute`.

`CLOSE_DISLOCATION_REVERSAL_V1` (`forced_eod_preclose_flow`) tests whether
abnormal close-window volume plus idiosyncratic final-5m displacement vs a
causal LOO market expectation reverses after the close.

This V1 tests a pre-close/end-of-day forced-flow dislocation. It does **not**
claim to isolate the official closing auction. Auction-specific research
requires verified auction/official-close data. It is not claimed to be an edge.

Options surface / local relative value is intentionally not coded here.

`instruments.parquet` is not full session-dated listing history. That limitation
is accepted for this first SIGNAL_ONLY campaign and is hashed into run identity.
Any V5 survivor must be replicated with proper session-dated listing history
before being considered trading-grade evidence.
