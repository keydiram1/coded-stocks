"""Outcomes from minute bars.

Same-bar high and low that both touch take-profit and stop is AMBIGUOUS.
Never choose the profitable ordering. Minute results are SIGNAL_ONLY.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from quant_edge_lab.models.hypothesis import HypothesisSpec

OUTCOME_SCHEMA = {
    "event_id": pl.String,
    "horizon": pl.String,
    "forward_return": pl.Float64,
    "mfe": pl.Float64,
    "mae": pl.Float64,
    "target_first": pl.Boolean,
    "stop_first": pl.Boolean,
    "ambiguous": pl.Boolean,
    "entry_price": pl.Float64,
    "execution_status": pl.String,
    "symbol_day": pl.String,
    "trading_day": pl.Date,
    "instrument_id": pl.String,
    "ticker": pl.String,
}


def parse_horizon_minutes(label: str) -> int:
    label = label.strip().lower()
    if label.endswith("m"):
        return int(label[:-1])
    if label.endswith("h"):
        return int(label[:-1]) * 60
    raise ValueError(f"Unsupported horizon '{label}'")


def _scan_barriers(
    windows: list[tuple[float, float, float]],
    entry: float,
    side: str,
    tp: float,
    sl: float,
) -> tuple[bool, bool | None, bool | None, float | None, float | None]:
    """windows: list of (open, high, low) in time order including entry bar."""
    if side == "long":
        tp_px = entry * (1 + tp)
        sl_px = entry * (1 + sl)
    else:
        tp_px = entry * (1 - tp)
        sl_px = entry * (1 - sl)

    mfe = 0.0
    mae = 0.0
    for _o, high, low in windows:
        if side == "long":
            mfe = max(mfe, high / entry - 1)
            mae = min(mae, low / entry - 1)
            hit_tp = high >= tp_px
            hit_sl = low <= sl_px
        else:
            mfe = max(mfe, 1 - low / entry)
            mae = min(mae, 1 - high / entry)
            hit_tp = low <= tp_px
            hit_sl = high >= sl_px
        if hit_tp and hit_sl:
            return True, None, None, mfe, mae
        if hit_tp:
            return False, True, False, mfe, mae
        if hit_sl:
            return False, False, True, mfe, mae
    return False, None, None, mfe, mae


def compute_outcomes(
    events: pl.DataFrame,
    bars: pl.DataFrame,
    spec: HypothesisSpec,
) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame(schema=OUTCOME_SCHEMA)

    bars_i = {
        iid: grp.sort("ts_utc")
        for iid, grp in bars.partition_by("instrument_id", as_dict=True).items()
    }
    # partition_by as_dict keys may be tuples
    normalized: dict[str, pl.DataFrame] = {}
    for k, v in bars_i.items():
        key = k[0] if isinstance(k, tuple) else k
        normalized[key] = v
    bars_i = normalized

    horizons = spec.outcomes.forward_returns
    horizon_mins = {h: parse_horizon_minutes(h) for h in horizons}
    max_h = max(horizon_mins.values())
    tp = spec.outcomes.barriers.take_profit
    sl = spec.outcomes.barriers.stop_loss
    side = spec.side

    rows: list[dict] = []
    for ev in events.iter_rows(named=True):
        bdf = bars_i.get(ev["instrument_id"])
        if bdf is None:
            continue
        entry_ts = ev["entry_ts"]
        future = bdf.filter(pl.col("ts_utc") >= entry_ts)
        if future.height == 0:
            continue
        entry_price = float(future["open"][0])
        end_ts = entry_ts + timedelta(minutes=max_h)
        window = future.filter(pl.col("ts_utc") < end_ts)
        ohl = list(zip(window["open"].to_list(), window["high"].to_list(), window["low"].to_list()))
        ambiguous, target_first, stop_first, mfe, mae = _scan_barriers(ohl, entry_price, side, tp, sl)

        for label, mins in horizon_mins.items():
            tgt = entry_ts + timedelta(minutes=mins - 1)
            match = future.filter(pl.col("ts_utc") == tgt)
            fwd = None
            if match.height:
                close_px = float(match["close"][0])
                if side == "long":
                    fwd = close_px / entry_price - 1
                else:
                    fwd = 1 - close_px / entry_price
            # MFE/MAE over the same horizon window
            h_end = entry_ts + timedelta(minutes=mins)
            h_win = future.filter(pl.col("ts_utc") < h_end)
            h_ohl = list(zip(h_win["open"].to_list(), h_win["high"].to_list(), h_win["low"].to_list()))
            amb_h, tf_h, sf_h, mfe_h, mae_h = _scan_barriers(h_ohl, entry_price, side, tp, sl)
            rows.append(
                {
                    "event_id": ev["event_id"],
                    "horizon": label,
                    "forward_return": fwd,
                    "mfe": mfe_h,
                    "mae": mae_h,
                    "target_first": tf_h,
                    "stop_first": sf_h,
                    "ambiguous": amb_h,
                    "entry_price": entry_price,
                    "execution_status": spec.execution.status,
                    "symbol_day": f"{ev['ticker']}|{ev['session_date']}",
                    "trading_day": ev["session_date"],
                    "instrument_id": ev["instrument_id"],
                    "ticker": ev["ticker"],
                }
            )
    if not rows:
        return pl.DataFrame(schema=OUTCOME_SCHEMA)
    return pl.DataFrame(rows, schema=OUTCOME_SCHEMA, infer_schema_length=0)
