# V4R final report (SIGNAL_ONLY)

SEALED OOS NOT OPENED.

RESEARCH PASS: **0**
stop: None
D1 candidates: 4
stable: 4
elapsed_sec: 166.03846168518066

## FEATURE DIAGNOSTIC

See stability stats (path rediscovery, not trading).

## CANDIDATE RULE / STABLE / D2 / D3 / RESEARCH PASS

```json
{
  "d2": [
    {
      "id": "v4_49de2c544d9e186c",
      "rule": "resid_rank_change_5m > 0.4528985507246377 \u2192 LONG",
      "d2": {
        "n": 1140008,
        "mean": 0.00015799739663197742,
        "median": 1.2384288444025367e-05,
        "se": 0.000117256701250135,
        "ticker_days": 224200,
        "trading_days": 247,
        "tickers": 1887,
        "top_ticker_share": 0.0028112223341807863,
        "win_rate": 0.5015499891228834,
        "candidate_id": "v4_49de2c544d9e186c",
        "n_conditions": 1,
        "direction": "LONG",
        "kill": null
      },
      "gates": {
        "sample": true,
        "floor_quarter": false
      },
      "status": "KILL"
    },
    {
      "id": "v4_576296db73c712cb",
      "rule": "resid_rank_change_5m <= 0.4528985507246377 AND resid_rank_change_5m <= -0.45277170600362465 \u2192 SHORT",
      "d2": {
        "n": 1138165,
        "mean": 0.00043697124348605134,
        "median": 0.00015144691204165602,
        "se": 0.00010690308712204352,
        "ticker_days": 223936,
        "trading_days": 247,
        "tickers": 1887,
        "top_ticker_share": 0.002827781169842021,
        "win_rate": 0.5138182952383881,
        "candidate_id": "v4_576296db73c712cb",
        "n_conditions": 2,
        "direction": "SHORT",
        "kill": null
      },
      "gates": {
        "sample": true,
        "floor_quarter": true
      },
      "status": "D2_PASS"
    },
    {
      "id": "v4_584b668627de5948",
      "rule": "resid_rank_change_5m <= 0.4528985507246377 AND resid_rank_change_5m > -0.45277170600362465 AND resid_rank_change_5m > 0.0002839223398037882 \u2192 LONG",
      "d2": {
        "n": 4382020,
        "mean": 5.251354083479461e-05,
        "median": 1.1745828869410635e-06,
        "se": 3.6449286846891864e-05,
        "ticker_days": 246082,
        "trading_days": 247,
        "tickers": 1899,
        "top_ticker_share": 0.0017983756731312856,
        "win_rate": 0.5003270181331897,
        "candidate_id": "v4_584b668627de5948",
        "n_conditions": 3,
        "direction": "LONG",
        "kill": null
      },
      "gates": {
        "sample": true,
        "floor_quarter": false
      },
      "status": "KILL"
    },
    {
      "id": "v4_e173c6e0f6bc49a1",
      "rule": "resid_rank_change_5m <= 0.4528985507246377 AND resid_rank_change_5m > -0.45277170600362465 AND resid_rank_change_5m <= 0.0002839223398037882 \u2192 SHORT",
      "d2": {
        "n": 4387063,
        "mean": 0.00017203118502780263,
        "median": 4.836150380214e-05,
        "se": 8.182259435578013e-05,
        "ticker_days": 246109,
        "trading_days": 247,
        "tickers": 1898,
        "top_ticker_share": 0.001807897947831432,
        "win_rate": 0.5079439251271295,
        "candidate_id": "v4_e173c6e0f6bc49a1",
        "n_conditions": 3,
        "direction": "SHORT",
        "kill": null
      },
      "gates": {
        "sample": true,
        "floor_quarter": false
      },
      "status": "KILL"
    }
  ],
  "d3": [
    {
      "id": "v4_576296db73c712cb",
      "rule": "resid_rank_change_5m <= 0.4528985507246377 AND resid_rank_change_5m <= -0.45277170600362465 \u2192 SHORT",
      "d2": {
        "n": 1138165,
        "mean": 0.00043697124348605134,
        "median": 0.00015144691204165602,
        "se": 0.00010690308712204352,
        "ticker_days": 223936,
        "trading_days": 247,
        "tickers": 1887,
        "top_ticker_share": 0.002827781169842021,
        "win_rate": 0.5138182952383881,
        "candidate_id": "v4_576296db73c712cb",
        "n_conditions": 2,
        "direction": "SHORT",
        "kill": null
      },
      "d3": {
        "n": 2449711,
        "mean": 0.00018563719305823379,
        "median": 0.0001650575943452008,
        "se": 9.992770093105228e-06,
        "ticker_days": 469146,
        "trading_days": 494,
        "tickers": 2665,
        "top_ticker_share": 0.0024448907885211424,
        "win_rate": 0.5129041752272002,
        "candidate_id": "v4_576296db73c712cb",
        "n_conditions": 2,
        "direction": "SHORT",
        "kill": null
      },
      "decision": "VALIDATED_SUBTHRESHOLD_PHENOMENON",
      "note": "SIGNAL_ONLY"
    }
  ],
  "survivors": [],
  "subthreshold": [
    {
      "id": "v4_576296db73c712cb",
      "rule": "resid_rank_change_5m <= 0.4528985507246377 AND resid_rank_change_5m <= -0.45277170600362465 \u2192 SHORT",
      "d2": {
        "n": 1138165,
        "mean": 0.00043697124348605134,
        "median": 0.00015144691204165602,
        "se": 0.00010690308712204352,
        "ticker_days": 223936,
        "trading_days": 247,
        "tickers": 1887,
        "top_ticker_share": 0.002827781169842021,
        "win_rate": 0.5138182952383881,
        "candidate_id": "v4_576296db73c712cb",
        "n_conditions": 2,
        "direction": "SHORT",
        "kill": null
      },
      "d3": {
        "n": 2449711,
        "mean": 0.00018563719305823379,
        "median": 0.0001650575943452008,
        "se": 9.992770093105228e-06,
        "ticker_days": 469146,
        "trading_days": 494,
        "tickers": 2665,
        "top_ticker_share": 0.0024448907885211424,
        "win_rate": 0.5129041752272002,
        "candidate_id": "v4_576296db73c712cb",
        "n_conditions": 2,
        "direction": "SHORT",
        "kill": null
      },
      "decision": "VALIDATED_SUBTHRESHOLD_PHENOMENON",
      "note": "SIGNAL_ONLY"
    }
  ],
  "stop": null,
  "linear": {
    "lambda": 0.1,
    "d3_decile_spread": 0.0008589111573882337,
    "n_features": 9,
    "scaler": "train_only",
    "reused_checkpoint": true
  }
}
```

V4R: ZERO VALIDATED DIRECTIONAL RULES

Did V4R discover a concrete directional relationship that could realistically deserve further execution research?

PARTIAL — validated phenomenon but not trading-edge candidate