# Daily Corn XGBoost Experiment

- Run ID: `2026-07-16-18-43-47-codex`
- Dataset: `corn_market_daily_factors_v1`
- Model: `xgboost 3.3.0`
- Scope: horizon 5, lookback 5, `chronological_811`
- Seeds: `[42, 2024, 3407]`
- RMSE mean/std/worst: `41.031539` / `0.856604` / `42.132936`
- Direction accuracy mean/std/worst: `0.559671` / `0.008890` / `0.547325`

## Signal Thresholds

| Threshold | Metric | Mean | Std | Worst |
| ---: | --- | ---: | ---: | ---: |
| 1.00% | active_signal_rate | 0.360768 | 0.018506 | 0.345679 |
| 1.00% | active_direction_accuracy | 0.528988 | 0.041584 | 0.482353 |
| 1.00% | cumulative_return_0bp | -0.002105 | 0.029083 | -0.043155 |
| 1.00% | sharpe_0bp | -0.037700 | 0.450587 | -0.674864 |
| 1.00% | max_drawdown_0bp | 0.057753 | 0.009098 | 0.070616 |
| 2.00% | active_signal_rate | 0.120713 | 0.036859 | 0.069959 |
| 2.00% | active_direction_accuracy | 0.621525 | 0.063513 | 0.552632 |
| 2.00% | cumulative_return_0bp | 0.014754 | 0.006682 | 0.005952 |
| 2.00% | sharpe_0bp | 0.617408 | 0.333417 | 0.204436 |
| 2.00% | max_drawdown_0bp | 0.019856 | 0.008309 | 0.028692 |

## Limitations

This is a shadow research backtest. The market-factor dataset has short effective history, extensive missingness, and unverified historical vintages. Results are not a live-trading return promise.
