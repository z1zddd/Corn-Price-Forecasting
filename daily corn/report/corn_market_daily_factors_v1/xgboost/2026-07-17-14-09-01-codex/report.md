# Daily Corn XGBoost Experiment

- Run ID: `2026-07-17-14-09-01-codex`
- Dataset: `corn_market_daily_factors_v1`
- Model: `xgboost 3.3.0`
- Scope: horizon 5, lookback 5, `chronological_811`
- Seeds: `[42, 2024, 3407]`
- RMSE mean/std/worst: `41.031539` / `0.856604` / `42.132936`
- Direction accuracy mean/std/worst: `0.559671` / `0.008890` / `0.547325`

## Signal Thresholds

| Threshold | Metric | Mean | Std | Worst |
| ---: | --- | ---: | ---: | ---: |
| 1.90% | active_signal_rate | 0.135802 | 0.037416 | 0.086420 |
| 1.90% | active_direction_accuracy | 0.584275 | 0.011841 | 0.571429 |
| 1.90% | cumulative_return_0bp | 0.010831 | 0.002383 | 0.008100 |
| 1.90% | sharpe_0bp | 0.399041 | 0.134046 | 0.222044 |
| 1.90% | max_drawdown_0bp | 0.023326 | 0.006830 | 0.030400 |

## Limitations

This is a shadow research backtest. The market-factor dataset has short effective history, extensive missingness, and unverified historical vintages. Results are not a live-trading return promise.
