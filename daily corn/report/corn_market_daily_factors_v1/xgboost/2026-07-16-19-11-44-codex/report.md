# Daily Corn XGBoost Experiment

- Run ID: `2026-07-16-19-11-44-codex`
- Dataset: `corn_market_daily_factors_v1`
- Model: `xgboost 3.3.0`
- Scope: horizon 5, lookback 5, `chronological_811`
- Seeds: `[42, 2024, 3407]`
- RMSE mean/std/worst: `41.031539` / `0.856604` / `42.132936`
- Direction accuracy mean/std/worst: `0.559671` / `0.008890` / `0.547325`

## Signal Thresholds

| Threshold | Metric | Mean | Std | Worst |
| ---: | --- | ---: | ---: | ---: |
| 3.00% | active_signal_rate | 0.027435 | 0.020530 | 0.000000 |
| 3.00% | active_direction_accuracy | 0.513889 | 0.373195 | 0.000000 |
| 3.00% | cumulative_return_0bp | 0.008749 | 0.007520 | 0.000000 |
| 3.00% | sharpe_0bp | 0.483119 | 0.345238 | 0.000000 |
| 3.00% | max_drawdown_0bp | 0.004210 | 0.004639 | 0.010672 |
| 4.00% | active_signal_rate | 0.000000 | 0.000000 | 0.000000 |
| 4.00% | active_direction_accuracy | 0.000000 | 0.000000 | 0.000000 |
| 4.00% | cumulative_return_0bp | 0.000000 | 0.000000 | 0.000000 |
| 4.00% | sharpe_0bp | 0.000000 | 0.000000 | 0.000000 |
| 4.00% | max_drawdown_0bp | 0.000000 | 0.000000 | 0.000000 |

## Limitations

This is a shadow research backtest. The market-factor dataset has short effective history, extensive missingness, and unverified historical vintages. Results are not a live-trading return promise.
