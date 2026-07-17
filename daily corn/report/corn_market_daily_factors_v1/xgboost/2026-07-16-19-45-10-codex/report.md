# Daily Corn XGBoost Experiment

- Run ID: `2026-07-16-19-45-10-codex`
- Dataset: `corn_market_daily_factors_v1`
- Model: `xgboost 3.3.0`
- Scope: horizon 5, lookback 5, `chronological_811`
- Seeds: `[42, 2024, 3407]`
- RMSE mean/std/worst: `41.031539` / `0.856604` / `42.132936`
- Direction accuracy mean/std/worst: `0.559671` / `0.008890` / `0.547325`

## Signal Thresholds

| Threshold | Metric | Mean | Std | Worst |
| ---: | --- | ---: | ---: | ---: |
| 1.30% | active_signal_rate | 0.262003 | 0.038653 | 0.213992 |
| 1.30% | active_direction_accuracy | 0.525972 | 0.019414 | 0.500000 |
| 1.30% | cumulative_return_0bp | -0.005070 | 0.013101 | -0.018815 |
| 1.30% | sharpe_0bp | -0.085433 | 0.231925 | -0.326548 |
| 1.30% | max_drawdown_0bp | 0.046297 | 0.002891 | 0.049896 |

## Limitations

This is a shadow research backtest. The market-factor dataset has short effective history, extensive missingness, and unverified historical vintages. Results are not a live-trading return promise.
