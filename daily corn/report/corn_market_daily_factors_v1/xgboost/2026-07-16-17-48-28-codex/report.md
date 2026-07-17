# Daily Corn XGBoost Experiment

- Run ID: `2026-07-16-17-48-28-codex`
- Dataset: `corn_market_daily_factors_v1`
- Model: `xgboost 3.3.0`
- Scope: horizon 5, lookback 5, `chronological_811`
- Seeds: `[42, 2024, 3407]`
- RMSE mean/std/worst: `41.031539` / `0.856604` / `42.132936`
- Direction accuracy mean/std/worst: `0.559671` / `0.008890` / `0.547325`

## Limitations

This is a shadow research backtest. The market-factor dataset has short effective history, extensive missingness, and unverified historical vintages. Results are not a live-trading return promise.
