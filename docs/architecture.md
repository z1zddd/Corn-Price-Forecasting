# Architecture

Commodity Backtest is organized as a small Python package with clear boundaries between configuration, data preparation, temporal splitting, model construction, evaluation, and reporting.

```text
YAML config
  -> config loader and validation
  -> CSV loader and data diagnosis
  -> forward target generation
  -> lookback window tensor construction
  -> chronological backtest windows
  -> model registry
  -> metrics
  -> report writer and agent verdict
```

## Package Layout

```text

  config/
    loader.py       load YAML files
    schema.py       validate required fields and temporal constraints
  data/
    loader.py       load CSVs with encoding fallback and select features
    targets.py      derive future price, return, and direction targets
    windowing.py    turn tabular rows into lookback windows
    diagnosis.py    summarize dataset health
  backtest/
    splits.py       expanding, rolling, capped expanding windows
    engine.py       config-driven experiment runner
  models/
    base.py         common model protocol
    baseline.py     simple benchmark models
    deep/           optional torch sequence classifiers
    loss_variants.py benchmark layer-2 model adapters
    sklearn_models.py
    registry.py     YAML model factory
  train/
    trainer.py      compact torch training loop
    losses.py       optional torch loss helpers
  eval/
    metrics.py      forecasting, trading, calibration, and CI metrics
  report/
    writer.py       CSV, JSON, Markdown, and chart outputs
    verdict.py      conservative machine-readable result status
  cli.py            command line entry point
```

## Backtest Flow

1. Load and validate the YAML config.
2. Load the CSV and sort by the configured date column.
3. Generate forward targets from `data.price_col` and `target.horizon`.
4. Select features from numeric columns or an explicit feature list.
5. Build lookback windows using `lookback.default`.
6. Build chronological train/test windows using `train_window.mode`.
7. Fit each enabled model separately inside each backtest window.
8. Collect out-of-sample predictions and compute metrics.
9. Rank models by `DirAcc`, `ProfitFactor`, and `Sharpe`.
10. Write reports for the best model and a full model comparison.

## Current Model Types

- `baseline` with `last_return`
- `baseline` with `mean_return` or `mean_direction`
- `sklearn_logistic_regression`
- `sklearn_random_forest`
- benchmark loss variants: `regression_mse_sign`, `regression_mae_sign`, `regression_huber_sign`, and `dual_head_mse_bce`
- optional torch focal classifier: `focal_logistic`
- optional torch sequence classifiers: `lstm`, `gru`, `transformer`, `patchtst`, `itransformer`, `dlinear`, and `dual_stream_lstm`
- optional `lightgbm`, `xgboost`, and `catboost`

New models should be added behind `models/registry.py` so the CLI remains YAML-driven.

Model configs can use either the original typed form:

```yaml
- name: random_forest
  type: sklearn_random_forest
  enabled: true
```

or the shorter zip-compatible form:

```yaml
- last_return
- logistic_regression
- name: random_forest
  params:
    n_estimators: 100
```

## Time-Series Safety

The framework uses chronological windows only. It does not call random train/test splitting for backtests. The generated test point always comes after its training slice.

`split.val_ratio` takes validation rows from the tail of each training window. The sequence standardizer is fit only on the remaining training rows, then reused to transform train, validation, and test windows.

The repository boundary test also prevents common large or generated artifacts from being committed, including model weights, pickled models, compressed archives, Office documents, and experiment output directories.
