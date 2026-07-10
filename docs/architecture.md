# Architecture

Commodity Backtest is organized as a small Python package with clear
boundaries between dataset materials, operators, configuration, temporal
splitting, model construction, evaluation, and reporting.

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

## Core Layers

The project uses this boundary:

```text
time-series model = operator + datasets
```

- `datasets` are the materials being operated on: corn prices, factor tables,
  news PCA columns, rolling prediction streams, prediction libraries, schemas,
  and source notes.
- `operator` contains methods that act on those materials: model families,
  wrappers, losses, aggregation operators, and future feature operators.
- `pipeline` orchestrates workflow: training, backtesting, evaluation,
  reporting, and scheduling. Pipeline code does not define model families.

Corn CSV fixtures and material documentation live under
`corn_forecast/datasets/corn/`. The old root-level `datasets/` directory has
been folded into that canonical corn material area.

## Package Layout

```text
  corn_forecast/
    cli.py          command line entry point
    config/
      loader.py     load YAML files
      schema.py     validate required fields and temporal constraints
    datasets/
      registry.py   lightweight material registry
      schema.py     lightweight schema declarations
      corn/         corn-specific material docs and schema contracts
    data_processing/
      loader.py     load CSVs with encoding fallback and select features
      targets.py    derive future price, return, and direction targets
      windowing.py  turn tabular rows into lookback windows
      diagnosis.py  summarize dataset health
    operator/
      model/        canonical model operators, families, registry, wrappers, losses, aggregation
    pipeline/
      backtest/
        splits.py   expanding, rolling, capped expanding windows
        engine.py   config-driven experiment runner
      train/        compact torch training and loss helpers
      eval/         forecasting, trading, calibration, and CI metrics
      report/       CSV, JSON, Markdown, chart outputs, and verdicts
    modeling/       legacy compatibility shims for corn_forecast.operator.model
```

## Datasets Material Layer

`corn_forecast/datasets/` contains metadata, schemas, and registry entries for
materials. It is intentionally lightweight and does not import pandas, sklearn,
torch, or other modeling/runtime dependencies.

Corn materials are organized as:

```text
corn_forecast/datasets/corn/
  raw/                 source-like corn CSV fixtures and source notes
  processed/           cleaned monthly modeling tables loaded by configs
  factors/             factor definitions, long-form values, registry, and matrices
  prediction_library/  completed rolling prediction streams
  metadata/            schema and lineage documents
```

Generated experiment outputs, model weights, compressed archives, and private
enterprise data should stay out of `corn_forecast/datasets/`.

## Backtest Flow

1. Load and validate the YAML config.
2. Load the CSV and sort by the configured date column.
3. Generate forward targets from `data.price_col` and `target.horizon`.
4. Select features from numeric columns or an explicit feature list.
5. Build lookback windows using `lookback.default`.
6. Build chronological train/test windows using `train_window.mode`; when
   `train_window.target_known_only` is true, keep only training/validation rows
   whose forward target date is known by the test anchor date.
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

New models should be added behind `corn_forecast/operator/model/registry/` so
the CLI remains YAML-driven. `corn_forecast/modeling/registry/` remains as a
legacy compatibility path.

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

For forward-label tasks, set `train_window.target_known_only: true`. This
guards horizons greater than one period by requiring each train/validation
sample's `target_date_fwd` to be no later than the current test anchor date.

The repository boundary test also prevents common large or generated artifacts from being committed, including model weights, pickled models, compressed archives, Office documents, and experiment output directories.
