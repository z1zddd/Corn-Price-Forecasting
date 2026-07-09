# Configuration

Each experiment is controlled by a YAML file. Use `configs/corn.yaml` as the corn reference config and `configs/template.yaml` as the starting point for new commodities.

## Top-Level Sections

```yaml
commodity:
data:
target:
lookback:
train_window:
split:
evaluation:
honest_bounds:
models:
```

`commodity` describes the market. It is metadata for reports and future automation.

`data` controls how the CSV is loaded and how features are selected.

`target` controls forward target generation.

`lookback` controls the input sequence length.

`train_window` controls the chronological backtest split.

`split` controls optional validation carving inside each training window.

`evaluation` controls metrics and confidence intervals.

`honest_bounds` records conservative interpretation thresholds.

`models` lists the enabled model candidates.

## Data Section

```yaml
data:
  csv_path: datasets/corn_sample_data.csv
  date_col: date
  date_format: "%Y-%m-%d"
  price_col: close
  feature_cols: auto_numeric
  encoding:
    - utf-8
    - gbk
    - gb18030
  exclude_feature_cols:
    - target_price_fwd
    - target_return_fwd
    - target_direction_fwd
```

`feature_cols: auto_numeric` selects numeric columns after excluding date and target columns. For stricter experiments, replace it with an explicit list.

`date_format` is optional but recommended when the CSV uses compact or ambiguous
date strings such as `16-Jun`. It is passed to `pandas.to_datetime(...,
format=date_format)` so runs are reproducible across pandas/dateutil versions.

## Target Section

```yaml
target:
  horizon: 1
  mode: classification
  spike_threshold: 0.0
```

Targets are derived from `price_col`:

- `target_price_fwd`: future price at `horizon`.
- `target_return_fwd`: future return.
- `target_direction_fwd`: `1` when future return is above `spike_threshold`, otherwise `0`.

The current backtest engine evaluates binary direction classification.

## Lookback Section

```yaml
lookback:
  candidates: [3, 6, 9]
  default: 3
```

`default` is used by `commodity-backtest run`.

`candidates` is used by `commodity-backtest run-lookbacks`.

## Train Window Section

```yaml
train_window:
  mode: expanding
  min_train_periods: 12
  stride_periods: 1
  window_size_periods: null
  max_train_periods: null
  target_known_only: true
```

Supported modes:

- `expanding`: train starts at the first sample and grows through time.
- `rolling`: train uses a fixed `window_size_periods`.
- `expanding_with_cap`: train grows but is capped at `max_train_periods`.

`lookback.default` must be smaller than `min_train_periods`.

`target_known_only: true` prevents forward-label leakage. For a test anchor date
`A`, every training and validation row must have `target_date_fwd <= A`. This is
important for horizons above one period: a row with anchor `t` and target `t+2`
is not available for training until `t+2` is known.

## Split Section

```yaml
split:
  val_ratio: 0.10
```

`val_ratio` removes the final fraction of each training window for validation. Scaling is fit only on the remaining training rows, then applied to train, validation, and test arrays.

## Models Section

```yaml
models:
  - name: last_return
    type: baseline
    enabled: true
  - name: logistic_regression
    type: sklearn_logistic_regression
    enabled: true
  - name: random_forest
    type: sklearn_random_forest
    enabled: true
    params:
      n_estimators: 20
      max_depth: 4
      random_state: 42
```

Every enabled model is fit and evaluated. The framework ranks model rows by `DirAcc`, `ProfitFactor`, and `Sharpe`.

The shorter form is also supported:

```yaml
models:
  - last_return
  - logistic_regression
  - name: random_forest
    params:
      n_estimators: 100
      max_depth: 5
      random_state: 42
```

Optional model names `lightgbm`, `xgboost`, and `catboost` require installing the `tree` or `trees` extra.

Benchmark layer-2 model names are built in:

- `regression_mse_sign`
- `regression_mae_sign`
- `regression_huber_sign`
- `dual_head_mse_bce`

Optional PyTorch model names require installing the `deep` extra:

- `focal_logistic`
- `lstm`
- `gru`
- `transformer`
- `patchtst`
- `itransformer`
- `dlinear`
- `dual_stream_lstm`

`dual_stream_lstm` uses all selected numeric features directly. Columns named like `pca_001` through `pca_032` are routed to the PCA/news branch, while the remaining numeric columns are routed to the structured branch. Use `configs/corn_dual_stream_lstm.yaml` for the corn dataset version that does not depend on random-forest feature ranking.

Keep deep models disabled in default commodity configs unless the target environment has `torch` and enough rows for a meaningful rolling backtest.

## Helper Commands

Use `commodity-backtest auto-window --config configs/corn.yaml` to recommend `lookback` and `train_window` settings from the CSV row count.

Use `commodity-backtest build-config --base-config configs/template.yaml --output configs/my_commodity.yaml --commodity-name my_commodity --csv local_data/my.csv --date-col date --price-col close` to create a new commodity YAML from an existing template.
