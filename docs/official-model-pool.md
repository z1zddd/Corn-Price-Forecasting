# Official And Deployment Model Pools

The framework includes the corn benchmark model pool as `models: official_57`.
It expands to the 57 model names completed in the long-lookback corn spike run:

- sklearn/tabular ML: linear, SVM, neighbors, tree, forest, boosting, bagging, MLP, Gaussian process, and Naive Bayes variants
- package-native gradient boosting: LightGBM, XGBoost, CatBoost
- aeon time-series models: MiniRocket, MultiRocket, interval forests, Catch22/Summary, RDST, KNN-DTW/Euclidean, and aeon deep classifiers/regressors
- Keras sequence models: LSTM, GRU, BiLSTM, and TCN variants built from official TensorFlow/Keras layers

The registry expands the pool through `models.specs.official.expand_model_pool`.
Each entry uses `type: official_pool`, which creates an adapter around the official package estimator classes. Optional dependencies are loaded only when a model is fitted.

The implementation is split under `models/specs/official/`:

- `tabular/`: sklearn, LightGBM, XGBoost, and CatBoost specs grouped by model family
- `aeon/`: one file per complex aeon method such as MiniRocket, RDST, and InceptionTime
- `keras/`: one file per Keras sequence method such as LSTM, GRU, BiLSTM, and TCN
- `adapter.py`, `pool.py`, `base.py`, and `io.py`: shared adapter, pool assembly, declarations, and input conversion

## Best Deployment Ensemble Pool

The framework also registers the fixed best deployment ensembles selected from
the completed rolling prediction library:

```yaml
models:
  pool: best_deployment_forward_ensembles
```

This expands to:

- `corn_h1_forward_replacement_ap_hard_vote`
- `corn_h2_forward_replacement_ba_hard_vote`

These entries use `type: deployment_ensemble`. They are not raw-window
estimators; they consume completed rolling prediction streams from the 57-model
pool and aggregate them with the fixed forward-replacement weights selected in
deployment discovery.

For convenience, the combined pool is also available:

```yaml
models:
  pool: official_57_plus_best_deployment
```

Use `scripts/run_best_deployment_combo.py` to evaluate the fixed H1/H2
deployment ensembles from `all_rolling_predictions.csv`.

## Corn Configs

Four benchmark configs are provided:

- `configs/corn_official_pool_57_h1_no_news.yaml`
- `configs/corn_official_pool_57_h2_no_news.yaml`
- `configs/corn_official_pool_57_h1_with_news.yaml`
- `configs/corn_official_pool_57_h2_with_news.yaml`

They use:

- horizon 1 or 2
- lookback candidates 6, 9, and 12
- expanding rolling tests with `target_known_only: true`
- fixed 0.5 classification threshold inside the framework
- `exclude_feature_patterns: [pca_*, PCA*]` for no-news runs
- precomputed PCA/news columns retained for with-news runs

Run a single lookback:

```bash
commodity-backtest run --config configs/corn_official_pool_57_h1_no_news.yaml --output-dir experiments/corn_official_57_h1_no_news_lb6
```

Run the lookback sweep:

```bash
commodity-backtest run-lookbacks --config configs/corn_official_pool_57_h1_no_news.yaml --output-dir experiments/corn_official_57_h1_no_news
```

Install optional dependencies for the full pool:

```bash
pip install -e ".[official-pool]"
```

Without those extras, configs still validate and sklearn-only entries run, but aeon/LightGBM/XGBoost/CatBoost/Keras entries will fail at fit time with clear import errors.
