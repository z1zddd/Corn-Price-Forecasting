# Official 57-Model Pool

The framework includes the corn benchmark model pool as `models: official_57`.
It expands to the 57 model names completed in the long-lookback corn spike run:

- sklearn/tabular ML: linear, SVM, neighbors, tree, forest, boosting, bagging, MLP, Gaussian process, and Naive Bayes variants
- package-native gradient boosting: LightGBM, XGBoost, CatBoost
- aeon time-series models: MiniRocket, MultiRocket, interval forests, Catch22/Summary, RDST, KNN-DTW/Euclidean, and aeon deep classifiers/regressors
- Keras sequence models: LSTM, GRU, BiLSTM, and TCN variants built from official TensorFlow/Keras layers

The registry expands the pool through `models.official_pool.expand_model_pool`.
Each entry uses `type: official_pool`, which creates an adapter around the official package estimator classes. Optional dependencies are loaded only when a model is fitted.

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
