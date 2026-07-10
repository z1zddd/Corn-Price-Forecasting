# Operator Layers

Reusable intermediate operators for graph-temporal forecasting.

Layers live beside `operator/model` and intentionally stop before final model
training. They fit train-only state such as adjacency matrices, graph filters,
recurrence encoders, visibility statistics, and VMD expansions, then expose a
small `fit` / `transform` / `fit_transform` interface.

Expected window shape is `[n_samples, n_nodes, lookback]`.
