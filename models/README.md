# Model Layer Layout

`models/` is now reserved for model definitions and model-pool specifications.
Cross-cutting concerns live outside this package:

- `registry/`: YAML model-name resolution and factory dispatch.
- `ensembles/`: ensemble and deployment-combination logic.
- `losses/`: loss-oriented model variants and output wrappers.
- `wrappers/`: shared adapters around third-party runtimes.

## Subpackages

- `baselines/`: simple benchmark models such as last-return and majority direction.
- `classical/`: tabular classical ML adapters and factories for sklearn-style models.
- `sequence/`: sequence neural models such as LSTM, GRU, TCN-style, Transformer, PatchTST-style, iTransformer-style, DLinear-style, and dual-stream LSTM.
- `specs/`: model-pool specifications such as the official 57-model pool.

Compatibility wrappers remain at the old paths (`models.registry`,
`models.official_pool`, `models.deployment_ensemble`, `models.loss_variants`,
`models.sklearn_models`, `models.baseline`, and `models.deep.*`) so existing
scripts and tests continue to run while new code can import from the clearer
packages.
