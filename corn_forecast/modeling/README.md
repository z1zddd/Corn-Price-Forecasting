# Model Layer Layout

`corn_forecast/modeling/` contains model definitions, model-pool
specifications, and model-adjacent infrastructure.

## Infrastructure

- `registry/`: YAML model-name resolution and factory dispatch.
- `ensembles/`: ensemble and deployment-combination logic.
- `losses/`: loss-oriented model variants and output wrappers.
- `wrappers/`: shared adapters around third-party runtimes.

## Subpackages

- `baselines/`: simple benchmark models such as last-return and majority direction.
- `classical/`: tabular classical ML adapters and factories for sklearn-style models.
- `sequence/`: sequence neural models such as LSTM, GRU, TCN-style, Transformer, PatchTST-style, iTransformer-style, DLinear-style, and dual-stream LSTM.
- `specs/official/`: the official 57-model pool split into tabular, aeon, and Keras method files.
