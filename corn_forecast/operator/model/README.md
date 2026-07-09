# Model Operators

`corn_forecast.operator.model` is the canonical model-operator package.

It contains:

- `base.py`: shared model protocols and interfaces.
- `registry/`: YAML-facing model creation and model-pool expansion.
- `losses/`: loss-oriented model variants.
- `wrappers/`: third-party runtime adapters.
- `families/`: baseline, classical, sequence, official, and aggregation model
  families.

`corn_forecast.modeling` remains as a short-term legacy compatibility package
that re-exports these implementations. New code should import from
`corn_forecast.operator.model`.

Pipelines are intentionally outside this package. Training loops, backtests,
evaluation, reporting, scheduling, and smoke scripts belong under
`corn_forecast.pipeline` or `scripts/`.
