# Operator Layer

`corn_forecast.operator` contains reusable methods that act on datasets.

In this project, a time-series model is understood as:

```text
operator + datasets
```

Operators include model families, feature transformations, target construction,
windowing, scaling, loss variants, wrappers, and aggregation methods. They do
not own experiment orchestration, evaluation reports, scheduling, or generated
outputs. Those responsibilities stay in `corn_forecast.pipeline`.

The first refactor phase moves model implementations to
`corn_forecast.operator.model` while keeping `corn_forecast.modeling` as a
legacy compatibility path.
