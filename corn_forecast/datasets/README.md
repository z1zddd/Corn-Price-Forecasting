# Datasets Material Layer

`corn_forecast.datasets` describes the data materials that operators consume.

In this project:

```text
time-series model = operator + datasets
```

This package is the framework-internal material layer. It records and stores
small repository-approved dataset assets, expected schemas, source notes, and
prediction-library conventions. It does not run feature engineering, target
generation, window construction, training, evaluation, reporting, scheduling,
or experiment output writing.

Corn materials live under `corn_forecast/datasets/corn/`. The old root-level
`datasets/` directory has been folded into that corn material area so configs
and schemas point to one canonical location.

Do not place large private feeds, model weights, archives, or generated
experiment outputs in this package. Put private/local data under `local_data/`
or another ignored external path.
