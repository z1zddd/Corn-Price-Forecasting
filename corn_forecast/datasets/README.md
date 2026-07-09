# Datasets Material Layer

`corn_forecast.datasets` describes the data materials that operators consume.

In this project:

```text
time-series model = operator + datasets
```

This package is the framework-internal material layer. It records dataset
assets, expected schemas, source notes, and prediction-library conventions. It
does not run feature engineering, target generation, window construction,
training, evaluation, reporting, scheduling, or experiment output writing.

The root-level `datasets/` directory is retained as the repository-compatible
sample data area. Existing configs such as `configs/corn.yaml` may continue to
point there. This package documents the materials and their schemas without
breaking those paths.

Do not place large raw data dumps, private enterprise data, model weights,
archives, or generated experiment outputs in this package.
