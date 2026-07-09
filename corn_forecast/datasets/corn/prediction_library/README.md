# Corn Prediction Library

Prediction-library materials contain completed rolling prediction streams, such
as `all_rolling_predictions.csv`.

These files are consumed by aggregation operators such as
`corn_forecast.operator.model.families.aggregation.deployment_vote`. They are
not raw feature tables, and aggregation operators should not retrain base
models from them.

Generated prediction libraries can become large. Keep large generated CSVs,
compressed files, and experiment output directories out of git unless a small
fixture is explicitly approved.
