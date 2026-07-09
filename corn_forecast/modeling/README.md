# Modeling Compatibility Path

`corn_forecast.modeling` is now a legacy compatibility package.

The canonical model-operator implementation lives in
`corn_forecast.operator.model`. Files under this directory re-export the new
paths so old imports continue to work during the refactor.

New code should import from `corn_forecast.operator.model`.
