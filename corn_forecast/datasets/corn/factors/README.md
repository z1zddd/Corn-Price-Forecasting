# Corn Factor Materials

Factor materials are explanatory inputs for corn forecasting. This directory is
organized as a small factor library rather than a loose collection of wide CSVs.

## Layout

```text
factors/
  README.md
  factor_storage_design.zh.md
  registry.yaml
  library/
    <factor_id>/
      factor.yaml
      values.csv
  matrix/
    corn_factors_monthly.csv
    corn_factors_weekly.csv
    corn_factors_yearly.csv
```

## Layers

- `library/` is the source of truth for factor definitions and long-form factor
  values. Each factor directory contains:
  - `factor.yaml`: how the factor is defined, which inputs it uses, what it
    outputs, and how leakage is controlled.
  - `values.csv`: long-form values with `period_end`, `period_key`,
    `frequency`, `instrument`, `factor_id`, `value`, `coverage`, `asof_date`,
    `quality_flag`, and `source_version`.
- `registry.yaml` is the factor index used to discover available factors,
  groups, frequencies, and outputs.
- `matrix/` contains model-adjacent wide tables. These files are convenient for
  quick training, smoke tests, and manual inspection, but they are not the only
  source of factor definitions.

## Current Factors

- `price_momentum`: price momentum.
- `basis_tightness`: basis tightness.
- `processing_demand`: processing demand.
- `market_activity`: market activity.
- `external_support`: external support.
- `weather_anomaly`: weather anomaly.
- `supply_pressure`: supply pressure.
- `risk_volatility`: risk and volatility.
- `harvest_pressure`: harvest pressure. The current source matrices do not
  include a dedicated coverage column for this factor.
- `seasonal`: calendar encodings, stored as `seasonal_sin` and `seasonal_cos`.

## Boundaries

Forward target columns such as `target_date_fwd`, `target_price_fwd`,
`target_return_fwd`, and `target_direction_fwd` may appear in matrix files for
training or evaluation convenience, but they do not belong in single-factor
`values.csv` files.

Feature selection, target generation, scaling, and windowing remain outside this
material layer.
