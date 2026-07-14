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
    monthly_v1/
      <factor_family>/
        factor.yaml
        values.csv
    daily_v1/
      factor_set.yaml
  matrix/
    corn_factors_daily_v1.csv
    corn_factors_monthly.csv
    corn_factors_monthly_v1.csv
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
  `daily_v1` uses one centralized `factor_set.yaml` and the canonical wide
  matrix instead of duplicating all daily values into long-form files.
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

## Monthly v1 Candidate Set

`monthly_v1` is the leakage-aware monthly redesign built from
`../processed/corn_monthly_core_v1.csv`. It adds 10 factor families with 21
candidate outputs under `library/monthly_v1/` and a target-free wide matrix at
`matrix/corn_factors_monthly_v1.csv`.

The set uses month-t factors to predict month t+1. Spot, basis, and 100PPI
inputs are lagged by one month until publication timestamps are available.
Weather anomalies use only prior years from the same calendar month. Missing
rolling history is preserved, and incomplete months are not strict-backtest
eligible.

See `docs/corn-monthly-factors-v1.md` for formulas and usage rules, and
`monthly_v1_data_gaps.yaml` for the data acquisition backlog. The old factor
library and all weekly/yearly files remain unchanged.

## Daily v1 Candidate Set

`daily_v1` is generated directly from `../raw/玉米价格原始数据.csv`. It contains
9 families and 30 target-free candidate factors for 2,426 DCE trading-day rows.
The wide matrix is `matrix/corn_factors_daily_v1.csv`; the centralized formula
and timing definition is `library/daily_v1/factor_set.yaml`.

Use factors available after DCE row t ends to predict the next actual DCE row.
Basis, 100PPI, and CBOT inputs are lagged by one DCE row. Rolling warm-up and
source gaps remain missing. See `docs/corn-daily-factors-v1.md` and
`daily_v1_data_gaps.yaml`. Monthly, weekly, yearly, and model configuration
files remain unchanged.
