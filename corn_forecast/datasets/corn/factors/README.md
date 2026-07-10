# Corn Factor Materials

Factor materials are explanatory inputs for corn forecasting, such as:

- DCE corn and corn starch futures features.
- CBOT corn and wheat features.
- Spot, basis, spread, or nearby futures features.
- Weather and seasonal indicators.
- News or text PCA columns.

Tracked files:

- `corn_factors_monthly.csv`: monthly factor table for model-ready or
  model-adjacent monthly features.
- `corn_factors_weekly.csv`: weekly factor table for higher-frequency source
  signals before monthly alignment.
- `corn_factors_yearly.csv`: yearly factor table for annual context features.

Feature selection, target generation, scaling, and windowing remain outside this
material layer.
