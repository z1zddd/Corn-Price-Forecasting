# Corn Dataset Materials

This is the canonical home for corn data materials used by the forecasting
framework. The former root-level `datasets/` CSV files now live here so corn
configs, schemas, and source notes point to one place.

## Directory Map

| directory | put these files here | do not put these here |
| --- | --- | --- |
| `raw/` | Source-like corn exports before monthly modeling cleanup, for example `玉米价格原始数据.csv`. | Backtest reports, model predictions, checkpoints, private vendor dumps. |
| `processed/` | Small cleaned modeling tables that configs can load directly, for example `corn_sample_data.csv`, `玉米价格月度_混合特征版 .csv`, and `玉米价格月度_混合特征无缺失值双头LSTM版.csv`. | Generated experiment outputs, model artifacts, huge local-only tables. |
| `factors/` | Factor tables and source notes such as DCE/CBOT futures, spot, basis, weather, seasonal, and news/PCA inputs. | Final training targets, predictions, or one-off notebooks. |
| `prediction_library/` | Completed rolling prediction streams such as `all_rolling_predictions.csv` when intentionally approved as small fixtures. | Raw feature tables or model training checkpoints. |
| `metadata/` | Schema, field definitions, source notes, update cadence, and lineage docs. | CSV data, metrics, plots, or generated reports. |

## Current Tracked Files

```text
raw/
  玉米价格原始数据.csv

processed/
  corn_sample_data.csv
  玉米价格月度_混合特征版 .csv
  玉米价格月度_混合特征无缺失值双头LSTM版.csv

factors/
  README.md
  corn_factors_monthly.csv
  corn_factors_weekly.csv
  corn_factors_yearly.csv

prediction_library/
  README.md

metadata/
  README.md
  corn_monthly.schema.md
  all_rolling_predictions.schema.md
```

Large raw data dumps, enterprise-private feeds, generated prediction libraries,
model weights, archives, and temporary notebook exports should stay outside git,
for example under `local_data/`.
