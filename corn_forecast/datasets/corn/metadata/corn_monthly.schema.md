# Corn Monthly Modeling Table Schema

The corn monthly modeling table is the tabular material loaded by configs such
as `configs/corn.yaml`.

Core fields:

| field | meaning |
| --- | --- |
| `month` | Default monthly date column used by corn configs. |
| `dce_corn_close` | Default price column used to generate forward targets. |

Feature columns:

- Configs may use `feature_cols: auto_numeric` to select numeric feature
  columns.
- Feature columns can include futures, spot, basis, spread, weather, seasonal,
  and PCA/news materials.
- Target-like columns should be excluded from features.

Target-like columns:

- `target_price_fwd`
- `target_return_fwd`
- `target_direction_fwd`
- `target_date_fwd`
- `dce_corn_close_next_month`
- `dce_corn_close_next_month_ret`
- `spike`

PCA/news columns:

- News or text-derived PCA features conventionally use names such as `pca_001`.
- No-news configs exclude `pca_*` and `PCA*`.
- With-news configs keep those PCA/news columns as part of the material.

Canonical location:

- Current corn configs read monthly fixtures from
  `corn_forecast/datasets/corn/processed/`.
- Raw/source-like corn fixtures live in `corn_forecast/datasets/corn/raw/`.
