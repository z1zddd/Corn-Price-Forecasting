# Corn Dataset Materials

This package documents corn-specific materials used by the forecasting
framework.

It is an internal material definition area, not a replacement for the existing
root-level `datasets/` sample data directory. Current configs can continue to
read files such as `datasets/玉米价格月度_混合特征无缺失值双头LSTM版.csv`.

Subdirectories:

- `raw/`: source price or external source materials.
- `processed/`: cleaned monthly modeling tables.
- `factors/`: corn factor materials such as futures, spot, basis, weather, or news PCA groups.
- `prediction_library/`: completed rolling prediction streams consumed by aggregation operators.
- `metadata/`: schema and source documentation.

Do not place generated experiment outputs, model weights, private data, large
archives, or ad hoc notebook exports here.
