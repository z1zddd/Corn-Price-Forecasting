# Processed Corn Materials

Processed materials are cleaned tables ready to be loaded by configs.

Current tracked processed fixtures:

- `corn_sample_data.csv`: tiny smoke/diagnostic sample used by `configs/template.yaml`.
- `玉米价格月度_混合特征版 .csv`: monthly mixed-feature corn modeling table.
- `玉米价格月度_混合特征无缺失值双头LSTM版.csv`: default monthly modeling table used by `configs/corn.yaml` and official pool configs.

Do not write backtest outputs, metrics, plots, model artifacts, checkpoints, or
large local-only tables here.

## 玉米月度数据集 v1

- `corn_monthly_core_v1.csv`：从日频原始数据重新聚合的 121 个月、94 列核心特征。它不含 PCA 和未来目标列，滚动窗口不足时保留缺失值。
- `corn_monthly_news_legacy.csv`：从旧无缺失月度表隔离出的 32 个 PCA 特征。其原始来源和发布时间未知，默认不具备严格回测资格。
- `corn_monthly_v1_manifest.json`：记录输入文件哈希、数据截止日、行列规模、不完整月份和生成策略。

新实验应优先使用 `corn_monthly_core_v1.csv`，并排除 `is_complete_period=false` 的月份。旧 PCA 表只能作为单独的增量对照数据，不能默认并入严格回测。

运行 `python scripts/build_corn_monthly_dataset.py` 可以重复生成这些文件。当前 `configs/corn.yaml` 和官方模型池配置仍使用旧月度表；切换数据源需要另行修改配置。完整说明见 `docs/corn-monthly-v1.md`。
