# Scripts

## 月度数据生成

`build_corn_monthly_dataset.py` 从仓库内的玉米日频原始 CSV 重新聚合月度核心特征，将旧无缺失月度表中的 PCA 字段隔离到单独文件，并生成记录来源哈希和数据边界的清单。

```bash
python scripts/build_corn_monthly_dataset.py
```

脚本输出：

- `corn_monthly_core_v1.csv`：推荐用于新实验的核心特征表。
- `corn_monthly_news_legacy.csv`：仅供增量对照实验使用的旧 PCA 表。
- `corn_monthly_v1_manifest.json`：输入来源、数据截止日和生成策略。

脚本会校验月份唯一性、时间顺序、不完整月份和目标列边界，但不会修改模型配置。详细规则见 `docs/corn-monthly-v1.md`。

## 月度因子生成 v1

`build_corn_monthly_factors.py` 从 `corn_monthly_core_v1.csv` 生成 10 个因子族、21 个候选月度因子，并输出无目标列的月度因子矩阵、长表因子库和可复现清单。

```bash
python scripts/build_corn_monthly_factors.py
```

该脚本只写入 `factors/library/monthly_v1/`、`factors/matrix/corn_factors_monthly_v1.csv` 和 `factors/monthly_v1_manifest.json`，不会修改旧月度矩阵、周度矩阵、年度矩阵或模型配置。详细规则见 `docs/corn-monthly-factors-v1.md`。

## 日频因子生成 v1

`build_corn_daily_factors.py` 直接从 `raw/玉米价格原始数据.csv` 生成 9 个因子族、30 个候选日频因子。基差、100PPI 和 CBOT 字段滞后 1 个 DCE 行，滚动预热与原始缺口保持为空。

```bash
python scripts/build_corn_daily_factors.py
```

脚本只写入 `factors/library/daily_v1/factor_set.yaml`、`factors/matrix/corn_factors_daily_v1.csv` 和 `factors/daily_v1_manifest.json`。它不会修改原始数据、月度/周度/年度因子或模型配置。详细规则见 `docs/corn-daily-factors-v1.md`。

## 产业链日频因子生成 v1

`build_corn_market_daily_factors.py` 从本地外部 `raw_quotes.csv` 和 `normalized_prices.csv` 生成 12 个短历史产业链因子。源明细应放在被 Git 忽略的 `local_data/corn_market/`，不会复制到公开数据目录。

```bash
python scripts/build_corn_market_daily_factors.py
```

脚本验证两份源文件的记录 ID 一致性，按 15:00 截止时间映射到 DCE 交易日，并写入 `factors/library/daily_market_v1/factor_set.yaml`、`factors/matrix/corn_market_daily_factors_v1.csv` 和 `factors/daily_market_v1_manifest.json`。该因子集只用于影子回测和残差增强，详见 `docs/corn-daily-market-factors-v1.md`。

这里存放研究和维护脚本。脚本可以调用 `corn_forecast` 主包，但不应该承载核心业务逻辑。

- `run_best_aggregate_from_predictions.py`: 从已有滚动预测结果复算聚合策略。
- `run_best_deployment_combo.py`: 运行部署候选组合。
- `search_deployment_combinations.py`: 搜索部署组合。
- `search_prediction_ensembles.py`: 搜索预测聚合方案。
- `evaluate_deployment_holdout.py`: 做留出区间评估。
- `validate_official_pool.py`: 审计官方 57 模型池配置和时间窗口。
- `clean_outputs.py`: 清理本地输出目录。
