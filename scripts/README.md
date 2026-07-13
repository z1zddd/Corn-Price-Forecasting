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

这里存放研究和维护脚本。脚本可以调用 `corn_forecast` 主包，但不应该承载核心业务逻辑。

- `run_best_aggregate_from_predictions.py`: 从已有滚动预测结果复算聚合策略。
- `run_best_deployment_combo.py`: 运行部署候选组合。
- `search_deployment_combinations.py`: 搜索部署组合。
- `search_prediction_ensembles.py`: 搜索预测聚合方案。
- `evaluate_deployment_holdout.py`: 做留出区间评估。
- `validate_official_pool.py`: 审计官方 57 模型池配置和时间窗口。
- `clean_outputs.py`: 清理本地输出目录。
