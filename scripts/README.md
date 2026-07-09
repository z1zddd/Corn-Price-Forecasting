# Scripts

这里存放研究和维护脚本。脚本可以调用 `corn_forecast` 主包，但不应该承载核心业务逻辑。

- `run_best_aggregate_from_predictions.py`: 从已有滚动预测结果复算聚合策略。
- `run_best_deployment_combo.py`: 运行部署候选组合。
- `search_deployment_combinations.py`: 搜索部署组合。
- `search_prediction_ensembles.py`: 搜索预测聚合方案。
- `evaluate_deployment_holdout.py`: 做留出区间评估。
- `validate_official_pool.py`: 审计官方 57 模型池配置和时间窗口。
- `clean_outputs.py`: 清理本地输出目录。

