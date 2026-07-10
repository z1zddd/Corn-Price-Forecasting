# Corn Forecast Package

`corn_forecast` 是框架主包。命令行、配置加载、数据处理、模型构建、回测、评估和报告都从这里进入。

- `cli.py`: `commodity-backtest` 命令入口。
- `config/`: YAML 配置读取和校验。
- `data_processing/`: 数据读取、目标生成、特征选择、标准化和窗口构造。
- `datasets/`: 数据材料、schema、来源说明和小型可提交 CSV fixture。
- `modeling/`: 单模型、模型池、聚合、损失变体和适配器。
- `pipeline/`: 回测、训练、评估和报告流程编排。
