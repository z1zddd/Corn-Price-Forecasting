# Configs

每个模型保存一个可编辑 YAML 配置，例如 `lstm.yaml` 或 `xgboost.yaml`。

配置可包含数据路径、模型参数、调参范围、预测步长、回看步长、数据划分策略、随机种子和输出路径。每次实验实际使用的配置应另存为结果目录中的 `config_resolved.yaml`。
