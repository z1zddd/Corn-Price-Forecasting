# Modeling

这里是模型层，负责把配置里的模型名称解析成可训练、可预测的模型对象。

- `baselines/`: 简单基准模型。
- `classical/`: scikit-learn 风格的传统机器学习模型。
- `sequence/`: LSTM、GRU、Transformer、PatchTST、iTransformer、DLinear 等序列模型。
- `losses/`: 损失函数变体和由回归信号转方向信号的模型。
- `registry/`: 模型注册和工厂分发。
- `wrappers/`: 第三方运行时适配器。
- `ensembles/`: 部署组合和投票聚合逻辑。
- `specs/`: 命名模型池定义，官方 57 模型池在 `specs/official/` 下继续拆分。

单个模型实现优先放到对应类型目录；聚合选择、模型池展开和训练流程不要混在一个文件里。
