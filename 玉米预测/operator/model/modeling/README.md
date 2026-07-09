# Modeling

这里是模型层本体。

- `classical/`: 传统机器学习和基准模型。
- `deep/`: LSTM、GRU、Transformer、PatchTST、DLinear、TimeXer 等深度时序模型。
- `official/`: Chronos2、TimePFN、Timer、TiRex 等官方模型适配。
- `layers/`: 深度模型内部层。
- `utils/`: 模型内部工具。
- `base.py`: 模型统一接口。

旧的 `src.models.*`、`layers.*`、`utils.*` import 通过兼容壳继续指向这里。

