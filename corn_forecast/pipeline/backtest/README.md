# Backtest

这里负责时间序列回测。

- `splits.py`: 根据 expanding、rolling、capped expanding 规则生成窗口。
- `engine.py`: 在每个窗口上训练模型、预测、汇总结果并写出产物。

这一层必须保持时间顺序，不能随机打乱训练和预测窗口。

