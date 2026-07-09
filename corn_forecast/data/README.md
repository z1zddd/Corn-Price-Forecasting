# Data

这里负责数据进入模型前的处理。

- `loader.py`: 读取 CSV、识别编码、选择特征列。
- `diagnosis.py`: 数据诊断和候选列识别。
- `targets.py`: 根据价格列生成 forward target。
- `windowing.py`: 按 lookback 构造时间序列样本。
- `scaler.py`: 在每个训练窗口内拟合标准化器，避免跨窗口泄漏。

