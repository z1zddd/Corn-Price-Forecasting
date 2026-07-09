# PatchTST-FM LoRA 微调待办规划

## 目标

使用云服务器上的 PatchTST-FM 时序基础模型，对中国玉米期货多变量数据做 LoRA 微调。

主要交易预测目标：

- 预测未来 5 个交易日收益率。
- 预测未来 10 个交易日收益率。
- 输出每日预测、方向胜率、收益曲线和最大回撤。

服务器模型路径：

```bash
/root/PatchTST-FM
```

服务器数据路径：

```bash
/root/china_corn_forecast/china_corn_trading_day_trend_dataset_enriched.csv
```

## 已有基线结果

已经完成：

- Rolling Ridge：756 天训练窗口，每天向前滚动一步。
- PatchTST-FM zero-shot：多变量输入，756 天上下文，每天预测。

zero-shot 关键结果：

- 5td 方向胜率：46.42%。
- 10td 方向胜率：45.09%。
- 结论：PatchTST-FM zero-shot 不能直接作为正向交易信号，LoRA 微调必须明显优于这个基线才有意义。

## 数据窗口形状

原始数据：

```text
2426 个交易日
```

窗口设计：

```text
context_length = 756
prediction_length = 10
step = 1
```

理论可构造样本数：

```text
2426 - 756 - 10 + 1 = 1661 个窗口样本
```

每个样本：

```text
X: [756, 44]
Y: [10]
```

说明：

- `X` 使用除了日期列和未来目标列之外的所有数值特征。
- 第一个特征固定为 `dce_corn_close`。
- `Y` 只使用未来 10 天的 `dce_corn_close` 路径。
- 模型实际会输出 `[batch, 10, 44]`，但第一版 loss 只计算 `dce_corn_close` 这一列。

## 防止未来数据泄漏

对某个日期 `t` 做预测时：

- 输入窗口可以使用到 `t` 当天为止的数据。
- 训练样本的标签必须在 `t` 当天已经完全可见。
- 如果训练目标是未来 10 天路径，那么训练样本最晚只能用到 `t - 10`。

也就是说：

```text
预测 t:
训练样本 anchor <= t - 10
预测输入窗口 = t-755 到 t
预测输出 = t+1 到 t+10
```

这个规则很重要，否则回测会看起来很好，但实盘不可用。

## LoRA 冻结与解冻策略

第一版不做全模型微调。

冻结：

- PatchTST-FM 原始基础模型全部参数。
- 输入投影层原始权重。
- 输出投影层原始权重。
- 归一化层。
- 分位数预测头原始权重。

训练：

- 只训练 LoRA adapter 参数。

第一版 LoRA 插入模块：

```text
qkv
proj
```

这些模块在服务器文件中：

```bash
/root/PatchTST-FM/basic.py
```

初始 LoRA 参数：

```text
r = 4
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["qkv", "proj"]
bias = "none"
```

如果第一版欠拟合，再尝试：

- `r = 8`
- 对全部 attention block 注入 LoRA
- 在第二阶段考虑解冻最后输出头的一小部分参数

不要一开始就全模型解冻，样本太少，过拟合风险很高。

## 第一版实验设计

实验名：

```text
lora_multivar_close_loss_exp001
```

设置：

```text
context_length = 756
prediction_length = 10
输入特征 = 44 个数值特征
loss = 只计算未来 dce_corn_close 路径的 MSE
```

时间切分：

```text
train: 2023-01-01 之前结束的窗口
valid: 2023 年结束的窗口
test: 2024 年及以后结束的窗口
```

训练参数建议：

```text
batch_size = 8
epochs = 5
learning_rate = 1e-4
weight_decay = 0.01
early_stopping_patience = 2
gradient_clip = 1.0
precision = 优先 bf16，不稳定则 fp32
```

GPU：

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
```

## 需要创建的脚本

建议放在：

```bash
/root/china_corn_forecast/scripts/
```

待办：

- [x] `prepare_patchtst_lora_dataset.py`
  - 构造窗口化数据。
  - 保存特征列列表。
  - 保存 scaler/imputer 元数据。
  - 保存 train/valid/test 切分索引。
  - 确认未来目标列不进入输入特征。

- [x] `train_patchtst_lora.py`
  - 加载 `/root/PatchTST-FM`。
  - 冻结基础模型。
  - 给 `qkv` 和 `proj` 注入 LoRA。
  - 只训练 LoRA 参数。
  - 保存 adapter、训练日志和验证集指标。

- [x] `evaluate_patchtst_lora.py`
  - 加载基础模型和 LoRA adapter。
  - 在测试集上预测。
  - 输出 5td 和 10td 预测收益。
  - 输出指标、收益曲线和最大回撤。

- [x] `evaluate_patchtst_lora_walk_forward.py`
  - 做严格 walk-forward 回测。
  - 使用 756 天窗口。
  - 每天输出预测。
  - 第一版先按固定间隔重训 adapter，不要每天重训。

## 推荐执行顺序

1. 确认 `peft` 安装成功。
2. 检查 PatchTST-FM 模块名，确认 LoRA 能匹配 `qkv/proj`。
3. 构造 32 个样本的小训练集做 smoke test。
4. 确认可训练参数占比小于 1%。
5. 跑 1 个 epoch，确认训练 loss 能下降。
6. 跑 `exp001` 固定 train/valid/test 切分实验。
7. 对比：
   - Rolling Ridge 基线。
   - PatchTST-FM zero-shot 基线。
8. 只有固定切分明显改善后，再做 walk-forward 版本。

## 评估指标

分别对 5td 和 10td 输出：

- MAE
- RMSE
- 方向胜率
- 看多信号胜率
- 看空信号胜率
- 预测看多比例
- 每次预测的平均策略收益
- 每次预测的中位数策略收益
- 收益曲线
- 最大回撤

进入下一阶段的最低标准：

```text
方向胜率 >= 52%
最大回撤显著好于 zero-shot
最终权益高于 Ridge 基线，或者高置信度子集明显更好
```

## 输出目录规范

建议输出到：

```bash
/root/china_corn_forecast/outputs/lora_multivar_close_loss_exp001/
```

目录结构：

```bash
config.json
feature_columns.json
train_log.csv
valid_metrics.csv
test_predictions_5td.csv
test_predictions_10td.csv
summary.csv
equity_drawdown.png
report.md
adapter/
```

## 主要风险

- 总样本只有约 1661 个窗口，LoRA 也可能过拟合。
- zero-shot 方向胜率低于 50%，说明原始模型和玉米市场分布不匹配。
- 很多外生变量未来不可预测，第一版 loss 不应该强迫模型预测所有变量。
- 每天重新 LoRA 微调成本高，第一版应先做固定切分，再做周期性重训。
- 缺失值处理和标准化必须只基于训练数据，不能用测试期信息。

## 待决策问题

- 使用每个窗口单独标准化，还是使用训练集全局标准化？
- LoRA 注入全部 attention block，还是只注入后几层？
- loss 只用点预测，还是加入分位数中位数约束？
- 是否增加信号过滤，例如：

```text
abs(predicted_return) > 0.5%
abs(predicted_return) > 1.0%
```
