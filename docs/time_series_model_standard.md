# Time Series Model Standard

本项目时序实验按成熟开源库的共同规范执行：

- scikit-learn `TimeSeriesSplit`：时间序列不能随机切分，测试索引必须在训练索引之后。
- sktime：测试集必须在训练集未来；单次 holdout 之外，正式评估建议 full backtesting。
- Nixtla NeuralForecast / MLForecast / StatsForecast：用 rolling/sliding window cross-validation，在多个历史 cutoff 上训练并预测后续窗口。
- Darts：用 historical forecasts/backtesting 模拟历史上的真实预测。
- GluonTS / PyTorch Forecasting：数据必须显式声明时间列、目标列、序列 id/时间索引、预测 horizon。

## 1. 数据契约

所有数据进入模型前必须落成统一语义：

| 字段 | 含义 |
|---|---|
| `date` | 当前可见交易日 |
| `target_date` | 被预测的未来交易日 |
| `today_close` | 当前交易日收盘价 |
| `y` | 模型训练目标 |
| `horizon` | `target_date - date` 对应的交易步数 |
| `feature_cols` | 仅允许使用 `date` 当天及以前可见的信息 |

外部库的等价概念：

| 本项目 | Nixtla | GluonTS | PyTorch Forecasting |
|---|---|---|---|
| 商品/合约 id | `unique_id` | item entry | `group_ids` |
| 时间列 | `ds` | `start` + target index | `time_idx` |
| 目标 | `y` | `target` | `target` |

## 2. 目标变量标准

默认训练目标是未来收益率：

```text
y = close[t+h] / close[t] - 1
```

评估时再还原为价格：

```text
pred_price = today_close * (1 + pred_return)
true_price = today_close * (1 + true_return)
```

原因：跨年份价格水平会漂移。直接预测未来绝对价格容易让模型把价格水平当成目标，导致训练低价区间、测试高价区间时方向全错。

如需复现原始 LSTM 脚本的 next-day price 任务，必须显式设置：

```yaml
horizon: 1
target_mode: price
```

## 3. 切分标准

所有 split 必须按 `target_date` 切，而不是按 anchor `date` 切：

```text
train/val target_date < test_start
test      target_date >= test_start
```

理由：模型样本的标签发生在 `target_date`。如果只按 anchor date 切，可能出现训练样本的输入日期在测试前，但标签日期已经进入测试期。

## 4. 归一化标准

归一化只能 fit 在训练集：

```text
fit scaler on train only
transform train/val/test with same scaler
```

X 按原始 LSTM 脚本：

```text
1. train median 填充 NaN
2. train mean/std 标准化
3. val/test 复用 train median/mean/std
```

y 按同一个训练集 scaler 记录：

```text
y_scaled = (y - train_y_mean) / train_y_std
```

禁止在 loader 里提前对全量数据 impute/scale。

## 5. 模型接口标准

所有模型遵守：

```python
fit(X_train, y_train, X_val=None, y_val=None)
predict(X) -> np.ndarray
save(path)
load(path)
```

输入统一为：

```text
X: [N, V, T]
y: [N]
```

DL 内部可以转成 `[N, T, V]`，但不能把这个形状泄漏给外层。

## 6. 评估标准

每个模型必须输出：

```text
predictions.csv
metrics.json
metrics.csv
equity_curve.png
model_config.json
model.pkl
```

主指标顺序：

```text
direction_accuracy
profit_factor
sharpe_ratio
win_rate
max_drawdown
```

辅助指标：

```text
rmse
mae
r2
```

方向和收益永远基于价格还原后的结果计算：

```text
pred_dir = pred_price > today_close
true_dir = true_price > today_close
strategy_return = long if pred_dir else short
```

## 7. 验证标准

单次 holdout 只能作为 smoke / sanity check。

正式报告必须至少包含 rolling-origin backtest：

```text
cutoff_1: train <= cutoff_1, predict next h
cutoff_2: train <= cutoff_2, predict next h
...
```

每个 cutoff 必须记录：

```text
cutoff_date
train_start
train_end
test_start
test_end
horizon
model
metrics
```

## 8. 失败即停检查

训练前必须检查：

- `target_date > date`
- train/val/test 按 `target_date` 单调向后
- scaler 只 fit train
- transform 后没有 NaN/inf
- `predictions.csv` 里 `pred_price` 不是常数
- `pred_direction` 不能长期单边，除非报告明确解释原因

