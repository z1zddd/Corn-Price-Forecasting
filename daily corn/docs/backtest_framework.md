# 玉米日度多步长价格预测回测框架

## 1. 数据口径

输入文件：`corn_factors_daily_v1.csv`

| 项目 | 数值 |
| --- | ---: |
| 行数 | 2,426 |
| 列数 | 75 |
| 数值特征 | 74 |
| 日期范围 | 2016-06-29 至 2026-06-26 |
| 目标列 | `dce_corn_close` |
| 目标缺失 | 0 |
| 重复日期 | 0 |
| 含缺失值的行 | 370 |

日期按 DCE 实际交易日排列。预测步长和回看步长均按交易日行数计算，不按自然日计算。

## 2. 预测任务

预测步长：

```text
horizons = [5, 10, 15, 20]
```

回看步长：

```text
lookbacks = [5, 10, 15, 20]
```

二者独立组合，共 16 种 `horizon × lookback` 设置。

### 2.1 唯一预测目标

模型直接预测未来 `dce_corn_close`：

```text
y_price(t,h) = dce_corn_close(t+h)
y_hat_price(t,h) = model(X[t-lookback+1:t])
```

框架不要求方向分类头、收益回归头或双头结构。模型内部可以是任意架构，只要能够接收历史特征并输出未来价格。

### 2.2 统一派生结果

所有趋势、收益和交易指标均由当前价格、真实未来价格和预测未来价格计算：

```text
actual_return = actual_close_target / close_t - 1
predicted_return = predicted_close_target / close_t - 1

actual_trend = 1 if actual_close_target > close_t else 0
predicted_trend = 1 if predicted_close_target > close_t else 0
```

最后 `horizon` 行没有未来价格，应从对应任务中删除。每个 horizon 单独生成目标并记录有效样本范围。

## 3. 特征使用

框架支持三套特征范围：

| 特征集 | 内容 |
| --- | --- |
| `price_core` | DCE 玉米 OHLC、结算、成交量、持仓量及价格历史因子 |
| `daily_factor_30` | 数据中已构造的 30 个日频候选因子 |
| `full_safe` | 所有满足时点要求的原始字段和派生字段 |

时点规则：

- DCE 当日行情可在日 `t` 收盘后使用；
- CBOT 同日收盘晚于 DCE，相关原始字段必须滞后一个 DCE 交易日；
- 基差、现货和 100PPI 缺少精确发布时间，相关原始字段必须滞后一个 DCE 交易日；
- 名称带 `_lag1d` 的派生因子可直接使用；
- 禁止使用任何未来价格、未来收益、未来方向或 `target_*` 字段作为特征；
- 禁止全样本填补、缩放、缩尾、PCA 和特征选择。

## 4. 三种数据划分策略

三种策略独立运行，不做交叉组合：

```text
chronological_811
chronological_712
expanding_rolling_backtest
```

### 4.1 `chronological_811`

```text
训练集 80% / 验证集 10% / 测试集 10%
```

- 按日期顺序固定切分；
- 验证集用于选择参数；
- 参数确定后，可在满足标签可知性要求的训练集与验证集上重新拟合一次；
- 测试期内模型固定，不吸收测试标签，不重新训练。

### 4.2 `chronological_712`

```text
训练集 70% / 验证集 10% / 测试集 20%
```

规则与 `chronological_811` 相同，仅测试区间更长，用于检验模型在较长样本外时期的稳定性。

### 4.3 `expanding_rolling_backtest`

使用与 `chronological_712` 相同的日期骨架：

```text
初始训练 70% / 参数验证 10% / rolling-origin 测试 20%
```

进入测试期后，预测起点逐日向前滚动，训练窗口持续扩张：

```text
train_start = first_eligible_sample
train_end = latest sample whose target_date <= current_anchor_date
```

- 参数在中间 10% 验证期确定并冻结；
- 默认每个测试锚点重新训练一次；
- 可以使用测试期内已经实现且标签已知的早期样本；
- 不能使用当前锚点之后才会实现的价格；
- 不设置固定长度 rolling window。

## 5. 防止时间泄漏

对任意 horizon，训练样本必须满足：

```text
train_target_date <= prediction_anchor_date
```

固定时间切分还须满足：

```text
train_target_date <= first_test_anchor_date
```

边界处理：

- 训练集、验证集和测试集按当前 horizon 自动执行 purge；
- embargo 默认等于当前 horizon；
- 输入窗口不得跨越不允许使用的数据边界；
- 所有预处理器只能在当前训练数据上拟合；
- 任一时间断言失败时终止该实验，不得只记录警告后继续。

每次运行保存：

```text
train_start_date
train_end_date
train_max_target_date
validation_start_date
validation_end_date
test_start_date
test_end_date
prediction_anchor_date
prediction_target_date
```

## 6. 通用数据处理

每个训练过程使用相同的折内处理原则：

1. 按日期排序并检查重复；
2. 根据 horizon 生成未来价格目标；
3. 根据 lookback 构造输入窗口；
4. 可选的缩尾、PCA 和特征选择只在训练数据上拟合；
5. 对验证集和测试集只执行 `transform`；
6. 预测结果逆变换回真实价格尺度后再计算指标。


## 7. 通用模型接口

所有模型统一为价格回归接口：

```text
fit(X_train, y_price_train, validation_data=None)
predict(X_test) -> predicted_dce_corn_close
```

模型适配器负责：

- 接收表格或序列输入；
- 执行模型所需的数据形状转换；
- 训练模型；
- 输出与样本一一对应的未来价格；
- 保存模型参数、依赖版本、随机种子和训练状态。

框架不限制：

- 模型是否有编码器；
- 模型有几个内部输出头；
- 使用何种损失函数；
- 使用递归预测还是直接多步预测；
- 使用 sklearn、PyTorch、Keras、Aeon 或其他实现。

共同要求只有一个：最终必须输出 `predicted_dce_corn_close`。无法输出连续价格的纯分类模型不进入通用价格回测；若存在同家族回归版本，应使用其回归版本。

## 8. 模型范围

框架复用月度项目中的模型注册机制，模型由配置决定，不在回测流程中硬编码。

保留但不限于以下层级：

- 基线：当前价格不变、历史均价、移动平均、线性回归；
- 传统模型：Ridge、Elastic Net、SVR、KNN、随机森林、Extra Trees、Gradient Boosting；
- Boosting：LightGBM、XGBoost、CatBoost；
- 时序模型：LSTM、GRU、TCN、Transformer、PatchTST、iTransformer、DLinear；
- 月度框架中已有的 Aeon、Keras 和其他官方模型回归版本。

所有模型必须使用完全相同的数据划分、目标价格和评价函数。单个模型依赖缺失或训练失败时，记录失败原因并继续其他模型。

## 9. 参数选择

以下内容只能根据验证集选择：

- 模型超参数；
- lookback；
- 特征集；
- 缺失处理和缩放方式；
- 早停轮数；
- 可选模型集成规则。

测试集不得参与选择。模型运行3个种子，并报告均值、标准差和最差种子结果，不得只保留最佳种子。

## 10. 评价指标

### 10.1 价格预测指标

价格预测是核心任务，至少报告：

- MAE；
- RMSE；
- MAPE；
- sMAPE；
- R²；
- MASE；
- 相对“未来价格等于当前价格”基线的误差改善率。

主排序建议使用 RMSE 或 MASE，并同时检查 MAE 和 R²。

### 10.2 趋势指标

根据预测价格与当前价格的大小关系派生趋势，至少报告：

- Direction Accuracy；
- Balanced Accuracy；
- Precision；
- Recall；
- AP
- F1；
- MCC；
- 混淆矩阵；
- 实际上涨率和预测上涨率；
- 常量趋势预测检查。


### 10.3 经济指标

交易方向由预测价格派生：

```text
position = +1 if predicted_close_target > close_t else -1
strategy_return = position * actual_return
```

至少报告：

- 累计收益；
- 年化收益；
- Sharpe；
- Sortino；
- Calmar；
- 最大回撤；
- Profit Factor；
- 胜率；
- 平均盈利和平均亏损；
- 换手率；
- 交易次数；
- 0、2、5、10 bp 成本情景。

经济指标只评价由价格预测产生的方向，不作为价格模型的唯一排序依据。

## 11. 重叠预测处理

5、10、15、20 日预测在相邻锚点间重叠，不能把每天的多日收益直接连乘。

采用两套结果：

1. 全部锚点用于价格误差和趋势指标；
2. 交易指标使用非重叠持有期或等权错位组合。

对 horizon `h`，可按锚点序号模 `h` 分成 `h` 个非重叠子序列，分别计算交易指标，再报告均值、中位数和最差结果。置信区间使用 block bootstrap，block 长度不得小于当前 horizon。

## 12. 三种策略的结果报告

必须分别输出：

```text
test_results_chronological_811.csv
test_results_chronological_712.csv
test_results_expanding_rolling_backtest.csv
split_strategy_comparison.csv
```

结果表至少包含：

```text
model
feature_set
horizon
lookback
split_strategy
seed
test_start
test_end
n_predictions
price_metrics
trend_metrics
economic_metrics
```

`chronological_712` 与 `expanding_rolling_backtest` 使用相同测试日期，因此应直接比较固定模型与动态扩张重训的指标差异。

最终报告按以下顺序组织：

1. 数据检查摘要；
2. `chronological_811` 测试结果；
3. `chronological_712` 测试结果；
4. `expanding_rolling_backtest` 测试结果；
5. 三种策略对比；
6. horizon 与 lookback 对比；
7. 模型稳定性和失败记录；
8. 最终推荐及限制。

## 13. 模型选择规则

模型先通过以下检查：

- 无时间泄漏；
- 预测价格无 NaN 和无穷值；
- 测试日期覆盖完整；
- 结果优于当前价格不变基线；
- 不依赖单个随机种子的异常高分。

排序原则：

1. 价格误差为辅；
2. 趋势准确性为主；
3. 经济指标用于判断预测是否具有实际方向价值；
4. 优先选择在三种数据划分策略下都稳定的模型；
5. 不选择仅在单一策略或单一 horizon 上偶然高分的模型。

不同 horizon 可以选择不同最佳模型，但所有模型必须通过相同接口和相同评价流程。

## 14. 输出目录

```text
experiments/<run_id>/
  config_resolved.yaml
  data_audit.json
  experiment_manifest.json
  fold_audit.csv
  all_predictions.csv
  test_results_chronological_811.csv
  test_results_chronological_712.csv
  test_results_expanding_rolling_backtest.csv
  split_strategy_comparison.csv
  model_failures.csv
  report.md
  horizon_5/
  horizon_10/
  horizon_15/
  horizon_20/
    <lookback>/
      <split_strategy>/
        <model>/
          predictions.csv
          metrics.json
          fold_audit.csv
          preprocessing_manifest.json
```

`predictions.csv` 至少包含：

```text
anchor_date
target_date
close_t
actual_dce_corn_close
predicted_dce_corn_close
actual_return
predicted_return
actual_trend
predicted_trend
horizon
lookback
split_strategy
model
seed
```

## 15. 配置示例

```yaml
data:
  csv_path: C:/Users/YLHP/Downloads/corn_factors_daily_v1.csv
  date_col: date
  target_col: dce_corn_close

target:
  type: future_price
  horizons: [5, 10, 15, 20]

lookback:
  candidates: [5, 10, 15, 20]

feature_sets:
  enabled: [price_core, daily_factor_30, full_safe]

split:
  strategies:
    chronological_811:
      ratios: [0.8, 0.1, 0.1]
      refit_during_test: false
    chronological_712:
      ratios: [0.7, 0.1, 0.2]
      refit_during_test: false
    expanding_rolling_backtest:
      ratios: [0.7, 0.1, 0.2]
      expanding_window: true
      rolling_origin: true
      refit_stride: 1
  purge: auto_horizon
  embargo: auto_horizon
  shuffle: false

preprocessing:
  fit_on_train_only: true
  imputer: median
  scaler: model_specific

evaluation:
  price_metrics: [MAE, RMSE, MAPE, sMAPE, R2, MASE]
  trend_metrics: [DirectionAccuracy, BalancedAccuracy, F1, MCC]
  economic_metrics: [AnnRet, Sharpe, Sortino, MaxDD, ProfitFactor]
  transaction_cost_bps: [0, 2, 5, 10]
  block_bootstrap: true

models:
  source: monthly_framework_registry
  task: price_regression
```

## 16. 执行流程

```text
读取并检查完整数据

for horizon in [5, 10, 15, 20]:
    生成未来 dce_corn_close 目标

    for lookback in [5, 10, 15, 20]:
        构造历史输入窗口

        for split_strategy in [chronological_811,
                               chronological_712,
                               expanding_rolling_backtest]:
            生成时间划分并执行 purge/embargo

            for model in configured_models:
                在训练数据上拟合预处理器
                训练价格预测模型
                输出 predicted_dce_corn_close
                由价格派生 predicted_return 和 predicted_trend
                计算价格、趋势和经济指标
                保存预测、指标和审计记录

分别汇总三种数据划分策略
生成策略对比和最终报告
```

## 17. 最低验收条件

- 5、10、15、20 四个预测步长全部运行；
- 5、10、15、20 四个回看步长全部运行；
- 三种数据划分策略分别产生测试结果；
- 所有模型直接输出未来 `dce_corn_close`；
- 趋势和收益均由价格预测结果派生；
- 固定切分测试期不重训；
- expanding rolling backtest 只使用当前时点已知标签；
- 所有预处理只在训练数据上拟合；
- 重叠持有期没有被直接连乘；
- 完整记录模型失败和时间审计信息。
