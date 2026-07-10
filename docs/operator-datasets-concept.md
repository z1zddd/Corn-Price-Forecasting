# Operator And Datasets Concept

本文档记录本项目后续结构优化的核心理念。它不是一次性任务清单，而是仓库长期遵守的分层原则。

## 1. 一句话

```text
一个时序模型 = operator + datasets
```

更完整地说：

```text
forecasting system = datasets + operator + pipeline
```

- `datasets` 是被操作的材料。
- `operator` 是作用在材料上的方法。
- `pipeline` 是把材料和方法组织起来运行的流程。

模型不应该被理解成孤立的一个 Python 文件。真正落地时，模型效果来自“材料是什么、方法怎么处理材料、流程如何保证时间顺序和评估边界”三件事共同作用。

## 2. 为什么要这样分

时序预测项目最容易乱在三个地方：

- 数据材料和模型方法混在一起。
- 回测评估和模型定义混在一起。
- 历史实验输出和可复用工程代码混在一起。

`operator + datasets` 的分法是为了让每个对象有清楚归属。

同一个 `operator` 可以作用在不同 `datasets` 上。例如同一个 random forest 可以跑玉米、豆粕、螺纹钢，也可以跑 no-news 材料或 with-news PCA 材料。

同一个 `datasets` 也可以被多个 `operator` 使用。例如同一份玉米月度建模表，可以给 logistic regression、random forest、LSTM、official 57 模型池使用。

`pipeline` 不应该关心某个模型内部怎么实现。它只负责按时间顺序读取材料、构造窗口、训练、预测、评估和写报告。

## 3. Datasets 是什么

`datasets` 不是简单等于 CSV 文件。它是一类可被操作的材料，至少包含：

- 文件或数据来源。
- 字段 schema。
- 时间粒度和日期列。
- 可用特征列。
- 目标相关列和需要排除的列。
- 是否允许进入 git。
- 数据来源和更新时间说明。

在本仓库里，材料层放在：

```text
corn_forecast/datasets/
```

玉米材料层放在：

```text
corn_forecast/datasets/corn/
  raw/
  processed/
  factors/
  prediction_library/
  metadata/
```

各目录含义：

| directory | meaning |
| --- | --- |
| `raw/` | 原始或近似原始材料，例如原始价格、外部来源数据说明。 |
| `processed/` | 清洗后可建模材料，例如月度玉米建模表。 |
| `factors/` | 因子材料说明，例如期货、现货、基差、天气、季节性、新闻 PCA。 |
| `prediction_library/` | 已完成的 rolling prediction streams，例如 `all_rolling_predictions.csv`。 |
| `metadata/` | schema、字段解释、材料来源和 lineage 文档。 |

旧的根目录 `datasets/` 已收口到 `corn_forecast/datasets/corn/`。玉米相关 CSV
按 raw / processed / prediction_library 等材料类型放入 corn 目录下，配置直接引用
这些 canonical paths。

## 4. Operator 是什么

`operator` 是作用在材料上的方法。它可以是模型，也可以是后续的特征处理算子。

当前已经落地的模型算子层是：

```text
corn_forecast/operator/model/
```

当前结构：

```text
corn_forecast/operator/model/
  base.py
  registry/
  losses/
  wrappers/
  families/
    baseline/
    classical/
    sequence/
    official/
      tabular/
      aeon/
      keras/
    aggregation/
```

各目录含义：

| directory | meaning |
| --- | --- |
| `registry/` | 从 YAML 配置创建模型算子。 |
| `families/baseline/` | 简单基线算子。 |
| `families/classical/` | sklearn、树模型、传统表格模型。 |
| `families/sequence/` | LSTM、GRU、Transformer、PatchTST 等序列模型。 |
| `families/official/` | official 57 模型池，按 tabular、aeon、keras 分组。 |
| `families/aggregation/` | 聚合算子，例如部署投票聚合。 |
| `losses/` | loss-oriented 模型变体。 |
| `wrappers/` | 第三方模型运行适配器。 |

旧路径：

```text
corn_forecast/modeling/
```

只作为兼容入口保留。新模型应该优先进入 `corn_forecast/operator/model/`。

## 5. Aggregation 也是 Operator

部署投票模型不是普通训练模型。

它不从原始特征表重新训练 base model，而是消费 prediction library：

```text
all_rolling_predictions.csv
```

它的输入是很多已经完成的 rolling prediction streams，输出是聚合后的方向预测。

因此它应该被理解为：

```text
aggregation operator over prediction-library datasets
```

当前位置：

```text
corn_forecast/operator/model/families/aggregation/deployment_vote.py
```

关键边界：

- 它消费 `prediction_library/` 材料。
- 它的 `candidate_weights` 表示 forward replacement 选择频次归一化后的权重。
- 它的 `search_protocol=full_history_deployment_discovery` 表示用完整历史预测库选择部署候选。
- 它不能被描述成严格 walk-forward 的自动模型选择结果。

## 6. Pipeline 是什么

`pipeline` 是流程，不是材料，也不是模型方法。

当前位置：

```text
corn_forecast/pipeline/
  train/
  backtest/
  eval/
  report/
```

pipeline 负责：

- 读取配置。
- 调用 data loader。
- 调用 feature 处理逻辑。
- 调用 model registry 创建 operator。
- 维持时间顺序切分。
- 每个窗口训练和预测。
- 汇总 out-of-sample predictions。
- 计算指标。
- 写报告和图表。

pipeline 不应该定义模型家族，不应该保存原始材料，也不应该混入一次性实验输出。

## 7. 一个完整例子

以 `configs/corn.yaml` 为例：

```text
datasets:
  CSV fixture under corn_forecast/datasets/corn/processed/
  documented as corn_monthly_modeling_table

operator:
  last_return baseline
  random_forest classical model

pipeline:
  expanding backtest
  target_known_only=true
  train-only standardization
  metrics and report writing
```

运行时逻辑是：

```text
corn monthly material
  -> target generation
  -> lookback windows
  -> chronological train/test windows
  -> model operator
  -> predictions
  -> evaluation report
```

这里的模型结果不是只由 random forest 决定。它同时取决于：

- 哪份玉米材料被读取。
- 哪些列被当成特征。
- 未来收益和方向如何生成。
- 窗口是否遵守时间顺序。
- operator 如何训练和预测。
- evaluation 如何计算指标。

## 8. 什么不能放进哪里

不放进 `operator`：

- 回测调度。
- 报告输出。
- 大规模实验结果。
- 一次性运行脚本。
- 企业私有数据。

不放进 `datasets`：

- 模型代码。
- 训练循环。
- 评估逻辑。
- 模型权重。
- 大型生成结果。

不放进 `pipeline`：

- 具体模型家族实现。
- 原始材料定义。
- 材料 schema。
- 一次性手工挑模型逻辑。

## 9. 后续演进方向

当前已经完成：

- `operator/model` 模型算子层。
- `datasets` 材料定义层。

后续可以继续推进：

```text
corn_forecast/operator/feature/
  target.py
  windowing.py
  scaling.py
  selection.py
```

这一步会把目标生成、窗口构造、标准化、特征选择从普通 `data_processing/` 逻辑中进一步抽成 feature operator。

再之后可以瘦身：

```text
corn_forecast/pipeline/backtest/
  engine.py
  dataset_builder.py
  window_runner.py
  prediction_collector.py
```

目标是让 `engine.py` 只做编排，让每个层级的职责都可以一眼看懂。

## 10. 企业落地含义

如果未来给企业做玉米采购决策，应该能明确回答：

- 当前预测用了哪份材料。
- 材料有哪些字段。
- 哪些字段可能产生未来信息风险。
- 当前使用了哪个 operator。
- operator 是训练模型还是聚合器。
- 权重或候选模型从哪里来。
- 评估分数是 strict walk-forward，还是 full-history deployment discovery。
- 当前输出能支持什么决策，不能承诺什么收益。

这就是 `operator + datasets` 分层的实际价值：让模型结果可追溯、可解释、可审计，而不是只看一个最终准确率。
