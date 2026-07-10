# 玉米因子库储存设计建议

> 落地状态：本目录已按本文建议拆分为 `library/` 和 `matrix/` 两层。原三张
> `corn_factors_*.csv` 宽表已移动到 `matrix/`，单因子定义和值表位于
> `library/<factor_id>/factor.yaml` 和 `library/<factor_id>/values.csv`。

本文用于说明玉米预测项目中因子材料应该如何储存、拆分和维护。核心观点是：不要把“每个因子”简单理解成“每个因子一个随手 CSV”。更稳妥的方式是参考开源量化因子库，把因子体系拆成三层：

1. **因子定义**：记录因子怎么算、使用哪些原始字段、是否需要滞后、如何防止未来函数。
2. **因子值库**：记录每个时间点、每个标的、每个频率下的因子值和质量信息。
3. **训练矩阵**：把多个因子拼成宽表，供模型训练、回测和诊断直接读取。

## 开源量化库给我们的启发

- Qlib 更接近“数据层 + 特征表达式 + Dataset”的模式。基础数据、特征生成、模型输入是分层的，不建议把所有东西混进一张表里。
- Alphalens 做因子评价时，会把单个 factor 与 forward returns 对齐。它提醒我们：因子值、未来收益、标签必须分清楚，否则容易引入未来函数。
- vectorbt 的 indicator/factory 思路更像是把指标或因子当作可复用组件。每个因子应该有定义、参数和输出，而不是只留下一个结果列。

参考资料：

- [Qlib Data Layer](https://qlib.readthedocs.io/en/v0.9.7/component/data.html)
- [Alphalens API](https://alphalens.ml4trading.io/api-reference.html)
- [vectorbt IndicatorFactory](https://vectorbt.dev/api/indicators/factory/)

## 当前三张表的定位

当前目录中的三张表：

- `corn_factors_monthly.csv`
- `corn_factors_weekly.csv`
- `corn_factors_yearly.csv`

更像是**因子宽表**或**训练矩阵候选表**，不应该被视为最终形态的“因子库本体”。

它们的优点是：

- 直观，方便人工查看。
- 适合快速训练和 smoke test。
- 容易和现有配置、模型输入对接。

它们的问题是：

- 因子定义不清楚。
- 因子值、目标列、覆盖率字段混在同一张表里。
- 不容易单独评估某个因子的 IC、覆盖率、稳定性和泄漏风险。
- 未来新增更多因子时，宽表会越来越难维护。

因此建议保留宽表，但把它放在 `matrix/` 层；真正的因子库应该再拆出 `library/` 层。

## 推荐目录结构

建议将 `corn_forecast/datasets/corn/factors/` 逐步整理成下面的结构：

```text
corn_forecast/datasets/corn/factors/
  README.md
  factor_storage_design.zh.md
  registry.yaml

  library/
    price_momentum/
      factor.yaml
      values.csv
    basis_tightness/
      factor.yaml
      values.csv
    processing_demand/
      factor.yaml
      values.csv
    market_activity/
      factor.yaml
      values.csv
    external_support/
      factor.yaml
      values.csv
    weather_anomaly/
      factor.yaml
      values.csv
    supply_pressure/
      factor.yaml
      values.csv
    risk_volatility/
      factor.yaml
      values.csv
    harvest_pressure/
      factor.yaml
      values.csv
    seasonal/
      factor.yaml
      values.csv

  matrix/
    corn_factors_monthly.csv
    corn_factors_weekly.csv
    corn_factors_yearly.csv
```

### `library/`

`library/` 是真正的因子库。每个子目录对应一个逻辑因子或一个强相关的因子组。

例如：

- `price_momentum/`：价格动量因子。
- `basis_tightness/`：基差紧张度因子。
- `weather_anomaly/`：天气异常因子。
- `seasonal/`：季节性编码因子。

每个因子目录至少包含：

- `factor.yaml`：因子定义。
- `values.csv`：因子值长表。

### `matrix/`

`matrix/` 存放模型可以直接读取的宽表。

这些表是从 `library/` 中的单因子值拼接出来的训练矩阵，可以继续保留当前三张：

- `corn_factors_monthly.csv`
- `corn_factors_weekly.csv`
- `corn_factors_yearly.csv`

这层的定位是“方便训练”，不是“因子定义的唯一来源”。

### `registry.yaml`

`registry.yaml` 用于维护因子清单，告诉系统当前有哪些因子、在哪个目录、默认是否启用、属于哪个频率或分组。

示例：

```yaml
factors:
  price_momentum:
    path: library/price_momentum
    group: price
    default_enabled: true
    frequencies: [weekly, monthly, yearly]

  basis_tightness:
    path: library/basis_tightness
    group: basis
    default_enabled: true
    frequencies: [weekly, monthly]

  seasonal:
    path: library/seasonal
    group: calendar
    default_enabled: true
    frequencies: [weekly, monthly, yearly]
```

## 单因子值表格式

每个 `values.csv` 建议使用长表，而不是每个频率单独一个宽表。

推荐字段：

```text
period_end,period_key,frequency,instrument,factor_id,value,coverage
```

示例：

```text
2024-01-31,2024-01,monthly,corn,price_momentum,0.23,1.0
2024-02-29,2024-02,monthly,corn,price_momentum,0.18,1.0
2024-W05,2024-W05,weekly,corn,price_momentum,0.07,0.8
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `period_end` | 因子所属周期结束日期。 |
| `period_key` | 周期键，例如 `2024-01` 或 `2024-W05`。 |
| `frequency` | 频率，例如 `weekly`、`monthly`、`yearly`。 |
| `instrument` | 标的，例如 `corn`。未来可以扩展到 `corn_starch`、`soybean` 等。 |
| `factor_id` | 因子 ID，例如 `price_momentum`。 |
| `value` | 因子值。 |
| `coverage` | 覆盖率或数据完整度。没有覆盖率时可以为空。 |

如果后续要更严格地防止未来函数，可以增加：

| 字段 | 含义 |
| --- | --- |
| `asof_date` | 该因子在真实世界中可被观察到的日期。 |
| `source_version` | 原始数据或计算脚本版本。 |
| `quality_flag` | 数据质量标记，例如 `ok`、`missing_source`、`partial_period`。 |

其中最重要的是 `asof_date`。它回答一个关键问题：模型在某一天做预测时，是否真的已经知道这个因子值？

## 单因子定义文件格式

每个因子目录中的 `factor.yaml` 用于说明因子如何计算。

示例：

```yaml
id: price_momentum
name: 价格动量因子
group: price
description: 衡量玉米价格在最近窗口内的趋势强弱。
frequencies:
  - weekly
  - monthly
  - yearly
inputs:
  - dce_corn_close
calculation:
  method: derived_from_existing_matrix
  expression: factor_price_momentum
leakage_control:
  uses_future_target: false
  requires_lag: true
  asof_policy: period_end
quality:
  coverage_column: factor_price_momentum_coverage
output:
  value_column: factor_price_momentum
```

这个文件不一定一开始就要非常复杂，但至少应该说明：

- 因子 ID。
- 因子中文名。
- 因子分组。
- 因子输入字段。
- 因子计算方式。
- 是否使用未来信息。
- 是否需要滞后。
- 对应覆盖率字段。

## 当前字段如何拆分

当前三张宽表中的字段可以先按下面方式理解。

### 索引和周期信息

这些字段是周期元信息，不是因子：

- `period_end`
- `period_key`
- `frequency`
- `period_start`
- `period_n_obs`
- `period_is_partial`

### 基础价格字段

基础价格可以作为原始观测或构造因子的输入，不建议直接混为某个因子：

- `dce_corn_close`

### 目标和标签字段

这些字段是预测目标或标签，不应该进入单因子库：

- `target_date_fwd`
- `target_price_fwd`
- `target_return_fwd`
- `target_direction_fwd`

这类字段更适合放在训练矩阵或标签构造结果中，不能作为模型特征误用。

### 真正的因子字段

这些字段可以拆进 `library/`：

- `factor_price_momentum`
- `factor_basis_tightness`
- `factor_processing_demand`
- `factor_market_activity`
- `factor_external_support`
- `factor_weather_anomaly`
- `factor_supply_pressure`
- `factor_risk_volatility`
- `factor_harvest_pressure`

对应目录建议为：

```text
factor_price_momentum      -> library/price_momentum/
factor_basis_tightness     -> library/basis_tightness/
factor_processing_demand   -> library/processing_demand/
factor_market_activity     -> library/market_activity/
factor_external_support    -> library/external_support/
factor_weather_anomaly     -> library/weather_anomaly/
factor_supply_pressure     -> library/supply_pressure/
factor_risk_volatility     -> library/risk_volatility/
factor_harvest_pressure    -> library/harvest_pressure/
```

### 因子质量字段

这些字段应该跟对应因子一起存放：

- `factor_price_momentum_coverage`
- `factor_basis_tightness_coverage`
- `factor_processing_demand_coverage`
- `factor_market_activity_coverage`
- `factor_external_support_coverage`
- `factor_weather_anomaly_coverage`
- `factor_supply_pressure_coverage`
- `factor_risk_volatility_coverage`

例如 `factor_price_momentum_coverage` 应该进入 `library/price_momentum/values.csv` 的 `coverage` 列。

### 季节性字段

这些字段属于日历或季节性因子组：

- `seasonal_sin`
- `seasonal_cos`

建议放在：

```text
library/seasonal/
```

如果一个目录内有两个输出列，可以在 `values.csv` 中通过 `factor_id` 区分：

```text
period_end,period_key,frequency,instrument,factor_id,value,coverage
2024-01-31,2024-01,monthly,corn,seasonal_sin,0.50,
2024-01-31,2024-01,monthly,corn,seasonal_cos,0.87,
```

## 推荐落地节奏

不建议一次性把所有数据结构重构得很重。可以分三步走。

### 第一步：保留宽表，明确定位

先把当前三张宽表移动到：

```text
corn_forecast/datasets/corn/factors/matrix/
```

并在 README 中明确说明：

- 这些是模型输入候选矩阵。
- 它们不是因子定义的唯一来源。
- 目标列只能用于标签和评估，不能作为特征。

### 第二步：新增因子定义

新增：

```text
registry.yaml
library/*/factor.yaml
```

先不急着拆 `values.csv`，也可以先只写定义。

这样做的好处是：先把“每个因子是什么”说清楚。

### 第三步：生成单因子长表

从当前宽表中拆出每个因子的 `values.csv`。

例如从 `corn_factors_monthly.csv`、`corn_factors_weekly.csv`、`corn_factors_yearly.csv` 中抽取 `factor_price_momentum` 和 `factor_price_momentum_coverage`，合并成：

```text
library/price_momentum/values.csv
```

再由这些单因子长表生成训练宽表。

## 关键原则

1. **因子定义和因子值分开**：`factor.yaml` 管定义，`values.csv` 管数值。
2. **因子值和标签分开**：`target_*` 字段不能进入单因子库。
3. **因子库和训练矩阵分开**：`library/` 是源头，`matrix/` 是给模型用的拼接结果。
4. **宽表可以保留**：宽表对训练很友好，但不要让它承担全部语义。
5. **覆盖率跟因子走**：`factor_*_coverage` 应该跟对应因子放在一起。
6. **频率用字段表达**：不要为 monthly、weekly、yearly 复制三套结构，优先用 `frequency` 字段区分。
7. **未来函数要显式管理**：后续最好补充 `asof_date` 或等价字段，记录因子真实可见时间。

## 结论

玉米因子库建议采用“因子定义 + 因子值库 + 训练矩阵”三层结构。

当前三张 `corn_factors_*.csv` 应该先作为 `matrix/` 层保留；真正的因子库应逐步拆成 `library/<factor_id>/factor.yaml` 和 `library/<factor_id>/values.csv`。

这样既保留了现有训练便利性，也为后续做因子评价、IC 分析、覆盖率审计、泄漏检查和多频率扩展留下了空间。
