# 玉米月度因子集 v1

`monthly_v1` 是基于 `corn_monthly_core_v1.csv` 重新定义的月度候选因子集。它与旧因子材料并行保存，不覆盖旧月度矩阵，也不修改周度和年度因子。

## 使用边界

- 用月度 `t` 的因子预测月度 `t+1`，目标由回测流程现场生成。
- 新矩阵不包含 `target_*`、`next_month` 或 `spike` 列。
- `is_complete_period=false` 的月份不能进入严格回测。
- 现货、基差和 100PPI 字段缺少可验证的发布时间，因此临时滞后 1 个月。
- 已实现天气异常只使用同一历月的过去年份建立基准，不使用全样本气候均值。
- 历史不足产生的缺失值保持为空，不后向填充，不做双向插值。
- 缩放、缩尾、插补和特征选择只能在每个训练折内部拟合。

## 候选因子

| 因子族 | 输出 | 来源与处理 |
| --- | --- | --- |
| 价格趋势 | `price_momentum_1m`、`price_momentum_3m`、`price_ma_gap_6m` | DCE 玉米月末收盘价；当月月末可用 |
| 风险波动 | `price_range_1m`、`price_volatility_3m`、`volatility_ratio_3m_12m` | DCE 月度高低价和月收益率 |
| 市场活跃度 | `volume_log_change_1m`、`open_interest_log_change_1m` | DCE 月成交量与月末持仓量的对数变化 |
| 基差紧张度 | `basis_rate_level_lag1`、`basis_rate_zscore_12m_lag1` | 基差率；统一滞后 1 个月，z-score 只用过去 12 个月 |
| 期限结构 | `nearby_main_spread_ratio_lag1`、`nearby_main_spread_change_1m_lag1` | 100PPI 近月/主力价格比；统一滞后 1 个月 |
| 加工价差代理 | `starch_corn_spread_ratio` | 淀粉-玉米期货价差/玉米价格；不是实际加工利润或加工需求 |
| 跨市场支持 | `cbot_corn_momentum_1m`、`domestic_minus_cbot_momentum_1m` | CBOT 玉米动量及 DCE-CBOT 相对动量 |
| 已实现天气异常 | `precip_anomaly_same_month`、`temperature_anomaly_same_month`、`hot_dry_weather_stress` | 东北三省区月度降水和温度；仅用同历月过去年份，至少 3 个历史观测 |
| 收获季代理 | `harvest_season_share` | 日历型收获季占比；不是实测收获进度 |
| 季节性 | `seasonal_sin`、`seasonal_cos` | 月份正余弦编码 |

共 21 个候选因子。它们是供后续比较和筛选的候选池，不代表模型必须一次使用全部因子。当前样本只有 119 个完整月份，建议在滚动回测的训练折内把最终输入控制在 6 至 10 个，并与“仅价格历史”的基线做增量比较。

## 文件结构

```text
corn_forecast/datasets/corn/factors/
  library/monthly_v1/<factor_family>/
    factor.yaml
    values.csv
  matrix/corn_factors_monthly_v1.csv
  monthly_v1_manifest.json
  monthly_v1_data_gaps.yaml
```

每个 `factor.yaml` 记录输入、公式、最低历史长度、滞后和可用规则；`values.csv` 是带 `asof_date` 与质量标志的长表；`corn_factors_monthly_v1.csv` 是无目标列的宽表；清单记录输入和输出哈希。

## 重新生成

```bash
python scripts/build_corn_monthly_factors.py
```

脚本会重建 10 个因子族的定义和值、宽矩阵及 manifest，并检查月份唯一性、目标列边界、早期缺失、常量因子和供给压力禁用状态。

## 当前不应构造的因子

`supply_pressure` 暂不启用。现有数据没有库存、仓单、港口库存、种植面积、单产或产量预期，强行构造只会得到没有经济含义的常量或价格替代变量。完整缺口和新增数据优先级见 `monthly_v1_data_gaps.yaml`。

旧 `corn_monthly_news_legacy.csv` 的 PCA 字段也没有进入本因子集。只有找回原始新闻、发布时间和折内 PCA 生成流程后，才能作为严格回测候选。
