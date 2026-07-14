# 玉米日频因子集 v1

`daily_v1` 直接从 `raw/玉米价格原始数据.csv` 生成，不经过月度聚合，也不替换现有月度、周度或年度因子。当前原始表包含 2,426 个唯一且已排序的 DCE 交易日行，日期范围为 2016-06-29 至 2026-06-26。

## 预测边界

- 使用 DCE 交易日 `t` 结束后可获得的因子，预测下一个实际 DCE 交易日。
- DCE 行情、成交量、持仓量和淀粉-玉米价差使用当日值。
- 基差和 100PPI 缺少可验证的发布时间，统一滞后 1 个 DCE 行。
- 美国市场同一日收盘晚于 DCE，CBOT 字段统一滞后 1 个 DCE 行。
- 实现天气只在日期 `t` 结束后使用。
- 矩阵不含 `target_*`、`next_day` 或 `spike` 列；目标必须由回测流程现场生成。
- 不后向填充，不做双向插值。滚动窗口历史不足和源数据缺口保留为空。

## 30 个候选因子

| 因子族 | 输出 | 计算口径 |
| --- | --- | --- |
| 价格趋势 | `price_momentum_1d/5d/20d` | 当日收盘相对 1/5/20 个 DCE 行前的收益 |
| 价格趋势 | `price_ma_gap_5d/20d/60d` | 当日收盘相对尾随均价的偏离 |
| 风险波动 | `price_range_1d` | `(high-low)/close` |
| 风险波动 | `price_volatility_5d/20d`、`volatility_ratio_5d_20d` | 尾随日收益标准差及短长波动率比 |
| 市场活跃度 | `volume_log_change_1d`、`open_interest_log_change_1d` | 成交量与持仓量的对数变化 |
| 基差 | `basis_rate_level_lag1d`、`basis_rate_change_1d_lag1d`、`basis_rate_zscore_20d_lag1d` | 滞后 1 行的基差率水平、变化和尾随 z-score |
| 期限结构 | `nearby_main_spread_ratio_lag1d`、`nearby_main_spread_change_1d_lag1d` | 滞后 1 行的 100PPI 近月/主力价差比 |
| 加工价差代理 | `starch_corn_spread_ratio`、`starch_corn_spread_change_5d` | 淀粉-玉米期货价差相对玉米价格的比例及 5 行变化 |
| 跨市场 | `cbot_corn_momentum_1d_lag1d`、`cbot_corn_momentum_5d_lag1d` | 滞后后的 CBOT 玉米动量 |
| 跨市场 | `domestic_minus_cbot_momentum_1d_lag1d`、`cbot_wheat_corn_ratio_change_1d_lag1d` | DCE-CBOT 相对动量和滞后小麦/玉米比变化 |
| 实现天气 | `precipitation_sum_5d`、`precipitation_week_vs_month`、`temperature_deviation_20d` | 尾随降水与温度偏离，仅解释已经发生的天气 |
| 日历 | `day_of_year_sin/cos`、`day_of_week_sin/cos` | 年内和周内周期编码 |

这些因子是候选池，不代表模型应一次使用全部 30 个。应在每个滚动训练折内部完成缩放、缩尾、插补和特征筛选，并与仅使用价格历史的基线做增量比较。

## 文件与质量字段

- `factors/matrix/corn_factors_daily_v1.csv`：2,426 行、30 个候选因子的无目标宽表。
- `factors/library/daily_v1/factor_set.yaml`：输入、公式、时间可用性和缺失策略。
- `factors/daily_v1_manifest.json`：输入/输出哈希、日期范围、逐因子缺失数和生成策略。
- `factors/daily_v1_data_gaps.yaml`：当前缺口和后续数据采集优先级。

矩阵中的 `available_factor_count` 给出每行可用因子数；`strict_backtest_eligible=true` 表示该行 30 个因子全部可用；`is_latest_observation` 只标记最后一个源数据行。非严格行可保留用于因子子集实验，但插补器必须只在训练折拟合。

## 当前覆盖情况

生成结果有 2,115 个完整 30 因子行，无常量因子。CBOT 相关输出缺失 118 至 121 行，20 日基差 z-score 缺失 100 行，主要来自原始缺口和滚动窗口传播。首 59 行的 60 日均线因子为空属于必要预热，不是数据错误。两个全空字段 `corn_import_volume_ton_ffill` 和 `reserve_corn_release_volume_ton` 未构造因子。

## 重新生成

```bash
python scripts/build_corn_daily_factors.py
```

脚本会重建宽表、集中式因子定义和 manifest，并校验日期唯一性、因子顺序、目标列边界、早期窗口缺失、常量因子和最新行标记。周度、月度、年度文件及模型配置不会被修改。
