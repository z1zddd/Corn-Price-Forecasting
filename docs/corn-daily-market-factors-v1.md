# 玉米产业链日频因子集 v1

`daily_market_v1` 使用外部的 `raw_quotes.csv` 与 `normalized_prices.csv` 生成短历史产业链因子。两份源文件拥有相同的 10,285 个唯一 `记录ID`：原始表负责来源和文本审计，标准化表负责价格聚合。源明细不提交到公开仓库，仓库只保存生成脚本、源文件哈希、定义和聚合因子。

## 使用边界

- 数据源日期为 2025-07-14 至 2026-07-09；与仓库 DCE 日历的有效交集截至 2026-06-26。
- 过滤条件为 `是否可训练=yes`、状态属于 `normalized/price_parsed/accepted`、单位为元/吨且价格为正。
- 当前导出中 22 条待复核记录被排除。
- 当前所有记录标示为 08:00 发布。脚本仍实现通用的 15:00 截止规则：收盘后发布和非交易日数据映射到下一 DCE 行。
- 不做前向填充、后向填充或双向插值。
- 历史抓取时间缺少完整年份和数据版本，无法证明每条历史记录在标示发布日期就已经可得。因此 `point_in_time_verified` 和 `strict_backtest_eligible` 固定为 false。
- 本因子集只用于影子回测、残差增强模型和未来实时监控，不替换长期 `daily_v1` 因子。

## 12 个候选因子

| 因子族 | 输出 | 说明 |
| --- | --- | --- |
| 玉米现货 | `corn_spot_momentum_5d`、`corn_spot_momentum_20d` | 玉米报价日中位数的 5/20 个 DCE 行动量 |
| 玉米现货 | `corn_spot_dce_basis` | 玉米现货中位数相对 DCE 玉米收盘价的基差率 |
| 地区离散 | `corn_regional_dispersion` | 至少 3 条报价时的 `IQR/中位数` |
| 报价质量 | `corn_quote_count_log`、`corn_source_count_log` | 玉米有效报价数与独立来源数的 `log1p` |
| 报价质量 | `corn_confidence_mean` | 当日玉米报价平均置信度 |
| 加工价差 | `starch_corn_spread_ratio`、`starch_corn_spread_change_5d` | 玉米淀粉/玉米现货价差率及其 5 行变化 |
| 副产品 | `byproduct_momentum_5d`、`byproduct_momentum_20d` | 玉米皮、胚芽、蛋白粉价格动量的等权均值，至少 2 个产品 |
| 深加工链 | `processing_chain_momentum_20d` | 葡萄糖、果糖、糖浆和麦芽糊精等 20 行动量均值，至少 3 个产品 |

短周期 1 日价格动量没有进入第一版。真实数据审计显示阶梯式报价会产生过多零值；改用 5/20 行窗口后，零值占比明显下降。

## 输出结构

```text
scripts/build_corn_market_daily_factors.py
tests/test_build_corn_market_daily_factors.py
corn_forecast/datasets/corn/factors/
  library/daily_market_v1/factor_set.yaml
  matrix/corn_market_daily_factors_v1.csv
  daily_market_v1_manifest.json
  daily_market_v1_data_gaps.yaml
```

矩阵覆盖与 `corn_factors_daily_v1.csv` 相同的 2,426 个 DCE 行。报价历史开始前的产业链因子保持为空；`shadow_backtest_eligible=true` 表示当日至少有一半候选因子可用。矩阵不包含目标列。

## 本地生成

将两份源文件放入被 `.gitignore` 排除的目录：

```text
local_data/corn_market/raw_quotes.csv
local_data/corn_market/normalized_prices.csv
```

然后运行：

```bash
python scripts/build_corn_market_daily_factors.py
```

也可以显式指定外部路径：

```bash
python scripts/build_corn_market_daily_factors.py \
  --raw-quotes /path/to/raw_quotes.csv \
  --normalized /path/to/normalized_prices.csv
```

脚本会验证两份源文件的 ID 集合一致，生成定义、矩阵和 manifest。manifest 只记录文件名、SHA-256、行数、授权状态统计和输出质量，不记录用户本机绝对路径。

## 辅助训练

现阶段推荐使用两层结构：

1. 基础模型继续使用 2016 年至今的 `corn_factors_daily_v1.csv`。
2. 对基础模型生成严格样本外预测。
3. 在 2025 年以来的共同区间，用“基础样本外预测 + `daily_market_v1` 因子”训练低复杂度残差模型。
4. 最终预测为 `base_prediction + residual_adjustment`。
5. 对比仅基础因子、直接合并和残差增强三种方案；只有多个滚动窗口稳定改善时才升级为正式特征。

增强模型优先使用 Ridge、ElasticNet 或小型树模型，不建议用约 231 个交易日单独训练 LSTM。所有插补、缩放和特征选择必须在每个训练折内部拟合。

## 仍需解决的问题

9,785 条记录的来源授权为 `unknown`，499 条为 `no`，只有 1 条为 `yes`。历史 API 数据也缺少完整抓取年份、修订时间和 vintage 标识。完整限制与后续采集要求见 `daily_market_v1_data_gaps.yaml`。
