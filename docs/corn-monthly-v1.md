# 玉米月度数据集 v1

本次新增一套不修改旧数据、可从仓库内材料重复生成的玉米月度特征表。

## 新增文件

- `corn_forecast/datasets/corn/processed/corn_monthly_core_v1.csv`
  - 由 `raw/玉米价格原始数据.csv` 重新聚合。
  - 121 个月，94 列。
  - 包含行情、成交量、持仓、现货、基差、天气和历史滚动指标。
  - 不包含 PCA、未来目标和旧 `spike` 标签。
- `corn_forecast/datasets/corn/processed/corn_monthly_news_legacy.csv`
  - 从旧无缺失月度表提取 `pca_001` 至 `pca_032`。
  - 121 个月，37 列，其中 32 列为 PCA 特征。
  - 标记为 `legacy_unknown_provenance`，默认不具备严格回测资格。
- `corn_forecast/datasets/corn/processed/corn_monthly_v1_manifest.json`
  - 记录输入文件哈希、数据截止日、行列数、不完整月份和生成策略。
- `scripts/build_corn_monthly_dataset.py`
  - 可重复生成上述文件并执行数据边界校验。

## 主要修改

1. 保留现有原始 CSV 和两份旧月度 CSV，不覆盖、不改名。
2. 从原始日频数据重新计算月度 OHLC、均值、标准差、成交量、持仓量、收益和振幅。
3. 从原始日频数据重新聚合现货、基差、CBOT、天气和收获季字段。
4. 重新计算 1、3、6、12 月收益、均线和波动率。
5. 滚动窗口不足时保留缺失值，不使用向后填充或双向插值。
6. 将首个来源月份 `2016-06` 和最新不完整月份 `2026-06` 标记为 `is_complete_period=false`。
7. 将 32 个旧 PCA 特征与核心数据拆开，避免默认混入严格回测。
8. 不把 `next_month`、`target_*` 或 `spike` 写入特征表；目标由回测流程按 horizon 生成。

## 生成命令

```bash
python scripts/build_corn_monthly_dataset.py
```

也可以显式指定输入和输出目录：

```bash
python scripts/build_corn_monthly_dataset.py \
  --raw path/to/raw_daily.csv \
  --legacy path/to/legacy_monthly.csv \
  --output-dir corn_forecast/datasets/corn/processed
```

## 使用策略

严格回测默认使用 `corn_monthly_core_v1.csv`，并排除
`is_complete_period=false` 的月份。旧 PCA 表仅用于单独的增量实验；在补齐原始新闻、发布时间和折内 PCA 生成流程之前，不应把其结果表述为严格无泄漏结果。

模型目标继续由框架根据 `dce_corn_close` 生成。不要把目标列重新写回核心特征表。

## 验证结果

- 月份键唯一且按时间升序。
- 核心表和 PCA 表覆盖相同的 121 个月。
- 核心表不含 PCA 和目标列。
- PCA 表包含 32 个 PCA 特征并标记为非严格回测数据。
- 完整月份的月度聚合值与旧 `玉米价格月度_混合特征版 .csv` 一致。
- 3、6、12 月滚动指标的早期缺失值保持为空，没有未来回填。

