# 双头 LSTM 模型

这个模型来自旧的双头 LSTM 脚本，但已经整理成项目内的标准模型 `dual_stream_lstm`。它不再读取随机森林特征排名文件，也不再固定“前 79 个特征”。

## 特征怎么进入模型

- `feature_cols: auto_numeric` 会先选出 CSV 中可用的数值列。
- `exclude_feature_cols` 会排除未来目标和旧标签列，例如 `target_*`、`dce_corn_close_next_month*`、`spike`。
- 列名匹配 `pca_001`、`pca_002`、`PCA001`、`PCA_001` 这类格式的特征会进入新闻/PCA 分支。
- 其他数值特征会进入结构化行情分支。

两个分支各自经过 LSTM。PCA 分支再经过一个简单 self-attention 汇总，然后和结构化分支拼接，最后输出上涨概率。

## 目标变量

默认配置没有直接使用 CSV 里的 `spike` 作为目标，而是从 `price_col: dce_corn_close` 重新计算：

```text
target_return_fwd = future_price / current_price - 1
target_direction_fwd = 1 if target_return_fwd > spike_threshold else 0
```

在 `configs/corn_dual_stream_lstm.yaml` 里，`horizon: 1`、`spike_threshold: 0.0`，所以目标是预测下一个月玉米价格是否上涨。

## 怎么运行

先确认已经安装 PyTorch：

```bash
pip install -e .[deep]
```

然后运行双头 LSTM 配置：

```bash
commodity-backtest run --config configs/corn_dual_stream_lstm.yaml --output-dir experiments/corn_dual_stream_lstm
```

如果不想安装命令行入口，也可以用 Python 模块方式：

```bash
python -m cli run --config configs/corn_dual_stream_lstm.yaml --output-dir experiments/corn_dual_stream_lstm
```

结果会写入：

```text
experiments/corn_dual_stream_lstm/
  comparison.csv
  report.md
  data_manifest.json
  model_outputs/
    dual_stream_lstm/
      predictions.csv
      metrics_summary.json
      equity_curve.png
      rolling_dir_acc.png
      rolling_sharpe.png
```

`predictions.csv` 可以查看每个测试月份的真实标签、预测标签、预测概率和训练窗口日期。
