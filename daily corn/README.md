# Daily Corn

玉米日度多步长价格预测项目目录。

## 目录

- `configs/`：每个模型一个可编辑 YAML 配置。
- `datasets/`：原始数据和处理后数据。
- `models/`：基线、机器学习、深度学习和组合模型。
- `scripts/`：数据处理、实验运行和结果评价脚本。
- `results/`：预测、指标和本次实际配置。
- `report/`：每次实验的 Markdown 实验报告。
- `checkpoints/`：模型权重和训练状态。
- `tests/`：自动检查脚本。
- `docs/`：回测框架文档。

## 实验目录约定

```text
results/<数据名>/<模型名>/<YYYY-MM-DD-HH-MM-SS>/
report/<数据名>/<模型名>/<YYYY-MM-DD-HH-MM-SS>/report.md
checkpoints/<数据名>/<模型名>/<YYYY-MM-DD-HH-MM-SS>/
```

每个结果目录至少保存：

```text
config_resolved.yaml
predictions.csv
metrics.csv
```

数据、结果和权重允许纳入版本管理；提交前应检查文件大小和内容。
