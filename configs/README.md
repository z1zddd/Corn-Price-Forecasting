# Configs

这里存放 YAML 实验配置。配置只描述数据路径、目标生成、窗口切分、模型池和评估参数，不放训练结果或本地临时路径。

- `template.yaml`: 新商品配置模板，默认指向 `corn_forecast/datasets/corn/processed/corn_sample_data.csv`。
- `corn.yaml`: 玉米基准配置。
- `corn_official_pool_57_*.yaml`: 官方 57 模型池配置，区分预测 horizon 和是否包含新闻 PCA 特征。
- `soybean.yaml`、`rebar.yaml`: 其他商品的参考配置。

