# 玉米价格预测回测框架

本项目是一个由 YAML 配置驱动的多商品时间序列预测回测框架。它由玉米价格基准实验代码整理而来，默认先支持玉米价格预测，也可以通过更换配置扩展到大豆、螺纹钢等其他商品。

框架会读取商品 CSV 数据，根据配置中的价格列生成未来收益和涨跌方向目标，按时间顺序构造窗口，运行启用的模型，评估预测指标和交易指标，并输出适合人工阅读和自动化流程使用的实验报告。

## 主要功能

- 支持按商品维护独立的 YAML 配置。
- 保持时间序列顺序，不使用随机打乱的训练/测试切分。
- 从 `price_col` 自动生成目标变量，不依赖预先写好的标签。
- 支持 expanding、rolling、capped expanding 三类回测窗口。
- 标准化器只在每个 rolling 窗口的训练集上 `fit`，并可从训练集尾部切出验证集。
- 内置基线模型、scikit-learn 模型、基准损失变体，并支持可选 PyTorch 序列模型。
- 输出 `comparison.csv`、`report.md`、`agent_verdict.json`、各模型预测结果、指标、权益曲线图、`rolling_dir_acc.png` 和 `rolling_sharpe.png`。
- 包含配置校验、数据处理、切分模式、模型、指标、报告、CLI 命令和仓库边界约束。

## 安装

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e .
```

如果使用 macOS 或 Linux，请把 `.venv\Scripts\python` 换成 `.venv/bin/python`。

## 快速开始

```bash
commodity-backtest diagnose --csv examples/corn/sample_data.csv --date-col date
commodity-backtest diagnose --config configs/corn.yaml
commodity-backtest auto-window --config configs/corn.yaml
commodity-backtest build-config --base-config configs/template.yaml --output configs/my_commodity.yaml --commodity-name my_commodity --csv local_data/my.csv --date-col date --price-col close
commodity-backtest run --config configs/corn.yaml
commodity-backtest run-lookbacks --config configs/corn.yaml
commodity-backtest compare --experiment experiments/manual_run
commodity-backtest interpret --experiment experiments/manual_run
```

默认运行结果会写入 `experiments/manual_run/`。该目录已被 git 忽略，用于避免把本地实验产物提交到仓库。

## 项目结构

```text
corn_forecast/          框架主包
  cli.py                命令行入口
  config/               YAML 配置读取与校验
  data/                 CSV 读取、目标生成、特征选择和窗口构造
  pipeline/             回测、训练、评估和报告流程
  modeling/             模型、模型池、损失变体、registry、wrapper 和 ensemble
configs/                示例和官方实验配置
examples/               示例数据
datasets/               小型仓库内数据资产
scripts/                实验、搜索、验证和维护脚本
docs/                   架构、配置和实验说明
local_data/             本地原始数据目录，默认不提交到 git
```

## 输出结构

执行 `commodity-backtest run --config configs/corn.yaml` 后，实验目录大致如下：

```text
experiments/manual_run/
  agent_verdict.json
  comparison.csv
  data_manifest.json
  report.md
  model_outputs/
    <best_model>/
      equity_curve.png
      metrics_summary.json
      predictions.csv
      rolling_dir_acc.png
      rolling_metrics.csv
      rolling_sharpe.png
```

每个启用的模型都会写入自己的 `model_outputs/<model>/` 子目录。权益曲线图同时包含策略曲线和买入持有曲线。

## 切换到其他商品

1. 将新的 CSV 放到 `local_data/` 或其他本地路径。`local_data/` 默认被 git 忽略，避免和代码模块 `data/` 混淆。
2. 复制 `configs/template.yaml`，也可以从 `configs/soybean.yaml` 或 `configs/rebar.yaml` 开始修改。
3. 更新 `commodity.name`、`data.csv_path`、`data.date_col`、`data.price_col` 和特征设置。
4. 运行 `commodity-backtest diagnose --csv <path> --date-col <date_col>` 检查数据。
5. 运行 `commodity-backtest run --config configs/soybean.yaml` 启动回测。

## 模型

内置可直接运行的模型：

- `last_return`
- `mean_return` / `mean_direction`
- `logistic_regression`
- `random_forest`
- `regression_mse_sign`
- `regression_mae_sign`
- `regression_huber_sign`
- `dual_head_mse_bce`

安装对应依赖后可使用的树模型：

- `lightgbm`
- `xgboost`
- `catboost`

安装 `deep` 可选依赖后可使用的 PyTorch 模型：

- `focal_logistic`
- `lstm`
- `gru`
- `transformer`
- `patchtst`
- `itransformer`
- `dlinear`
- `dual_stream_lstm`

安装可选树模型依赖，可使用任一写法：

```bash
pip install -e .[trees]
pip install -e .[tree]
```

安装可选深度学习依赖：

```bash
pip install -e .[deep]
```

## 设计约束

- 不打乱时间序列数据。
- 目标变量必须从配置指定的价格列生成。
- 原始数据、实验输出、模型权重和压缩包不进入 git。
- 回测结果只能作为研究证据，不能被表述为实盘收益承诺。

## 文档

- [架构说明](docs/architecture.md)
- [配置说明](docs/configuration.md)
- [指标说明](docs/metrics.md)
- [双头 LSTM 模型](docs/dual-stream-lstm.md)
- [57 模型池](docs/official-model-pool.md)
- [无泄漏组合搜索](docs/ensemble-search.md)
- [Agent 工作流](docs/agent-workflow.md)

## 验证

需要做本地验证时，优先使用：

```bash
commodity-backtest --help
python -m corn_forecast.cli --help
commodity-backtest diagnose --config configs/corn.yaml
```
