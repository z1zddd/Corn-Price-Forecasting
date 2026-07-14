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
commodity-backtest diagnose --csv corn_forecast/datasets/corn/processed/corn_sample_data.csv --date-col date
commodity-backtest diagnose --config configs/corn.yaml
commodity-backtest auto-window --config configs/corn.yaml
commodity-backtest build-config --base-config configs/template.yaml --output configs/my_commodity.yaml --commodity-name my_commodity --csv local_data/my.csv --date-col date --price-col close
commodity-backtest run --config configs/corn.yaml
commodity-backtest run-lookbacks --config configs/corn.yaml
commodity-backtest compare --experiment experiments/manual_run
commodity-backtest interpret --experiment experiments/manual_run
```

默认运行结果会写入 `experiments/manual_run/`。该目录已被 git 忽略，用于避免把本地实验产物提交到仓库。

## 玉米月度数据集 v1

仓库提供一套从日频原始数据重新聚合、避免未来回填的月度特征材料：

- `corn_forecast/datasets/corn/processed/corn_monthly_core_v1.csv`：推荐用于新实验的核心月度特征，不含 PCA 和未来目标列。
- `corn_forecast/datasets/corn/processed/corn_monthly_news_legacy.csv`：从旧月度表隔离出的 32 个 PCA 特征，仅用于单独的增量对照实验。
- `corn_forecast/datasets/corn/processed/corn_monthly_v1_manifest.json`：记录输入哈希、数据截止日、不完整月份和生成策略。

使用前应排除 `is_complete_period=false` 的月份，并由回测流程根据 `dce_corn_close` 和预测期限生成目标。当前 `configs/corn.yaml` 和官方模型池配置仍指向旧月度数据，尚未自动切换到 v1 数据集。

运行 `python scripts/build_corn_monthly_dataset.py` 可以重复生成这套数据。详细字段、限制和验证结果见 [玉米月度数据集 v1 说明](docs/corn-monthly-v1.md)。

## 玉米月度因子集 v1

仓库基于 `corn_monthly_core_v1.csv` 新增了一个与旧因子材料并行的月度候选因子集：10 个因子族、21 个候选因子。宽表位于 `corn_forecast/datasets/corn/factors/matrix/corn_factors_monthly_v1.csv`，不包含任何未来目标列；各因子定义和值位于 `corn_forecast/datasets/corn/factors/library/monthly_v1/`。

现货、基差和 100PPI 字段在补齐发布时间戳前统一滞后 1 个月；天气异常只使用同一历月的过去年份；历史不足保留缺失值。现有旧月度因子、周度因子、年度因子和模型配置均未切换。运行 `python scripts/build_corn_monthly_factors.py` 可重复生成，详见 [玉米月度因子集 v1 说明](docs/corn-monthly-factors-v1.md)。

## 玉米日频因子集 v1

仓库从 `raw/玉米价格原始数据.csv` 直接生成独立的日频候选因子集：9 个因子族、30 个候选因子、2,426 个 DCE 交易日行。无目标宽表位于 `corn_forecast/datasets/corn/factors/matrix/corn_factors_daily_v1.csv`，集中式定义位于 `corn_forecast/datasets/corn/factors/library/daily_v1/factor_set.yaml`。

日频口径使用交易日 `t` 结束后可获得的信息预测下一个实际 DCE 交易日。基差、100PPI 和 CBOT 字段保守滞后 1 个 DCE 行；滚动预热和原始缺口保持为空，不做后向填充。该候选集尚未自动接入模型配置，也未修改现有月度、周度和年度因子。运行 `python scripts/build_corn_daily_factors.py` 可重复生成，详见 [玉米日频因子集 v1 说明](docs/corn-daily-factors-v1.md)。

## 玉米产业链日频因子集 v1

仓库新增 `daily_market_v1` 短历史候选因子集，由外部 `raw_quotes.csv` 和 `normalized_prices.csv` 在本地生成。两份源明细因授权状态和记录级来源信息不提交到公开仓库；仓库只保存可复现脚本、源哈希、12 个聚合因子、定义和数据缺口说明。

宽表位于 `corn_forecast/datasets/corn/factors/matrix/corn_market_daily_factors_v1.csv`，覆盖完整 DCE 日历，但产业链因子只在 2025-07-14 以后出现。历史 vintage 尚未验证，因此它是影子回测和残差增强模型的候选输入，不是严格回测默认因子。运行 `python scripts/build_corn_market_daily_factors.py` 可在本地源文件就位后重建，详见 [玉米产业链日频因子集 v1 说明](docs/corn-daily-market-factors-v1.md)。

## 项目结构

```text
corn_forecast/          框架主包
  cli.py                命令行入口
  config/               YAML 配置读取与校验
  data_processing/      CSV 读取、目标生成、特征选择和窗口构造
  datasets/corn/        玉米 raw、processed、factor、prediction library 和 metadata 材料
  pipeline/             回测、训练、评估和报告流程
  modeling/             模型、模型池、损失变体、registry、wrapper 和 ensemble
configs/                模板和官方实验配置
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

1. 将新的 CSV 放到 `local_data/` 或其他本地路径。`local_data/` 默认被 git 忽略，避免和包内数据材料混淆。
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
- [玉米月度数据集 v1](docs/corn-monthly-v1.md)
- [玉米月度因子集 v1](docs/corn-monthly-factors-v1.md)
- [玉米日频因子集 v1](docs/corn-daily-factors-v1.md)
- [玉米产业链日频因子集 v1](docs/corn-daily-market-factors-v1.md)

## 验证

需要做本地验证时，优先使用：

```bash
commodity-backtest --help
python -m corn_forecast.cli --help
commodity-backtest diagnose --config configs/corn.yaml
```
