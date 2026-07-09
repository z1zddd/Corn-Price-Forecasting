# src/ Optimization TODO

目标：把当前 `src/` 从“能跑通的统一框架”优化成“时序实验协议稳定、模型结果可比、AI agent 不会各跑各的”的工程框架。

## P0：先修实验协议

- [ ] 统一任务定义
  - [ ] 明确默认任务：预测未来 `horizon` 个交易日收益率。
  - [ ] 在配置中显式声明 `target_mode: return | price | direction`。
  - [ ] 在配置中显式声明 `horizon`、`seq_len`、`include_today`。
  - [ ] 禁止模型文件内部自行定义 target。
  - 验收：`data_config.json` 中能完整复现实验目标。

- [ ] 统一样本元信息
  - [ ] `meta` 必须包含 `date`、`target_date`、`today_close`、`target_idx`、`anchor_idx`。
  - [ ] 增加 `series_id` 字段，为玉米/白糖/多品种扩展预留。
  - [ ] 增加 `horizon` 字段，避免报告里看不出预测步长。
  - 验收：任意 `predictions.csv` 都能追溯每条预测样本。

- [ ] 统一时间切分
  - [ ] 所有 split 按 `target_date`，不按 anchor `date`。
  - [ ] `train target_date < val target_date < test target_date` 必须自动检查。
  - [ ] smoke / full run 都走同一套 split 函数。
  - 验收：切分不满足时直接 raise，不能静默继续训练。

- [ ] 统一归一化
  - [ ] loader 只负责读数据和造窗口，不允许 impute/scale。
  - [ ] scaler 只在 train 上 fit。
  - [ ] X 缺失值按 train median 填充。
  - [ ] X 按 train mean/std 标准化。
  - [ ] y 按 train y_mean/y_std 标准化。
  - [ ] 保存 scaler 参数到每次实验目录。
  - 验收：`run_dir/preprocess.json` 或 `preprocess.pkl` 存在，且只包含 train-fit 参数。

## P1：把评估变成真正时序 backtest

- [ ] 增加 rolling-origin backtest 入口
  - [ ] 实现 `src/run_backtest.py`。
  - [ ] 使用 `src/data/cv.py` 中的 rolling-origin split。
  - [ ] 支持配置 `n_windows`、`h`、`step_size`、`min_train_size`、`max_train_size`。
  - 验收：一次运行产出多个 cutoff 的预测和汇总指标。

- [ ] 区分 smoke、holdout、backtest
  - [ ] `--smoke` 只用于快速检查。
  - [ ] `src/run.py` 明确输出 single holdout。
  - [ ] `src/run_backtest.py` 输出正式 rolling result。
  - [ ] 报告中禁止把 smoke 指标当正式结果。
  - 验收：输出目录里有 `run_type` 字段。

- [ ] 统一 predictions 文件
  - [ ] 每个模型输出统一列：
    - `series_id`
    - `date`
    - `target_date`
    - `horizon`
    - `today_close`
    - `y_true_return`
    - `y_pred_return`
    - `actual_price`
    - `pred_price`
    - `actual_direction`
    - `pred_direction`
    - `strategy_return`
  - 验收：所有模型的 `predictions.csv` schema 完全一致。

- [ ] 统一 metrics 文件
  - [ ] 主指标固定顺序：
    - `direction_accuracy`
    - `profit_factor`
    - `sharpe_ratio`
    - `win_rate`
    - `max_drawdown`
  - [ ] 辅助指标固定顺序：
    - `rmse`
    - `mae`
    - `r2`
  - [ ] 增加 `pred_up_rate`、`actual_up_rate`、`pred_constant_flag`。
  - 验收：`metrics.json` 和 `metrics.csv` 字段稳定。

## P2：模型接口收紧

- [ ] 完善 `BaseModel`
  - [ ] 增加 `predict_target()` 和 `predict_price()` 的职责说明。
  - [ ] 明确模型只输出训练目标空间的预测，不直接算指标。
  - [ ] 增加 `get_params()`，保存可复现实验参数。
  - 验收：所有模型都可序列化并恢复预测。

- [ ] 经典模型适配器规范化
  - [ ] flatten 逻辑集中在一个地方。
  - [ ] 支持回归/分类但默认回归收益率。
  - [ ] XGBoost/LightGBM/CatBoost 的线程参数统一配置。
  - [ ] 训练失败时输出明确依赖/运行时错误。
  - 验收：同一份 X/y 下四个 classical 模型都能 fit/predict。

- [ ] 深度模型训练规范化
  - [ ] DL 模型内部只接受 `[N, V, T]`，内部自己转 `[N, T, V]`。
  - [ ] 训练循环统一 early stopping 监控 val MAE 或 val loss。
  - [ ] 保存 best epoch、best val metric、history.csv。
  - [ ] DL 保存格式统一为 `.pt`，不要混用 `.pkl`。
  - 验收：LSTM/GRU/Transformer 类模型产物结构一致。

- [ ] baseline 必须存在
  - [ ] 增加 naive last-return baseline。
  - [ ] 增加 moving-average return baseline。
  - [ ] 增加 zero-return baseline。
  - 验收：任何正式报告必须先展示 baseline，再展示复杂模型。

## P3：配置和产物治理

- [ ] 配置结构整理
  - [ ] `src/config/data.yaml` 只管数据。
  - [ ] `src/config/task.yaml` 只管 target/horizon/metric。
  - [ ] `src/config/validation.yaml` 只管 holdout/backtest。
  - [ ] `玉米预测/operator/model/configs/*.yaml` 只管模型参数。
  - [ ] `src/config/experiment.yaml` 只组合以上配置。
  - 验收：改模型不需要改数据配置，改 horizon 不需要改模型配置。

- [ ] 数据 manifest
  - [ ] 每次 run 保存输入 CSV 路径、文件大小、mtime、列名 hash。
  - [ ] 保存 feature_cols。
  - [ ] 保存 split 日期范围。
  - 验收：`data_manifest.json` 能复现实验数据版本。

- [ ] 产物目录标准化
  - [ ] run 目录命名包含时间戳、实验名、run_type。
  - [ ] 每个模型独立子目录。
  - [ ] 根目录保存 `comparison.csv`、`experiment_config.json`、`data_manifest.json`。
  - 验收：不同模型、不同 cutoff 不会互相覆盖。

- [ ] 报告生成
  - [ ] 增加 `report.md`。
  - [ ] 报告必须包含：
    - 数据范围
    - split 范围
    - target 定义
    - 模型列表
    - 主指标表
    - equity curve
    - 风险提示
  - 验收：非代码同学能直接读报告判断模型是否可用。

## P4：质量检查和测试

- [ ] 数据单元测试
  - [ ] 测试窗口形状 `[N, V, T]`。
  - [ ] 测试 `target_date > date`。
  - [ ] 测试 split 无泄漏。
  - [ ] 测试 scaler 只用 train fit。

- [ ] 模型单元测试
  - [ ] 所有模型在小随机数据上 fit/predict 不报错。
  - [ ] 所有模型预测 shape 为 `[N]`。
  - [ ] save/load 后预测一致或近似一致。

- [ ] 评估单元测试
  - [ ] 人工构造全对/全错方向样本。
  - [ ] 检查 profit factor、sharpe、max drawdown。
  - [ ] 检查 predictions schema。

- [ ] 回归测试
  - [ ] 固定一个小数据切片作为 smoke fixture。
  - [ ] CI 或本地命令跑：
    ```bash
    python -m compileall src
    python -m src.run --models random_forest --smoke
    ```
  - 验收：任何改动不能破坏最小链路。

## 推荐执行顺序

1. P0：任务、target、split、scaler 全部定死。
2. P1：补 rolling-origin backtest。
3. P2：补 baseline，并收紧 BaseModel。
4. P3：整理配置和产物。
5. P4：补测试，防止以后 AI agent 改坏。

## 最重要的红线

- [ ] 模型文件不能读 CSV。
- [ ] 模型文件不能切 train/test。
- [ ] 模型文件不能 fit scaler。
- [ ] 模型文件不能计算正式指标。
- [ ] 模型文件只能实现 `fit/predict/save/load`。
- [ ] 所有正式指标只能来自 `src/eval/`。
- [ ] 所有正式样本只能来自 `src/data/`。
