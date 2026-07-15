# Daily Corn 实验运行技能设计

## 1. 目标与位置

创建个人 Codex 技能：

```text
C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\
```

技能用于将新模型接入 `daily corn` 日度回测框架，或复现项目中已有实验。技能必须先读取：

```text
daily corn/docs/backtest_framework.md
daily corn/README.md
```

回测行为以 `backtest_framework.md` 为准，产物目录以项目根 README 的 `results/`、`checkpoints/`、`report/` 约定为准。

## 2. 触发方式与工作模式

技能支持以下输入：

- 模型名称；
- 论文或论文链接；
- 用户提供的模型 GitHub 仓库；
- 对 `daily corn` 中已有实验的复现要求。

### 2.1 新模型模式

模型或配套文件不存在时：

1. 检查用户提供的 GitHub 仓库；若未提供，则依次查找论文官方仓库和成熟库实现。
2. 审查源码、提交版本、依赖、许可证和与日度回测框架的兼容性。
3. 列出全部新增和修改文件，逐项说明路径、用途、主要内容、具体改动、依赖、命令和影响。
4. 得到用户一次性批准后，才可下载、创建或修改文件。
5. 若源码没有允许复制或修改的许可证，不复制受限源码；根据论文和公开接口提出重新实现方案，得到用户批准后再实现。

### 2.2 复现模式

已有模型产物时：

1. 从 `models/` 查找模型实现。
2. 从 `configs/` 查找模型配置。
3. 从 `scripts/` 查找通用实验入口和支撑脚本。
4. 从 `checkpoints/` 查找权重或序列化状态。
5. 恢复历史 `config_resolved.yaml`、数据哈希、依赖环境和 Git commit。
6. 不重新设计模型，不覆盖历史运行，使用新的运行标识复现。

若历史配置、权重或数据版本不完整，技能必须列出缺失项、影响和恢复方案，并区分严格复现与近似复现；得到用户确认后才能补齐或运行。

## 3. 项目文件约束

新模型接入采用以下结构：

```text
daily corn/
├─ models/<category>/<ModelName>.py
├─ configs/<model_name>.yaml
├─ scripts/data_processing/data_pipeline.py
├─ scripts/experiments/run_experiment.py
├─ scripts/evaluation/evaluate.py
└─ tests/
```

规则：

- `models/` 中每个模型只有一个 `.py` 文件，不建立模型子目录。
- 模型专属网络层、损失和训练细节保存在该模型文件中。
- 数据处理、时间划分、回测、指标和产物保存使用通用脚本。
- 所有模型共用 `scripts/experiments/run_experiment.py`。
- 每个模型只有一个可编辑 YAML 配置。
- 每次实验把实际配置保存为不可变的 `config_resolved.yaml`。
- 缺少模型、配置、脚本或测试时，必须先提交完整变更清单并获得批准。

模型文件提供统一能力：

```text
fit(...)
predict(...) -> predicted_dce_corn_close
save(...)
load(...)
```

不同模型的架构参数放在各自 YAML 的 `model.params` 中，由对应模型文件解释和校验。通用实验脚本不硬编码模型参数。

## 4. 数据规则

数据文件不在项目内时，技能先检查文件内容和处理状态，再询问用户将其放入：

```text
daily corn/datasets/raw/
```

或：

```text
daily corn/datasets/processed/
```

得到确认后才复制，并在配置中使用项目内相对路径。

同名数据已存在时，比较文件哈希、日期范围、行列数、字段和缺失情况。内容不一致时停止并报告差异，由用户选择重新命名或替换，禁止自动覆盖。

## 5. 实验契约

所有模型必须直接预测未来 `dce_corn_close`。趋势、收益和经济指标只能由当前价格、真实未来价格和预测未来价格派生。

框架候选参数为：

```text
horizons = [5, 10, 15, 20]
lookbacks = [5, 10, 15, 20]
split_strategies = [
  chronological_811,
  chronological_712,
  expanding_rolling_backtest
]
```

技能不得默认运行全部组合。每次实验开始前必须让用户确认：

- 数据集和模型；
- 预测步长和回看步长；
- 数据划分策略和特征集；
- 调参方式、范围和预算；
- 主评价指标和随机种子；
- 运行环境和 runner 标识。

调参可选：

1. 根据模型提出推荐方法和预算；
2. 网格搜索；
3. Optuna。

默认调参主指标为 RMSE，但每次运行前必须确认并允许修改。测试集不得参与调参。新模型以论文或官方参数为基准提出精简范围；复现实验默认使用历史配置。

默认随机种子为：

```text
[42, 2024, 3407]
```

运行前允许修改。报告三个种子的均值、标准差和最差结果，不得只保留最好种子。

## 6. 审批与执行流程

新模型实验按以下顺序执行：

1. 展示源码来源和全部文件变更，获得用户批准。
2. 检查当前指定环境；缺少依赖时，列出包、版本、冲突风险和安装命令。
3. 得到用户批准后，将依赖安装到当前环境；不创建新环境。
4. 执行导入、配置、数据形状、输出形状、时间泄漏和小样本快速检查。
5. 报告检查结果、资源需求、预计耗时和正式命令。
6. 再次得到用户批准后，才开始正式训练、调参和回测。

本机与 AutoDL 可以使用不同的现有环境和绝对路径，但同一次实验从检查到正式运行必须始终使用同一环境。

## 7. 产物结构

每次运行前由操作者输入 `runner`。`runner` 只允许简短的英文字母、数字或连字符。运行标识为：

```text
<YYYY-MM-DD-HH-MM-SS>-<runner>
```

三类产物使用同一个运行标识：

```text
results/<dataset>/<model>/<run_id>/
checkpoints/<dataset>/<model>/<run_id>/
report/<dataset>/<model>/<run_id>/
```

`results/` 至少包含：

```text
config_resolved.yaml
experiment_manifest.json
data_audit.json
predictions.csv
metrics.csv
fold_audit.csv
```

运行多个划分策略时，保存各策略测试结果和对比表。expanding rolling backtest 的测试结果必须独立报告，不能遗漏。

`experiment_manifest.json` 至少记录操作系统、设备、Python、CUDA、框架和依赖版本、Git commit、数据哈希、runner、模型来源和运行状态。

`report/<dataset>/<model>/<run_id>/report.md` 包含数据与环境版本、实际运行范围、参数、价格指标、派生趋势指标、经济指标、种子稳定性、失败记录和复现说明。

## 8. 权重与失败处理

- 每个正式实验设置和种子只保存验证集最优 checkpoint。
- 不保存每个 epoch 或全部调参候选权重。
- 机器学习模型保存拟合后的序列化状态。
- expanding rolling backtest 不保存每个滚动时点的权重，只保存必要状态和完整预测审计。

实验失败或中断时：

- 保留已完成结果、错误摘要、配置和环境信息；
- 使用 `FAILED` 标记运行状态；
- 默认不把失败产物列入 Git 提交；
- 不覆盖历史运行。

## 9. 本机、AutoDL 与 GitHub

本机和 AutoDL 使用同一 GitHub 仓库的独立工作副本。多人共用同一套 `results/`、`checkpoints/` 和 `report/` 目录，仅通过 `run_id` 区分实验。

同步流程：

```text
检查工作区
→ 同步远端最新版本
→ 检查实验输入
→ 运行实验
→ 写入唯一 run_id 目录
→ 展示待提交和忽略文件
→ 用户确认
→ 再次同步并检查冲突
→ 提交并推送
```

发现未提交修改、远端冲突或同名运行目录时停止，不自动覆盖、强制合并或丢弃他人修改。

实验完成后不自动上传。技能先展示待提交文件、文件大小、忽略文件和提交说明，得到用户确认后才提交并推送。

提交前检查 GitHub 文件大小限制。需要 Git LFS 时，先说明涉及的文件、LFS 配置改动和仓库影响，得到单独批准后才启用和上传。

技能只使用运行环境已有的 Git、SSH 或 GitHub 登录状态，不把密码、Token、私钥或本机敏感路径写入项目、配置、结果或报告。

## 10. 验收标准

技能必须满足：

- 能区分新模型接入和历史实验复现；
- 能接受模型名称、论文和用户提供的 GitHub 仓库；
- 未经批准不创建模型、脚本、配置、测试或安装依赖；
- 未经第二次运行批准不开始正式实验；
- 每个模型在 `models/` 中只有一个 `.py` 文件；
- 所有模型通过通用实验入口运行并直接输出未来价格；
- 时间划分、预处理和 expanding rolling backtest 遵守框架且无时间泄漏；
- 结果、权重和报告写入各自目录并使用相同唯一运行标识；
- 复现不覆盖历史，资料不全时先报告并请求确认；
- 未经批准不提交、推送或启用 Git LFS；
- 能在本机和 AutoDL 的现有环境中执行并记录完整复现信息。
