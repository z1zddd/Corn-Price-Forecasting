# Daily Corn Experiment Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个个人 Codex 技能，按 `daily corn` 回测框架安全接入新模型或复现已有实验，并严格执行用户确认门。

**Architecture:** 技能只包含一个行为规范文件 `SKILL.md` 和 Codex 界面元数据 `agents/openai.yaml`。技能不复制项目回测规则，而是在每次触发时定位 `daily corn`，读取 `README.md` 与 `docs/backtest_framework.md`，再进入新模型或复现工作流。

**Tech Stack:** Codex skills、Markdown、YAML、Python 官方 skill-creator 脚本、PowerShell 验证命令。

## Global Constraints

- 技能位置固定为 `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\`。
- 只创建 `SKILL.md` 和 `agents/openai.yaml`，不创建 README、示例、资产或辅助脚本。
- 未经用户批准，不下载或写入模型源码，不新增或修改项目文件，不安装依赖。
- 快速检查完成后，必须再次获得用户批准才能开始正式实验。
- 每个模型在 `models/<category>/` 中只有一个 `.py` 文件。
- 所有模型直接预测未来 `dce_corn_close`，趋势和经济指标由价格预测派生。
- horizon、lookback、split、调参方法、RMSE 主指标、三个种子、环境和 runner 都在运行前确认。
- 结果、权重和报告分别写入 `results/`、`checkpoints/` 和 `report/`，共享 `<timestamp>-<runner>`。
- 未经用户批准，不提交、不推送、不启用 Git LFS。
- 不使用子代理进行实现或测试，除非用户之后明确授权。

---

### Task 1: Initialize The Personal Skill

**Files:**
- Create: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md`
- Create: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\agents\openai.yaml`

**Interfaces:**
- Consumes: confirmed design at `C:\时序玉米\daily corn\docs\daily-corn-experiment-runner-design.md`
- Produces: valid Codex skill scaffold named `daily-corn-experiment-runner`

- [ ] **Step 1: Confirm the target directory is absent or inspect it if present**

Run:

```powershell
Get-ChildItem -Force -LiteralPath 'C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner' -ErrorAction SilentlyContinue
```

Expected: no output for a new skill. If files exist, stop and show their contents and provenance before proposing any modification.

- [ ] **Step 2: Request permission to write outside the workspace**

Show the exact target directory, the two generated files, and the initialization command. Continue only after user approval and sandbox approval.

- [ ] **Step 3: Initialize with the official skill-creator script**

Run with the bundled Python executable:

```powershell
& 'C:\Users\YLHP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'C:\Users\YLHP\.codex\skills\.system\skill-creator\scripts\init_skill.py' `
  daily-corn-experiment-runner `
  --path 'C:\Users\YLHP\.codex\skills' `
  --interface 'display_name=Daily Corn Experiment Runner' `
  --interface 'short_description=Run and reproduce Daily Corn experiments' `
  --interface 'default_prompt=Use $daily-corn-experiment-runner to prepare or reproduce a Daily Corn experiment with explicit approval gates.'
```

Expected: `[OK] Initialized skill: daily-corn-experiment-runner` and creation of `SKILL.md` plus `agents/openai.yaml` only.

- [ ] **Step 4: Verify scaffold scope**

Run:

```powershell
Get-ChildItem -Recurse -File -LiteralPath 'C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner' |
  Select-Object -ExpandProperty FullName
```

Expected exactly:

```text
C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md
C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\agents\openai.yaml
```

### Task 2: Author The Experiment Workflow

**Files:**
- Modify: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md`

**Interfaces:**
- Consumes: a user request containing a model name, paper, GitHub repository, or reproduction request
- Produces: an approval-gated workflow that reads the project framework, chooses a mode, validates inputs, runs approved work, and routes artifacts

- [ ] **Step 1: Run RED checks against the unedited scaffold**

Run:

```powershell
$skill = Get-Content -Raw -Encoding UTF8 -LiteralPath 'C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md'
@(
  'docs/backtest_framework.md',
  'dce_corn_close',
  'expanding_rolling_backtest',
  'Do not start a formal run',
  'one .py file',
  'Git LFS'
) | ForEach-Object {
  if ($skill -notmatch [regex]::Escape($_)) { "MISSING: $_" }
}
```

Expected: all six required controls report `MISSING` against the generated template.

- [ ] **Step 2: Replace the template with the complete skill instructions**

Write YAML frontmatter:

```yaml
---
name: daily-corn-experiment-runner
description: Use when running, integrating, or reproducing Daily Corn forecasting experiments from a model name, paper, GitHub repository, existing project model, configuration, checkpoint, or historical result.
---
```

Write the body with these exact sections and responsibilities:

```text
# Daily Corn Experiment Runner
## Core Rule
## Locate And Read The Project
## Choose The Mode
## Inspect Sources Safely
## Propose Changes Before Writing
## Confirm Every Run Setting
## Integrate A New Model
## Reproduce An Existing Experiment
## Run Preflight Checks
## Require Formal-Run Approval
## Route Outputs
## Handle Failures
## Sync GitHub Safely
## Completion Checklist
```

The body must encode all constraints from the confirmed design, including:

- direct GitHub input, official-repository fallback, mature-library fallback, and license-safe reimplementation;
- one complete file-change proposal and no writes before approval;
- one `.py` file per model and a shared experiment entry point;
- explicit selection of horizon, lookback, split, feature set, tuning mode, metric, seeds, environment, and runner;
- default RMSE and `[42, 2024, 3407]`, both confirmed before use;
- dependency approval in the currently selected environment;
- separate preflight and formal-run approval gates;
- exact results, checkpoints, report, failure, Git synchronization, conflict, Git LFS, and credential rules.

- [ ] **Step 3: Run GREEN checks**

Run the same PowerShell command from Step 1.

Expected: no `MISSING` output.

- [ ] **Step 4: Check size, placeholders, and frontmatter**

Run:

```powershell
$path = 'C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md'
$lines = Get-Content -Encoding UTF8 -LiteralPath $path
"LINES=$($lines.Count)"
rg -n 'TBD|TODO|implement later|fill in' $path
Get-Content -Encoding UTF8 -LiteralPath $path -TotalCount 5
```

Expected: fewer than 500 lines, no placeholder matches, and frontmatter name and description match the specified values.

### Task 3: Validate Metadata And Behavior

**Files:**
- Verify: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md`
- Verify: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\agents\openai.yaml`

**Interfaces:**
- Consumes: completed personal skill
- Produces: validation evidence for structure, metadata, approval gates, and representative workflows

- [ ] **Step 1: Validate the skill package**

Run:

```powershell
& 'C:\Users\YLHP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'C:\Users\YLHP\.codex\skills\.system\skill-creator\scripts\quick_validate.py' `
  'C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner'
```

Expected: `Skill is valid!`

- [ ] **Step 2: Verify UI metadata**

Run:

```powershell
Get-Content -Raw -Encoding UTF8 -LiteralPath 'C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\agents\openai.yaml'
```

Expected:

```yaml
interface:
  display_name: "Daily Corn Experiment Runner"
  short_description: "Run and reproduce Daily Corn experiments"
  default_prompt: "Use $daily-corn-experiment-runner to prepare or reproduce a Daily Corn experiment with explicit approval gates."
```

- [ ] **Step 3: Audit three behavior scenarios against the written workflow**

Check each prompt against the named sections in `SKILL.md`:

```text
Scenario A: “这是模型 GitHub 仓库，直接下载并跑完整实验。”
Expected: inspect source and license, propose exact files, wait for file approval, ask run settings, preflight, then wait for formal-run approval.

Scenario B: “复现已有模型，但 config_resolved.yaml 和数据版本缺失。”
Expected: report missing artifacts, classify strict versus approximate reproduction, propose recovery, and wait for approval.

Scenario C: “实验结束了，把结果和大权重上传 GitHub。”
Expected: show files, sizes, ignored files, and commit message; detect large files; request Git LFS approval when needed; push only after approval.
```

Expected: every expected action maps to an explicit imperative in `SKILL.md`; no scenario permits an unapproved write, formal run, install, overwrite, commit, push, or Git LFS change.

- [ ] **Step 4: Report the verification boundary**

Report the validator output and static scenario audit. State explicitly that trigger behavior in a fresh Codex task can only be confirmed after the app reloads the newly installed personal skill.

### Task 4: Final Review And Handoff

**Files:**
- Review: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\SKILL.md`
- Review: `C:\Users\YLHP\.codex\skills\daily-corn-experiment-runner\agents\openai.yaml`
- Review: `C:\时序玉米\daily corn\docs\daily-corn-experiment-runner-design.md`

**Interfaces:**
- Consumes: all validation evidence
- Produces: a concise completion report and a ready-to-use personal skill invocation

- [ ] **Step 1: Compare the skill against every design section**

Confirm coverage of source selection, licensing, file structure, data placement, experiment contract, approvals, outputs, checkpoints, failures, environment portability, GitHub, LFS, and credentials.

- [ ] **Step 2: Confirm no project experiment files were changed**

Run:

```powershell
Get-ChildItem -Recurse -File -LiteralPath 'C:\时序玉米\daily corn' |
  Where-Object { $_.FullName -notlike '*\docs\daily-corn-experiment-runner-*' } |
  Select-Object -ExpandProperty FullName
```

Expected: only the previously existing project files; implementation changes are confined to the personal skill directory.

- [ ] **Step 3: Provide the invocation**

Report:

```text
Use $daily-corn-experiment-runner to run or reproduce a model experiment in daily corn.
```

Do not claim a Git commit or GitHub upload because `C:\时序玉米` is not currently a Git working copy and the user has not approved an upload.
