# Zip Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the useful modules from `commodity-ts-backtest-ready.zip` into the current verified `commodity-backtest` framework while staying aligned with DESIGN.md.

**Architecture:** Keep the current framework as the core. Extend models, reporting, configs, and GitHub packaging through narrow, tested changes instead of replacing the existing pipeline.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, matplotlib Agg backend, pytest, YAML configs.

---

### Task 1: Model Registry Extensions

**Files:**
- Modify: `src/commodity_backtest/models/sklearn_models.py`
- Modify: `src/commodity_backtest/models/registry.py`
- Test: `tests/test_models.py`

- [ ] Write tests for logistic regression, string config compatibility, and optional tree dependency errors.
- [ ] Add `create_logistic_regression`.
- [ ] Add optional `create_lightgbm`, `create_xgboost`, and `create_catboost` helpers that raise clear `ImportError` if dependencies are absent.
- [ ] Update `create_model` to support both existing typed dict configs and zip-style string/name configs.
- [ ] Run `pytest tests/test_models.py -v`.
- [ ] Commit with `feat: expand classical model registry`.

### Task 2: Per-Model Report Outputs

**Files:**
- Modify: `src/commodity_backtest/report/writer.py`
- Modify: `src/commodity_backtest/backtest/engine.py`
- Test: `tests/test_report_outputs.py`
- Test: `tests/test_smoke_cli.py`

- [ ] Write tests that require `rolling_metrics.csv`, `config_resolved.json`, and per-model output directories for every enabled model.
- [ ] Add rolling metrics generation based on cumulative direction accuracy, cumulative return, and rolling 12-window accuracy.
- [ ] Update equity curve plotting to include strategy and buy-and-hold lines.
- [ ] Update the backtest engine to write each model's outputs and keep experiment-level comparison/verdict files.
- [ ] Run report and CLI tests.
- [ ] Commit with `feat: write full per-model experiment outputs`.

### Task 3: Config And CLI Compatibility

**Files:**
- Modify: `src/commodity_backtest/cli.py`
- Modify: `src/commodity_backtest/config/schema.py`
- Create: `configs/soybean.yaml`
- Create: `configs/rebar.yaml`
- Test: `tests/test_config_examples.py`
- Test: `tests/test_smoke_cli.py`

- [ ] Write tests for `diagnose --config` and new config validation.
- [ ] Add `diagnose --config` while preserving `diagnose --csv`.
- [ ] Add soybean and rebar YAML templates that validate but point to user-provided raw data paths.
- [ ] Keep default corn config runnable with installed dependencies only.
- [ ] Run config and smoke CLI tests.
- [ ] Commit with `feat: add config diagnose and commodity templates`.

### Task 4: GitHub Packaging

**Files:**
- Create: `LICENSE`
- Create: `.github/workflows/tests.yml`
- Create: `scripts/run_corn_smoke.py`
- Create: `scripts/clean_outputs.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/configuration.md`
- Test: `tests/test_github_boundary.py`

- [ ] Add MIT license and GitHub Actions test workflow.
- [ ] Add helper scripts that do not delete source files.
- [ ] Update docs to mention integrated model options and output layout.
- [ ] Extend repository boundary test to allow GitHub workflow files and scripts while still blocking generated artifacts.
- [ ] Run docs/boundary tests.
- [ ] Commit with `docs: add github packaging and integrated workflow`.

### Task 5: Final Verification

**Files:**
- No source edits unless verification finds a real bug.

- [ ] Run `python -m pytest -v`.
- [ ] Run `python -m commodity_backtest.cli diagnose --config configs/corn.yaml`.
- [ ] Run `python -m commodity_backtest.cli run --config configs/corn.yaml --output-dir experiments/integration_verify`.
- [ ] Run `python -m commodity_backtest.cli run-lookbacks --config configs/corn.yaml --output-dir experiments/integration_lookbacks`.
- [ ] Verify expected output files exist.
- [ ] Run `git status --short`.
- [ ] Commit any verification fixes with a focused message.