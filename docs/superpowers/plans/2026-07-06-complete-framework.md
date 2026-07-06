# Complete Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the commodity backtest framework so it covers the runnable requirements from DESIGN.md, the reusable modules from `C:/????/src`, and the edge-case/metric behavior from `corn-benchmark-data-explore`.

**Architecture:** Keep the current GitHub-ready framework as the core. Add missing capabilities as optional, tested modules: train-only scaling, train/val/test slicing, layer-2 loss models, PyTorch deep adapters, stricter diagnostics, and richer reports. Do not commit experiment outputs, raw data, model weights, or one-off artifacts.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, matplotlib Agg, optional torch for deep models, pytest.

---

### Task 1: Train-Only Scaling And Validation Slices

**Files:**
- Create: `src/commodity_backtest/data/scaler.py`
- Modify: `src/commodity_backtest/backtest/engine.py`
- Modify: `src/commodity_backtest/config/schema.py`
- Modify: `configs/template.yaml`
- Modify: `configs/corn.yaml`
- Modify: `configs/soybean.yaml`
- Modify: `configs/rebar.yaml`
- Test: `tests/test_scaler_and_validation.py`

- [ ] Add tests that assert a scaler fitted on train rows transforms train/validation/test arrays without fitting on future rows.
- [ ] Add tests that assert every backtest row records `train_start_date`, `train_end_date`, optional `val_start_date`, `val_end_date`, and `test_date`.
- [ ] Implement `SequenceStandardizer` with `fit`, `transform_x`, `transform_y`, `inverse_y`, and `to_dict`.
- [ ] Add `split.val_ratio` config validation.
- [ ] Update the engine to carve validation rows from the tail of each training window, fit the scaler only on the remaining train rows, and pass validation arrays into models.
- [ ] Run `pytest tests/test_scaler_and_validation.py tests/test_smoke_cli.py -v`.
- [ ] Commit with `feat: add train-only scaling and validation slices`.

### Task 2: Benchmark Layer-2 Models And Regression Health

**Files:**
- Create: `src/commodity_backtest/models/loss_variants.py`
- Modify: `src/commodity_backtest/models/registry.py`
- Modify: `src/commodity_backtest/eval/metrics.py`
- Test: `tests/test_loss_variants.py`
- Test: `tests/test_metrics.py`

- [ ] Add tests for `regression_mse_sign`, `regression_mae_sign`, `regression_huber_sign`, `dual_head_mse_bce`, and `focal_logistic`.
- [ ] Add tests that `compute_all_metrics` can emit `R2_health` when raw regression predictions are provided.
- [ ] Implement sklearn-based regression-sign adapters and dual-head adapter.
- [ ] Implement focal logistic with a small torch-backed binary linear head when torch exists, and a clear `ImportError` otherwise.
- [ ] Register the new model names and keep them disabled in default corn config.
- [ ] Run `pytest tests/test_loss_variants.py tests/test_metrics.py -v`.
- [ ] Commit with `feat: add benchmark loss variant models`.

### Task 3: Optional Deep Sequence Models

**Files:**
- Create: `src/commodity_backtest/train/losses.py`
- Create: `src/commodity_backtest/train/trainer.py`
- Create: `src/commodity_backtest/models/deep/base.py`
- Create: `src/commodity_backtest/models/deep/lstm.py`
- Create: `src/commodity_backtest/models/deep/gru.py`
- Create: `src/commodity_backtest/models/deep/transformer.py`
- Create: `src/commodity_backtest/models/deep/patchtst.py`
- Create: `src/commodity_backtest/models/deep/dlinear.py`
- Create: `src/commodity_backtest/models/deep/itransformer.py`
- Create: `src/commodity_backtest/models/deep/__init__.py`
- Modify: `src/commodity_backtest/models/registry.py`
- Test: `tests/test_deep_models.py`

- [ ] Add tests that each deep model name can be created.
- [ ] Add one tiny smoke-fit test for LSTM using `epochs=1`, `hidden_size=4`, and six samples.
- [ ] Implement a shared torch classifier adapter with `fit`, `predict`, `predict_proba`, and `save`.
- [ ] Implement compact but real model classes for LSTM, GRU, Transformer, PatchTST-style patch MLP, DLinear, and iTransformer-style inverted projection.
- [ ] Add data-size guard metadata so deep models can be disabled by default and explicitly enabled with small test params.
- [ ] Run `pytest tests/test_deep_models.py tests/test_models.py -v`.
- [ ] Commit with `feat: add optional deep sequence models`.

### Task 4: Richer Reports And Edge-Case Tests

**Files:**
- Modify: `src/commodity_backtest/report/writer.py`
- Modify: `src/commodity_backtest/report/verdict.py`
- Modify: `src/commodity_backtest/data/diagnosis.py`
- Test: `tests/test_report_outputs.py`
- Test: `tests/test_edge_cases.py`

- [ ] Add tests for `rolling_dir_acc.png` and `rolling_sharpe.png`.
- [ ] Add tests for malformed one-column probability handling through model adapters.
- [ ] Add diagnosis status values: `usable`, `usable_with_warnings`, `unusable`.
- [ ] Update report writer to write rolling charts.
- [ ] Update verdict to compare best model against the first baseline and reject wide CI point-estimate storytelling.
- [ ] Run `pytest tests/test_report_outputs.py tests/test_edge_cases.py -v`.
- [ ] Commit with `feat: add richer reports and edge diagnostics`.

### Task 5: Agent Workflow And Final Documentation

**Files:**
- Modify: `src/commodity_backtest/cli.py`
- Modify: `docs/agent-workflow.md`
- Modify: `docs/architecture.md`
- Modify: `docs/configuration.md`
- Modify: `docs/metrics.md`
- Modify: `README.md`
- Test: `tests/test_secondary_cli.py`
- Test: `tests/test_github_boundary.py`

- [ ] Add CLI `auto-window` to recommend expanding/rolling/capped settings from a CSV/config row count.
- [ ] Add CLI `build-config` to create a commodity YAML from an existing config and user-supplied fields.
- [ ] Document the required diagnose/build/run/interpret workflow.
- [ ] Update docs to list every model family, which ones are default, and which ones require optional dependencies.
- [ ] Run `pytest tests/test_secondary_cli.py tests/test_github_boundary.py -v`.
- [ ] Commit with `docs: complete agent workflow and model documentation`.

### Task 6: Full Verification

**Files:**
- No source edits unless verification exposes a bug.

- [ ] Run `python -m pytest -v`.
- [ ] Run `python -m commodity_backtest.cli diagnose --config configs/corn.yaml`.
- [ ] Run `python -m commodity_backtest.cli run --config configs/corn.yaml --output-dir experiments/complete_verify`.
- [ ] Run `python -m commodity_backtest.cli run-lookbacks --config configs/corn.yaml --output-dir experiments/complete_lookbacks`.
- [ ] Run `python scripts/run_corn_smoke.py`.
- [ ] Check that generated output contains `comparison.csv`, `report.md`, `agent_verdict.json`, per-model predictions, rolling metrics, equity curve, rolling DirAcc chart, and rolling Sharpe chart.
- [ ] Run `git status --short`.
- [ ] Commit any verification fixes.