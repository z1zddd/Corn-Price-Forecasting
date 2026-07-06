# Zip Integration Design

## Goal

Use the current `commodity-backtest` project as the stable core and integrate the useful, DESIGN.md-aligned parts of `commodity-ts-backtest-ready.zip` without replacing the working interfaces.

## Design Boundary

The current framework keeps these core decisions:

- YAML-driven commodity configs.
- `train_window` naming and expanding / rolling / expanding_with_cap modes.
- automatic target generation from `data.price_col`.
- conservative report and `agent_verdict.json` behavior.
- `model_outputs/<model_name>/...` output layout.

The zip framework contributes capabilities, not wholesale architecture:

- additional classical model adapters: logistic regression and optional LightGBM/XGBoost/CatBoost wrappers.
- optional string/name-only model config compatibility.
- per-model output files instead of only best-model files.
- rolling metrics CSV.
- config-resolved JSON output.
- sample commodity YAMLs for soybean and rebar.
- GitHub-ready metadata: LICENSE, CI workflow, scripts, docs updates.

## Exclusions

Do not integrate deep learning placeholder files from the zip. DESIGN.md names LSTM/GRU/Transformer/PatchTST/iTransformer/DLinear as future model families, but the zip implementations are placeholders that raise `NotImplementedError`. Adding them now would make the project look more complete while adding no runnable capability.

Do not hide model failures. Optional tree models may raise clear `ImportError` when enabled without dependencies. Default configs must only enable installed, verified models.

## Output Requirements

Each enabled model should produce:

- `predictions.csv`
- `rolling_metrics.csv`
- `metrics_summary.json`
- `equity_curve.png`

The experiment directory should produce:

- `comparison.csv`
- `report.md`
- `agent_verdict.json`
- `data_manifest.json`
- `config_resolved.json`

The equity chart should include both strategy and buy-and-hold curves and must use matplotlib's non-GUI `Agg` backend.

## Test Requirements

Tests must cover:

- logistic regression model creation and prediction.
- optional tree model error messages when dependencies are missing.
- string/name-only model config compatibility.
- per-model report outputs and rolling metrics.
- config-based diagnose CLI compatibility.
- new config examples validate without running missing data files.
- full smoke CLI still runs on corn sample data.

## Acceptance

The integration is accepted when:

- `python -m pytest -v` passes.
- `commodity_backtest.cli diagnose --config configs/corn.yaml` runs.
- `commodity_backtest.cli run --config configs/corn.yaml` runs and writes all output files.
- `commodity_backtest.cli run-lookbacks --config configs/corn.yaml` runs.
- `git status --short` is clean after committing.