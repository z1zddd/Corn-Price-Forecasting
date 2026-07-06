# Commodity Backtest

Commodity Backtest is a YAML-driven, multi-commodity time-series forecasting backtest framework.
It was organized from the corn benchmark exploration code into a reusable project that can run corn first and then switch to other commodities by changing configuration.

The framework loads a commodity CSV, derives forward targets from the configured price column, builds chronological windows, runs enabled models, evaluates both forecasting and trading metrics, and writes reports that are readable by people and automation.

## What It Does

- Supports commodity-specific YAML configs.
- Keeps time-series order intact; no shuffled train/test split.
- Generates targets from `price_col` instead of trusting prebuilt labels.
- Supports expanding, rolling, and capped expanding backtest windows.
- Fits scalers only on each rolling window's training slice and can carve a validation tail from that slice.
- Includes baselines, scikit-learn models, benchmark loss variants, and optional PyTorch sequence models.
- Writes `comparison.csv`, `report.md`, `agent_verdict.json`, per-model predictions, metrics, equity charts, `rolling_dir_acc.png`, and `rolling_sharpe.png`.
- Includes tests for config validation, data processing, split modes, models, metrics, reports, CLI commands, and repository boundaries.

## Install

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -e .[dev]
```

On macOS or Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`.

## Quickstart

```bash
commodity-backtest diagnose --csv examples/corn/sample_data.csv --date-col date
commodity-backtest diagnose --config configs/corn.yaml
commodity-backtest auto-window --config configs/corn.yaml
commodity-backtest build-config --base-config configs/template.yaml --output configs/my_commodity.yaml --commodity-name my_commodity --csv data/raw/my.csv --date-col date --price-col close
commodity-backtest run --config configs/corn.yaml
commodity-backtest run-lookbacks --config configs/corn.yaml
commodity-backtest compare --experiment experiments/manual_run
commodity-backtest interpret --experiment experiments/manual_run
```

The default run writes outputs under `experiments/manual_run/`, which is intentionally ignored by git.

## Output Layout

After `commodity-backtest run --config configs/corn.yaml`, the experiment directory contains:

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

Each enabled model writes its own `model_outputs/<model>/` directory. The equity chart includes both strategy and buy-and-hold curves.

## Switch To Another Commodity

1. Put the new CSV under `data/raw/` or another local path. `data/raw/` is ignored by git.
2. Copy `configs/template.yaml`, or start from `configs/soybean.yaml` / `configs/rebar.yaml`.
3. Update `commodity.name`, `data.csv_path`, `data.date_col`, `data.price_col`, and feature settings.
4. Run `commodity-backtest diagnose --csv <path> --date-col <date_col>`.
5. Run `commodity-backtest run --config configs/soybean.yaml`.

## Models

Built-in runnable models:

- `last_return`
- `mean_return` / `mean_direction`
- `logistic_regression`
- `random_forest`
- `regression_mse_sign`
- `regression_mae_sign`
- `regression_huber_sign`
- `dual_head_mse_bce`

Optional tree models are available when their dependencies are installed:

- `lightgbm`
- `xgboost`
- `catboost`

Optional PyTorch models are available when the `deep` extra is installed:

- `focal_logistic`
- `lstm`
- `gru`
- `transformer`
- `patchtst`
- `itransformer`
- `dlinear`

Install optional tree dependencies with either spelling:

```bash
pip install -e .[trees]
pip install -e .[tree]
```

Install optional deep dependencies with:

```bash
pip install -e .[deep]
```

## Design Rules

- Never shuffle time-series data.
- Generate targets from the configured price column.
- Keep raw data, experiment outputs, model weights, and compressed artifacts out of git.
- Treat backtest results as research evidence, not live trading promises.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Metrics](docs/metrics.md)
- [Agent Workflow](docs/agent-workflow.md)

## Test

```bash
python -m pytest -v
```

GitHub Actions is configured in `.github/workflows/tests.yml` to run the same test suite on push and pull request.