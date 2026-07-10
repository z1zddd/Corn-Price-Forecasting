# Agent Workflow

`agent_verdict.json` is a conservative machine-readable summary intended for later automation.

## Verdict Status

Possible statuses:

- `invalid`: predictions are constant or otherwise fail a basic health check.
- `signal`: the model beats the baseline, uncertainty is controlled, and Sharpe is positive.
- `weak_signal`: the model beats the baseline but uncertainty or trading quality is weak.
- `no_signal`: the model does not beat the baseline.

The verdict includes:

```json
{
  "pass": false,
  "status": "weak_signal",
  "primary_metric": "DirAcc",
  "primary_value": 0.55,
  "baseline_value": 0.50,
  "ci_width": 0.18,
  "warnings": [],
  "next_actions": []
}
```

## Recommended Human Loop

1. Run `commodity-backtest diagnose` before every new dataset.
2. Run `commodity-backtest build-config` when adapting a template to a new commodity.
3. Run `commodity-backtest auto-window` to choose a conservative starting window from available rows.
4. Run `commodity-backtest run` for the main config.
5. Run `commodity-backtest run-lookbacks` when lookback sensitivity matters.
6. Read `comparison.csv`, `report.md`, and per-model rolling charts.
7. Run `commodity-backtest interpret` or read `agent_verdict.json`.
8. If the verdict is `weak_signal`, run more windows, compare more baselines, or test additional features.
9. If the verdict is `signal`, treat it as research evidence and continue with out-of-sample validation before any live decision.

## CLI Commands

```bash
commodity-backtest diagnose --csv corn_forecast/datasets/corn/processed/corn_sample_data.csv --date-col date
commodity-backtest build-config --base-config configs/template.yaml --output configs/my_commodity.yaml --commodity-name my_commodity --csv local_data/my.csv --date-col date --price-col close
commodity-backtest auto-window --config configs/corn.yaml
commodity-backtest run --config configs/corn.yaml
commodity-backtest run-lookbacks --config configs/corn.yaml
commodity-backtest compare --experiment experiments/manual_run
commodity-backtest interpret --experiment experiments/manual_run
```

## GitHub Hygiene

Generated experiments are intentionally ignored by git. Keep only source code, configs, docs, and small reference data in the repository.

Do not commit:

- trained weights or pickled models
- large compressed archives
- Office files
- local virtual environments
- experiment output folders
