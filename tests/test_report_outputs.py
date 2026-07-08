import json
from pathlib import Path

import pandas as pd
import pytest

from corn_forecast.pipeline.backtest.engine import run_backtest
from corn_forecast.config.loader import load_config
from corn_forecast.pipeline.report.verdict import build_agent_verdict
from corn_forecast.pipeline.report.writer import write_experiment_report


def test_agent_verdict_marks_invalid_constant_predictions():
    metrics = {"DirAcc": 0.55, "DirAcc_CI": [0.40, 0.70], "Sharpe": 0.2, "pred_constant_flag": True}
    baseline = {"DirAcc": 0.50}

    verdict = build_agent_verdict(metrics, baseline_metrics=baseline, primary_metric="DirAcc")

    assert verdict["pass"] is False
    assert verdict["status"] == "invalid"
    assert "constant" in " ".join(verdict["warnings"]).lower()


def test_write_experiment_report_outputs_files(tmp_path):
    predictions = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "actual_label": [1],
            "predicted_label": [1],
            "direction_correct": [1],
            "actual_return": [0.02],
            "strategy_return": [0.02],
            "equity": [1.02],
            "model": ["last_return"],
            "window_id": [0],
        }
    )
    comparison = pd.DataFrame([{"model": "last_return", "DirAcc": 1.0, "Sharpe": 0.0, "ProfitFactor": 1.0}])
    metrics = {"DirAcc": 1.0, "DirAcc_CI": [1.0, 1.0], "Sharpe": 0.0, "pred_constant_flag": True}
    verdict = {"pass": False, "status": "invalid", "warnings": ["constant predictions"], "next_actions": ["check data"]}

    write_experiment_report(
        output_dir=tmp_path,
        model_name="last_return",
        predictions=predictions,
        comparison=comparison,
        metrics=metrics,
        verdict=verdict,
        config={"commodity": {"name": "corn"}},
    )

    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "agent_verdict.json").exists()
    assert (tmp_path / "config_resolved.json").exists()
    assert (tmp_path / "model_outputs" / "last_return" / "predictions.csv").exists()
    assert (tmp_path / "model_outputs" / "last_return" / "rolling_metrics.csv").exists()
    assert (tmp_path / "model_outputs" / "last_return" / "metrics_summary.json").exists()
    assert (tmp_path / "model_outputs" / "last_return" / "equity_curve.png").exists()
    assert (tmp_path / "model_outputs" / "last_return" / "rolling_dir_acc.png").exists()
    assert (tmp_path / "model_outputs" / "last_return" / "rolling_sharpe.png").exists()
    rolling = pd.read_csv(tmp_path / "model_outputs" / "last_return" / "rolling_metrics.csv")
    assert {"cumulative_dir_acc", "cumulative_return", "rolling_12_dir_acc", "rolling_12_sharpe"}.issubset(
        rolling.columns
    )
    loaded = json.loads((tmp_path / "agent_verdict.json").read_text(encoding="utf-8"))
    assert loaded["status"] == "invalid"


def test_agent_verdict_rejects_wide_ci_point_estimate_storytelling():
    metrics = {"DirAcc": 0.64, "DirAcc_CI": [0.43, 0.85], "Sharpe": 1.2, "pred_constant_flag": False}
    baseline = {"DirAcc": 0.50}

    verdict = build_agent_verdict(metrics, baseline_metrics=baseline, primary_metric="DirAcc")

    assert verdict["pass"] is False
    assert verdict["status"] == "weak_signal"
    assert "uncertainty" in " ".join(verdict["warnings"]).lower()


def test_run_backtest_verdict_uses_first_baseline_even_if_not_first_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    cfg = load_config(Path("configs/corn.yaml"), validate=True)
    cfg["models"] = [
        {
            "name": "random_forest",
            "type": "sklearn_random_forest",
            "enabled": True,
            "params": {"n_estimators": 5, "max_depth": 2, "random_state": 7},
        },
        {"name": "last_return", "type": "baseline", "enabled": True},
    ]
    output_dir = tmp_path / "experiment"

    run_backtest(cfg, output_dir=output_dir)

    comparison = pd.read_csv(output_dir / "comparison.csv")
    verdict = json.loads((output_dir / "agent_verdict.json").read_text(encoding="utf-8"))
    baseline_dir_acc = float(comparison.loc[comparison["model"] == "last_return", "DirAcc"].iloc[0])
    assert verdict["baseline_value"] == pytest.approx(baseline_dir_acc)
