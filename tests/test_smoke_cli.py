import json
import os
import subprocess
import sys
from pathlib import Path


def subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    return env


def test_cli_diagnose_runs(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "cli", "diagnose", "--csv", "examples/corn/sample_data.csv", "--date-col", "date"],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    assert "rows" in result.stdout
    assert "price_candidates" in result.stdout


def test_cli_diagnose_runs_from_config(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "cli", "diagnose", "--config", "configs/corn.yaml"],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    assert "rows" in result.stdout
    assert "price_candidates" in result.stdout
    assert "encoding" in result.stdout


def test_cli_run_writes_outputs(tmp_path: Path):
    output_dir = tmp_path / "experiment"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "run",
            "--config",
            "configs/corn.yaml",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    assert "comparison.csv" in result.stdout
    assert (output_dir / "comparison.csv").exists()
    assert (output_dir / "agent_verdict.json").exists()
    assert (output_dir / "config_resolved.json").exists()
    assert (output_dir / "model_outputs" / "last_return" / "rolling_metrics.csv").exists()
    assert (output_dir / "model_outputs" / "last_return" / "equity_curve.png").exists()
    assert (output_dir / "model_outputs" / "random_forest" / "predictions.csv").exists()
    assert (output_dir / "model_outputs" / "random_forest" / "metrics_summary.json").exists()
    verdict = json.loads((output_dir / "agent_verdict.json").read_text(encoding="utf-8"))
    assert "status" in verdict
