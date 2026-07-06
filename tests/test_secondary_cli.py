import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    return env


def write_small_config(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(Path("configs/corn.yaml").read_text(encoding="utf-8"))
    cfg["models"] = [{"name": "last_return", "type": "baseline", "enabled": True}]
    cfg["lookback"]["candidates"] = [3, 4]
    cfg_path = tmp_path / "small_corn.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def test_compare_and_interpret_commands(tmp_path: Path):
    cfg_path = write_small_config(tmp_path)
    output_dir = tmp_path / "experiment"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "run",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    compare = subprocess.run(
        [sys.executable, "-m", "cli", "compare", "--experiment", str(output_dir)],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )
    interpret = subprocess.run(
        [sys.executable, "-m", "cli", "interpret", "--experiment", str(output_dir)],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    assert "last_return" in compare.stdout
    verdict = json.loads(interpret.stdout)
    assert "status" in verdict


def test_run_lookbacks_command(tmp_path: Path):
    cfg_path = write_small_config(tmp_path)
    output_dir = tmp_path / "lookbacks"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "run-lookbacks",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    assert "lookback_comparison.csv" in result.stdout
    assert (output_dir / "lookback_comparison.csv").exists()


def test_run_accepts_string_model_config(tmp_path: Path):
    cfg = yaml.safe_load(Path("configs/corn.yaml").read_text(encoding="utf-8"))
    cfg["models"] = ["last_return"]
    cfg["lookback"]["candidates"] = [3]
    cfg_path = tmp_path / "string_model.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "string_model_run"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "run",
            "--config",
            str(cfg_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    assert "last_return" in result.stdout
    assert (output_dir / "model_outputs" / "last_return" / "predictions.csv").exists()


def test_auto_window_command_recommends_settings(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "cli", "auto-window", "--config", "configs/corn.yaml"],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    payload = json.loads(result.stdout)
    recommendation = payload["recommendation"]
    assert payload["rows"] > 0
    assert recommendation["train_window"]["mode"] in {"expanding", "rolling", "expanding_with_cap"}
    assert recommendation["lookback"]["default"] < recommendation["train_window"]["min_train_periods"]


def test_build_config_command_writes_commodity_yaml(tmp_path: Path):
    output_path = tmp_path / "new_commodity.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "build-config",
            "--base-config",
            "configs/template.yaml",
            "--output",
            str(output_path),
            "--commodity-name",
            "test_corn",
            "--exchange",
            "TEST",
            "--frequency",
            "monthly",
            "--csv",
            "examples/corn/sample_data.csv",
            "--date-col",
            "date",
            "--price-col",
            "close",
        ],
        check=True,
        capture_output=True,
        env=subprocess_env(tmp_path),
        text=True,
    )

    cfg = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert str(output_path) in result.stdout
    assert cfg["commodity"]["name"] == "test_corn"
    assert cfg["commodity"]["exchange"] == "TEST"
    assert cfg["data"]["csv_path"] == "examples/corn/sample_data.csv"
    assert cfg["data"]["date_col"] == "date"
    assert cfg["data"]["price_col"] == "close"
