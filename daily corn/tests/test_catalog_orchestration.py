from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing.data_pipeline import build_supervised_samples, make_fixed_split
from scripts.experiments.run_experiment import _sha256, load_config, run_preflight, tune_parameters


def _frame(rows: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    factor = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "dce_corn_close": 2000.0 + factor,
            "factor": factor,
        }
    )


def test_tuning_uses_explicit_seed_and_records_it(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = __import__("scripts.experiments.run_experiment", fromlist=["build_model"])
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")
    config["model"]["fixed_params"]["n_jobs"] = 1
    config["tuning"]["grid"] = {"max_depth": [2, 4]}
    samples = build_supervised_samples(_frame(), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)
    observed_seeds: list[int] = []
    real_build_model = runner.build_model

    def recording_build_model(config, candidate, seed, lookback):
        observed_seeds.append(seed)
        return real_build_model(config, candidate, seed, lookback)

    monkeypatch.setattr(runner, "build_model", recording_build_model)
    _, rows = tune_parameters(
        samples, split, config, "chronological_712", seed=2024
    )

    assert observed_seeds == [2024, 2024]
    assert {row["seed"] for row in rows} == {2024}


def test_each_seed_selects_parameters_from_its_own_validation_rmse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = __import__("scripts.experiments.run_experiment", fromlist=["build_model"])
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")
    config["tuning"]["grid"] = {"choice": [0, 1]}
    samples = build_supervised_samples(_frame(), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)

    class SeedSensitiveRegressor:
        def __init__(self, seed: int, choice: int):
            self.seed = seed
            self.choice = choice

        def fit(self, X, y, validation_data=None):
            self.validation_mean = float(np.mean(validation_data[1]))
            return self

        def predict(self, X):
            preferred = 0 if self.seed == 42 else 1
            penalty = 0.0 if self.choice == preferred else 1000.0
            return np.full(len(X), self.validation_mean + penalty)

    monkeypatch.setattr(
        runner,
        "build_model",
        lambda config, candidate, seed, lookback: SeedSensitiveRegressor(
            seed, candidate["choice"]
        ),
    )

    best_42, rows_42 = tune_parameters(
        samples, split, config, "chronological_712", seed=42
    )
    best_2024, rows_2024 = tune_parameters(
        samples, split, config, "chronological_712", seed=2024
    )

    assert best_42 == {"choice": 0}
    assert best_2024 == {"choice": 1}
    assert {row["seed"] for row in rows_42 + rows_2024} == {42, 2024}


def test_catalog_expands_two_datasets_by_eighteen_models() -> None:
    catalog = yaml.safe_load((PROJECT_ROOT / "configs" / "model_catalog.yaml").read_text(encoding="utf-8"))
    module = __import__("scripts.experiments.run_model_catalog", fromlist=["expand_catalog"])

    tasks = module.expand_catalog(catalog, PROJECT_ROOT)

    assert len(tasks) == 36
    assert {task.config["run"]["dataset_name"] for task in tasks} == {
        "corn_factors_daily_v2",
        "corn_飞天_daily_factors_v2",
    }
    assert all(task.config["formal"]["seeds"] == [42, 2024, 3407] for task in tasks)
    assert len({task.config["run"]["model_name"] for task in tasks}) == 18


def test_catalog_formal_requires_confirmation(tmp_path: Path) -> None:
    module = __import__("scripts.experiments.run_model_catalog", fromlist=["run_catalog"])

    with pytest.raises(ValueError, match="confirm-formal-run"):
        module.run_catalog(
            PROJECT_ROOT / "configs" / "model_catalog.yaml",
            mode="formal",
            confirm_formal_run=False,
            output_dir=tmp_path,
        )


def test_catalog_records_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = __import__("scripts.experiments.run_model_catalog", fromlist=["run_catalog"])
    catalog = yaml.safe_load((PROJECT_ROOT / "configs" / "model_catalog.yaml").read_text(encoding="utf-8"))
    catalog["models"] = catalog["models"][:2]
    catalog["datasets"] = catalog["datasets"][:1]
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    calls = 0

    def fake_preflight(config, project_root):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("expected failure")
        return {"status": "PASS", "total_fits": 7}

    monkeypatch.setattr(module, "run_preflight", fake_preflight)

    summary = module.run_catalog(
        catalog_path,
        mode="preflight",
        confirm_formal_run=False,
        output_dir=tmp_path / "out",
    )

    assert calls == 2
    assert [row["status"] for row in summary] == ["FAILED", "PASS"]
    assert (tmp_path / "out" / "catalog_summary.csv").exists()
    assert json.loads((tmp_path / "out" / "catalog_summary.json").read_text(encoding="utf-8"))[0]["status"] == "FAILED"


def test_preflight_tuning_fit_estimate_multiplies_formal_seeds(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "date": dates,
            "dce_corn_close": 2000.0 + np.arange(len(dates)),
            "factor": np.sin(np.arange(len(dates))),
        }
    ).to_csv(csv_path, index=False)
    config = load_config(PROJECT_ROOT / "configs" / "catalog" / "random_forest.yaml")
    config["data"]["csv_path"] = csv_path.name
    config["data"]["expected_sha256"] = _sha256(csv_path)
    config["target"]["horizons"] = [5]
    config["lookback"]["candidates"] = [5]
    config["tuning"] = {
        "method": "grid_search",
        "metric": "RMSE",
        "grid": {"n_estimators": [5]},
    }
    result = run_preflight(config, project_root=tmp_path)

    assert result["tuning_fits"] == 3


def test_catalog_preflight_only_writes_summary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = __import__("scripts.experiments.run_model_catalog", fromlist=["run_catalog"])
    monkeypatch.setattr(module, "run_preflight", lambda config, project_root: {"status": "PASS"})
    output_dir = tmp_path / "preflight"

    rows = module.run_catalog(
        PROJECT_ROOT / "configs" / "model_catalog.yaml",
        mode="preflight",
        output_dir=output_dir,
    )

    assert len(rows) == 36
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "catalog_summary.csv",
        "catalog_summary.json",
    ]
    assert not (tmp_path / "results").exists()


def test_every_catalog_template_keeps_confirmed_scope() -> None:
    catalog = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "model_catalog.yaml").read_text(encoding="utf-8")
    )
    module = __import__("scripts.experiments.run_model_catalog", fromlist=["expand_catalog"])
    tasks = module.expand_catalog(catalog, PROJECT_ROOT)

    assert len(tasks) == 36
    for task in tasks:
        config = task.config
        assert config["target"]["horizons"] == [5, 10, 15, 20]
        assert config["lookback"]["candidates"] == [5, 10, 15, 20]
        assert list(config["split"]["strategies"]) == ["chronological_712"]
        assert config["split"]["strategies"]["chronological_712"]["ratios"] == [
            0.7,
            0.1,
            0.2,
        ]
        assert config["feature_set"] == {
            "name": "full_safe",
            "external_lag": 0,
            "max_training_missing_rate": 0.5,
        }
        assert config["formal"]["seeds"] == [42, 2024, 3407]
        assert config["tuning"]["method"] == "fixed"
        assert config["tuning"]["selection_seed"] == 42
        assert config["tuning"]["grid"] == {}
        assert config["run"]["runner"] == "model-catalog-autodl"
        assert config["data"]["csv_path"].endswith("_v2.csv")
        assert config["data"]["expected_sha256"] in {
            "B4AF1E478F69FE4BE55C29B5147CA7794CC06BC30823F7A57AE1E5D7DFD98191",
            "2B66E514EABFD598455F7D552074171EABF91327EE043DC0D61A6DFB7DEE1813",
        }


def test_fixed_parameter_plan_does_not_run_validation_tuning() -> None:
    runner = __import__("scripts.experiments.run_experiment", fromlist=["build_parameter_plan"])
    config = load_config(PROJECT_ROOT / "configs" / "catalog" / "rnn.yaml")
    config["tuning"] = {"method": "fixed", "metric": "RMSE", "grid": {}}

    parameters, rows = runner.build_parameter_plan(
        samples=None,
        splits={},
        config=config,
        strategy_names=["chronological_712"],
    )

    assert parameters == {
        42: {"chronological_712": {}},
        2024: {"chronological_712": {}},
        3407: {"chronological_712": {}},
    }
    assert rows == [
        {
            "split_tuning_skeleton": "chronological_712",
            "seed": 42,
            "selection_mode": "fixed_by_codex",
            "validation_rmse": None,
        }
    ]


def test_fixed_parameter_preflight_has_zero_tuning_fits(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "date": dates,
            "dce_corn_close": 2000.0 + np.arange(len(dates)),
            "factor": np.sin(np.arange(len(dates))),
        }
    ).to_csv(csv_path, index=False)
    config = load_config(PROJECT_ROOT / "configs" / "catalog" / "random_forest.yaml")
    config["data"]["csv_path"] = csv_path.name
    config["data"]["expected_sha256"] = _sha256(csv_path)
    config["target"]["horizons"] = [5]
    config["lookback"]["candidates"] = [5]
    config["tuning"] = {"method": "fixed", "metric": "RMSE", "grid": {}}
    config["model"]["fixed_params"]["n_estimators"] = 5

    result = run_preflight(config, project_root=tmp_path)

    assert result["tuning_fits"] == 0


def test_fixed_parameter_mode_does_not_use_validation_for_model_selection() -> None:
    runner = __import__(
        "scripts.experiments.run_experiment", fromlist=["model_fit_validation_data"]
    )
    config = {"tuning": {"method": "fixed"}}
    X_validation = np.ones((3, 2))
    y_validation = np.arange(3, dtype=float)

    assert runner.model_fit_validation_data(
        config, X_validation, y_validation
    ) is None


def test_catalog_partitions_cpu_and_gpu_tasks() -> None:
    catalog = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "model_catalog.yaml").read_text(encoding="utf-8")
    )
    module = __import__(
        "scripts.experiments.run_model_catalog", fromlist=["partition_catalog_tasks"]
    )
    tasks = module.expand_catalog(catalog, PROJECT_ROOT)

    cpu_tasks, gpu_tasks = module.partition_catalog_tasks(tasks)

    assert {task.model_name for task in gpu_tasks} == {
        "xgboost",
        "catboost",
        "lstm",
        "rnn",
        "lnn",
        "simpletm",
        "tcn",
        "raft",
        "mafs",
        "dlinear",
        "xlinear",
        "patchtst",
        "itransformer",
    }
    assert {task.model_name for task in cpu_tasks} == {
        "random_forest",
        "svr",
        "knn",
        "gradient_boosting",
        "lightgbm",
    }
