from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


MODEL_CLASSES = {
    "RandomForestPriceRegressor",
    "XGBoostPriceRegressor",
    "LSTMPriceRegressor",
    "SVRPriceRegressor",
    "KNNPriceRegressor",
    "GradientBoostingPriceRegressor",
    "LightGBMPriceRegressor",
    "CatBoostPriceRegressor",
    "RNNPriceRegressor",
    "LNNPriceRegressor",
    "SimpleTMPriceRegressor",
    "TCNPriceRegressor",
    "RAFTPriceRegressor",
    "MAFSPriceRegressor",
    "DLinearPriceRegressor",
    "XLinearPriceRegressor",
    "PatchTSTPriceRegressor",
    "ITransformerPriceRegressor",
}


ADAPTERS = [
    ("models.machine_learning.SVR", "SVRPriceRegressor", {}, "joblib"),
    (
        "models.machine_learning.KNNRegressor",
        "KNNPriceRegressor",
        {"n_neighbors": 3},
        "joblib",
    ),
    (
        "models.machine_learning.GradientBoosting",
        "GradientBoostingPriceRegressor",
        {"n_estimators": 8, "max_depth": 2, "random_state": 42},
        "joblib",
    ),
    (
        "models.machine_learning.LightGBM",
        "LightGBMPriceRegressor",
        {"n_estimators": 8, "random_state": 42, "verbosity": -1},
        "joblib",
    ),
    (
        "models.machine_learning.CatBoost",
        "CatBoostPriceRegressor",
        {"iterations": 8, "random_state": 42, "verbose": False},
        "joblib",
    ),
    ("models.deep_learning.RNN", "RNNPriceRegressor", {}, "pt"),
    ("models.deep_learning.LNN", "LNNPriceRegressor", {}, "pt"),
    ("models.deep_learning.SimpleTM", "SimpleTMPriceRegressor", {}, "pt"),
    ("models.deep_learning.TCN", "TCNPriceRegressor", {}, "pt"),
    ("models.deep_learning.RAFT", "RAFTPriceRegressor", {}, "pt"),
    ("models.deep_learning.MAFS", "MAFSPriceRegressor", {}, "pt"),
    ("models.deep_learning.DLinear", "DLinearPriceRegressor", {}, "pt"),
    ("models.deep_learning.XLinear", "XLinearPriceRegressor", {}, "pt"),
    ("models.deep_learning.PatchTST", "PatchTSTPriceRegressor", {}, "pt"),
    (
        "models.deep_learning.iTransformer",
        "ITransformerPriceRegressor",
        {},
        "pt",
    ),
]


def _dependency_available(module_name: str) -> bool:
    dependency = {
        "models.machine_learning.LightGBM": "lightgbm",
        "models.machine_learning.CatBoost": "catboost",
    }.get(module_name)
    return dependency is None or importlib.util.find_spec(dependency) is not None


def test_registry_resolves_all_catalog_classes() -> None:
    registry = importlib.import_module("scripts.experiments.model_registry")

    assert set(registry.registered_model_classes()) == MODEL_CLASSES
    for class_name in MODEL_CLASSES:
        resolved = registry.resolve_model_class(class_name)
        assert resolved.__name__ == class_name


def test_runner_builds_registered_deep_model_with_lookback() -> None:
    runner = importlib.import_module("scripts.experiments.run_experiment")
    config = {
        "model": {
            "class_name": "RNNPriceRegressor",
            "fixed_params": {"max_epochs": 1, "device": "cpu"},
        },
        "preprocessing": {
            "add_missing_indicators": False,
            "preserve_lag_groups": True,
        },
    }

    model = runner.build_model(config, {}, seed=42, lookback=4)

    assert model.__class__.__name__ == "RNNPriceRegressor"
    assert model.lookback == 4
    assert model.random_state == 42


def test_preflight_shrinking_does_not_inject_deep_parameters_into_svr() -> None:
    runner = importlib.import_module("scripts.experiments.run_experiment")
    config = {
        "model": {
            "class_name": "SVRPriceRegressor",
            "fixed_params": {"C": 1.0},
        }
    }

    smoke_config, smoke_candidate = runner.make_preflight_model_config(
        config, {"epsilon": 0.1}
    )

    assert smoke_config["model"]["fixed_params"] == {"C": 1.0}
    assert smoke_candidate == {"epsilon": 0.1}


def test_preflight_shrinking_only_limits_parameters_that_exist() -> None:
    runner = importlib.import_module("scripts.experiments.run_experiment")
    config = {
        "model": {
            "class_name": "RNNPriceRegressor",
            "fixed_params": {"max_epochs": 60, "patience": 8},
        }
    }

    smoke_config, _ = runner.make_preflight_model_config(config, {})

    assert smoke_config["model"]["fixed_params"]["max_epochs"] == 2
    assert smoke_config["model"]["fixed_params"]["patience"] == 2
    assert set(smoke_config["model"]["fixed_params"]) == {"max_epochs", "patience"}


@pytest.mark.parametrize(
    "preprocessing",
    [
        {},
        {"add_missing_indicators": True, "preserve_lag_groups": True},
        {"add_missing_indicators": False, "preserve_lag_groups": False},
    ],
)
def test_sequence_models_reject_preprocessing_that_breaks_lag_groups(
    preprocessing: dict[str, bool],
) -> None:
    runner = importlib.import_module("scripts.experiments.run_experiment")
    config = {
        "model": {"class_name": "RNNPriceRegressor", "fixed_params": {}},
        "preprocessing": preprocessing,
    }

    with pytest.raises(ValueError, match="add_missing_indicators=False.*preserve_lag_groups=True"):
        runner.build_model(config, {}, seed=42, lookback=4)


def test_sequence_fold_with_missing_values_keeps_complete_flattened_windows() -> None:
    pipeline = importlib.import_module("scripts.data_processing.data_pipeline")
    runner = importlib.import_module("scripts.experiments.run_experiment")
    rows = 60
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=rows),
            "dce_corn_close": 2000.0 + np.arange(rows, dtype=float),
            "factor": np.where(np.arange(rows) % 7 == 0, np.nan, np.arange(rows)),
        }
    )
    samples = pipeline.build_supervised_samples(frame, horizon=2, lookback=4)
    split = pipeline.make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=2)
    config = {
        "model": {"class_name": "RNNPriceRegressor", "fixed_params": {}},
        "preprocessing": {
            "add_missing_indicators": False,
            "preserve_lag_groups": True,
        },
        "feature_set": {"max_training_missing_rate": 0.5},
    }

    preprocessor, X_train, X_validation = runner._prepare_fold(
        samples, split.train_idx, split.validation_idx, config
    )

    assert X_train.shape[1] % 4 == 0
    assert X_validation.shape[1] == X_train.shape[1]
    assert not any(column.endswith("__missing") for column in preprocessor.selected_columns)


def test_boosting_dependency_versions_are_reported_without_model_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = importlib.import_module("scripts.experiments.model_registry")
    requested: list[str] = []

    def fake_version(distribution: str) -> str:
        requested.append(distribution)
        return {"xgboost": "3.2.0", "lightgbm": "4.6.0", "catboost": "1.2.8"}[distribution]

    monkeypatch.setattr(registry.importlib_metadata, "version", fake_version)
    details = {
        name: registry.model_dependency_manifest(name)
        for name in (
            "XGBoostPriceRegressor",
            "LightGBMPriceRegressor",
            "CatBoostPriceRegressor",
        )
    }

    assert requested == ["xgboost", "lightgbm", "catboost"]
    assert details["XGBoostPriceRegressor"] == {"name": "xgboost", "version": "3.2.0"}
    assert details["LightGBMPriceRegressor"]["version"] == "4.6.0"
    assert details["CatBoostPriceRegressor"]["version"] == "1.2.8"


def test_benchmark_times_every_grid_candidate_and_returns_worst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module("scripts.experiments.run_experiment")
    fitted: list[tuple[dict[str, object], object]] = []

    class FakeModel:
        def __init__(self, candidate: dict[str, object]) -> None:
            self.candidate = candidate

        def fit(self, X, y, validation_data=None):
            fitted.append((self.candidate, validation_data))
            return self

    monkeypatch.setattr(
        runner,
        "build_model",
        lambda config, candidate, seed, lookback: FakeModel(dict(candidate)),
    )
    ticks = iter([0.0, 0.5, 1.0, 3.0, 4.0, 5.5])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    config = {
        "model": {"class_name": "SVRPriceRegressor", "fixed_params": {}},
        "tuning": {"grid": {"C": [0.5, 1.0, 2.0]}},
    }
    X = np.ones((8, 4), dtype=float)
    y = np.arange(8, dtype=float)

    worst, benchmark_epochs, epoch_scale = runner.benchmark_grid_fits(
        config, X, y, X[:2], y[:2], lookback=1
    )

    assert len(fitted) == 3
    assert [item[0]["C"] for item in fitted] == [0.5, 1.0, 2.0]
    assert all(item[1] is not None for item in fitted)
    assert worst == pytest.approx(2.0)
    assert benchmark_epochs is None
    assert epoch_scale == 1.0


@pytest.mark.parametrize("module_name,class_name,params,suffix", ADAPTERS)
def test_new_adapter_outputs_finite_prices_and_round_trips(
    tmp_path: Path,
    module_name: str,
    class_name: str,
    params: dict[str, object],
    suffix: str,
) -> None:
    if not _dependency_available(module_name):
        pytest.skip(f"optional dependency for {class_name} is not installed")
    if module_name.startswith("models.deep_learning") and importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    rng = np.random.default_rng(42)
    lookback = 4
    X = rng.normal(size=(24, lookback * 3)).astype(np.float32)
    y = 2000.0 + 2.0 * X[:, -1] + np.arange(len(X), dtype=float) * 0.05
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    model_params = dict(params)
    if module_name.startswith("models.deep_learning"):
        model_params.update(
            {
                "lookback": lookback,
                "hidden_size": 8,
                "batch_size": 8,
                "max_epochs": 1,
                "patience": 1,
                "learning_rate": 0.001,
                "random_state": 42,
                "device": "cpu",
            }
        )
    model = model_class(**model_params)
    model.fit(X[:18], y[:18], validation_data=(X[18:], y[18:]))
    predictions = model.predict(X[18:])
    checkpoint = tmp_path / f"{class_name}.{suffix}"
    model.save(checkpoint)
    loaded = model_class.load(checkpoint)

    assert predictions.shape == (6,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, loaded.predict(X[18:]), rtol=1e-5, atol=1e-5)


def test_deep_adapter_rejects_width_not_divisible_by_lookback() -> None:
    module = importlib.import_module("models.deep_learning.RNN")
    model = module.RNNPriceRegressor(lookback=4, max_epochs=1, device="cpu")

    with pytest.raises(ValueError, match="divisible by lookback"):
        model.fit(np.ones((8, 10)), np.arange(8, dtype=float))


def test_dlinear_uses_configured_kernel_size_and_individual_linear_heads() -> None:
    module = importlib.import_module("models.deep_learning.DLinear")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 4 * 3)).astype(np.float32)
    y = 2000.0 + rng.normal(size=12)
    model = module.DLinearPriceRegressor(
        lookback=4,
        kernel_size=7,
        individual=True,
        batch_size=4,
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    model.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]))

    assert model.model is not None
    assert model.model.kernel_size == 7
    assert model.model.individual is True
    assert len(model.model.trend) == 3
    assert len(model.model.seasonal) == 3


def test_optional_dependency_is_loaded_only_when_adapter_is_instantiated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("models.machine_learning.LightGBM")
    real_import = importlib.import_module

    def fail_lightgbm(name: str, package: str | None = None):
        if name == "lightgbm":
            raise ModuleNotFoundError("simulated missing lightgbm")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_lightgbm)
    with pytest.raises(ImportError, match="pip install lightgbm"):
        module.LightGBMPriceRegressor()
