from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.machine_learning.RandomForest import RandomForestPriceRegressor
try:
    from models.deep_learning.LSTM import LSTMPriceRegressor
except ModuleNotFoundError:
    LSTMPriceRegressor = None
from scripts.data_processing.data_pipeline import (
    FoldPreprocessor,
    assert_no_temporal_leakage,
    build_supervised_samples,
    iter_expanding_origins,
    make_fixed_split,
)
from scripts.evaluation.evaluate import (
    _economic_metrics,
    _non_overlapping_economics,
    evaluate_predictions,
)
from scripts.experiments.run_experiment import (
    ExperimentContractError,
    build_seed_stability,
    build_model,
    load_config,
    make_parameter_grid,
    raise_formal_failure,
    resolve_strategy_plan,
    run_expanding_setting,
    run_fixed_setting,
    tune_parameters,
    validate_formal_coverage,
)


def make_frame(rows: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    close = 2000.0 + np.arange(rows, dtype=float)
    factor = np.sin(np.arange(rows) / 4.0)
    sparse = np.where(np.arange(rows) % 3 == 0, np.nan, np.arange(rows))
    return pd.DataFrame(
        {
            "date": dates,
            "dce_corn_close": close,
            "factor": factor,
            "sparse_factor": sparse,
        }
    )


def test_supervised_samples_predict_future_price_and_lag_external_factors() -> None:
    frame = make_frame()
    samples = build_supervised_samples(frame, horizon=5, lookback=10)

    first_anchor = 9
    assert samples.metadata.iloc[0]["anchor_date"] == frame.iloc[first_anchor]["date"]
    assert samples.metadata.iloc[0]["target_date"] == frame.iloc[first_anchor + 5]["date"]
    assert samples.y.iloc[0] == frame.iloc[first_anchor + 5]["dce_corn_close"]
    assert samples.X.iloc[0]["dce_corn_close__lag0"] == frame.iloc[first_anchor]["dce_corn_close"]
    assert samples.X.iloc[0]["factor__lag0"] == frame.iloc[first_anchor - 1]["factor"]


def test_preprocessor_uses_training_missingness_only() -> None:
    train = pd.DataFrame(
        {
            "keep": [1.0, np.nan, 3.0, 4.0],
            "drop": [np.nan, np.nan, np.nan, 1.0],
        }
    )
    test = pd.DataFrame({"keep": [np.nan], "drop": [999.0]})
    preprocessor = FoldPreprocessor(max_missing_rate=0.5)

    transformed_train = preprocessor.fit_transform(train)
    transformed_test = preprocessor.transform(test)

    assert preprocessor.selected_columns == ["keep"]
    assert transformed_train.shape[0] == 4
    assert transformed_test.shape[0] == 1
    assert np.isfinite(transformed_test).all()


def test_sequence_preprocessor_keeps_complete_lag_groups_without_indicators() -> None:
    train = pd.DataFrame(
        {
            "keep__lag1": [1.0, np.nan, 3.0, 4.0],
            "drop__lag1": [np.nan, np.nan, np.nan, 1.0],
            "keep__lag0": [2.0, 3.0, np.nan, 5.0],
            "drop__lag0": [np.nan, np.nan, 2.0, np.nan],
        }
    )
    evaluation = pd.DataFrame(
        {
            "keep__lag1": [np.nan],
            "drop__lag1": [9.0],
            "keep__lag0": [6.0],
            "drop__lag0": [9.0],
        }
    )
    preprocessor = FoldPreprocessor(
        max_missing_rate=0.5,
        add_missing_indicators=False,
        preserve_lag_groups=True,
    )

    transformed_train = preprocessor.fit_transform(train)
    transformed_evaluation = preprocessor.transform(evaluation)

    assert preprocessor.selected_columns == ["keep__lag1", "keep__lag0"]
    assert transformed_train.shape == (4, 2)
    assert transformed_evaluation.shape == (1, 2)
    assert np.isfinite(transformed_evaluation).all()


@pytest.mark.parametrize("ratios", [(0.8, 0.1, 0.1), (0.7, 0.1, 0.2)])
def test_fixed_split_purges_unknown_targets(ratios: tuple[float, float, float]) -> None:
    samples = build_supervised_samples(make_frame(), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=ratios, embargo=5)

    first_validation_anchor = samples.metadata.iloc[split.validation_idx[0]]["anchor_date"]
    first_test_anchor = samples.metadata.iloc[split.test_idx[0]]["anchor_date"]
    assert (
        samples.metadata.iloc[split.train_idx]["target_date"] <= first_validation_anchor
    ).all()
    assert (
        samples.metadata.iloc[split.refit_idx]["target_date"] <= first_test_anchor
    ).all()


def test_expanding_origins_only_use_known_targets() -> None:
    samples = build_supervised_samples(make_frame(), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)

    origins = list(iter_expanding_origins(samples, split.test_idx))

    assert len(origins) == len(split.test_idx)
    for origin in origins:
        assert_no_temporal_leakage(
            samples.metadata.iloc[origin.train_idx], origin.prediction_anchor_date
        )
        assert origin.prediction_idx in split.test_idx


def test_random_forest_outputs_prices_and_round_trips(tmp_path: Path) -> None:
    X = np.arange(120, dtype=float).reshape(40, 3)
    y = 1800.0 + X[:, 0] * 0.5
    model = RandomForestPriceRegressor(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X, y)
    predictions = model.predict(X[:5])
    checkpoint = tmp_path / "rf.joblib"
    model.save(checkpoint)
    loaded = RandomForestPriceRegressor.load(checkpoint)

    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, loaded.predict(X[:5]))


@pytest.mark.skipif(LSTMPriceRegressor is None, reason="PyTorch is not installed")
def test_lstm_outputs_prices_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(48, 20)).astype(np.float32)
    y = 2000.0 + 5.0 * X[:, -1] + np.arange(48, dtype=float) * 0.1
    model = LSTMPriceRegressor(
        lookback=5,
        hidden_size=8,
        num_layers=1,
        learning_rate=0.001,
        batch_size=8,
        max_epochs=3,
        patience=2,
        weight_decay=0.0,
        gradient_clip=1.0,
        random_state=42,
        device="cpu",
    )
    model.fit(X[:40], y[:40], validation_data=(X[40:], y[40:]))
    predictions = model.predict(X[40:])
    checkpoint = tmp_path / "lstm.pt"
    model.save(checkpoint)
    loaded = LSTMPriceRegressor.load(checkpoint)

    assert predictions.shape == (8,)
    assert np.isfinite(predictions).all()
    assert model.best_epoch_ >= 1
    np.testing.assert_allclose(predictions, loaded.predict(X[40:]), rtol=1e-5)


@pytest.mark.skipif(LSTMPriceRegressor is None, reason="PyTorch is not installed")
def test_lstm_refit_without_validation_keeps_selected_final_epoch() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(48, 20)).astype(np.float32)
    y = 2000.0 + rng.normal(size=48) * 20.0
    model = LSTMPriceRegressor(
        lookback=5,
        hidden_size=8,
        num_layers=1,
        learning_rate=1.0,
        batch_size=8,
        max_epochs=5,
        patience=2,
        weight_decay=0.0,
        gradient_clip=1.0,
        random_state=42,
        device="cpu",
    )

    model.fit(X, y)

    assert model.best_epoch_ == 5


@pytest.mark.skipif(LSTMPriceRegressor is None, reason="PyTorch is not installed")
def test_model_factory_builds_lstm_from_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "lstm.yaml")
    model = build_model(
        config,
        {"hidden_size": 32, "learning_rate": 0.001},
        seed=42,
        lookback=5,
    )

    assert isinstance(model, LSTMPriceRegressor)
    assert model.params["lookback"] == 5


@pytest.mark.skipif(LSTMPriceRegressor is None, reason="PyTorch is not installed")
def test_lstm_tuning_returns_validation_selected_epoch() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "lstm.yaml")
    config["model"]["fixed_params"]["hidden_size"] = 8
    config["model"]["fixed_params"]["max_epochs"] = 2
    config["model"]["fixed_params"]["patience"] = 2
    config["tuning"]["grid"] = {
        "hidden_size": [8],
        "learning_rate": [0.001],
    }
    samples = build_supervised_samples(make_frame(100), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)

    best, rows = tune_parameters(samples, split, config, "chronological_712")

    assert len(rows) == 1
    assert 1 <= best["max_epochs"] <= 2
    assert np.isfinite(rows[0]["validation_rmse"])


def test_evaluation_derives_trend_and_non_overlapping_economics() -> None:
    predictions = pd.DataFrame(
        {
            "anchor_date": pd.bdate_range("2024-01-02", periods=12),
            "close_t": np.full(12, 100.0),
            "actual_dce_corn_close": [102, 98] * 6,
            "predicted_dce_corn_close": [101, 99] * 6,
        }
    )

    metrics, enriched = evaluate_predictions(
        predictions, horizon=2, mase_scale=1.0, transaction_cost_bps=[0, 2]
    )

    assert enriched["actual_trend"].tolist() == [1, 0] * 6
    assert enriched["predicted_trend_threshold_0"].tolist() == [1, 0] * 6
    assert enriched["predicted_trend_selected"].tolist() == [1, 0] * 6
    assert metrics["price_rmse"] > 0
    assert metrics["trend_direction_accuracy"] == 1.0
    assert "economic_selected_0bp_mean_sharpe" in metrics
    assert "economic_selected_2bp_worst_cumulative_return" in metrics


def test_random_forest_config_contains_confirmed_experiment_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")
    grid = make_parameter_grid(config["tuning"]["grid"])

    assert config["target"]["horizons"] == [5, 10, 15, 20]
    assert config["lookback"]["candidates"] == [5, 10, 15, 20]
    assert list(config["split"]["strategies"]) == ["chronological_712"]
    assert config["feature_set"]["name"] == "market_factors_lag1"
    assert config["tuning"]["metric"] == "RMSE"
    assert config["tuning"]["seed"] == 42
    assert config["formal"]["seeds"] == [42]
    assert config["run"]["runner"] == "zzm"
    assert len(grid) == 8


def test_lstm_config_contains_confirmed_experiment_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "lstm.yaml")
    grid = make_parameter_grid(config["tuning"]["grid"])

    assert config["target"]["horizons"] == [5, 10]
    assert config["lookback"]["candidates"] == [5, 10]
    assert list(config["split"]["strategies"]) == ["chronological_712"]
    assert config["preprocessing"] == {
        "add_missing_indicators": False,
        "preserve_lag_groups": True,
    }
    assert config["formal"]["seeds"] == [42]
    assert config["run"]["runner"] == "zzm"
    assert len(grid) == 4


def test_strategy_plan_uses_only_configured_strategy() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")

    plan = resolve_strategy_plan(config)

    assert plan == {
        "configured": ["chronological_712"],
        "fixed": ["chronological_712"],
        "tuning": ["chronological_712"],
        "run_expanding": False,
    }


def test_formal_setting_failure_is_recorded_and_raised(tmp_path: Path) -> None:
    failures: list[dict[str, object]] = []

    with pytest.raises(ExperimentContractError, match="chronological_712"):
        raise_formal_failure(
            tmp_path,
            failures,
            horizon=5,
            lookback=10,
            split_strategy="chronological_712",
            seed=42,
            error=RuntimeError("fit failed"),
        )

    recorded = pd.read_csv(tmp_path / "model_failures.csv")
    assert len(recorded) == 1
    assert recorded.iloc[0]["split_strategy"] == "chronological_712"
    assert "fit failed" in recorded.iloc[0]["error"]


def test_fixed_and_expanding_runs_emit_complete_audits(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")
    config["model"]["fixed_params"]["n_jobs"] = 1
    samples = build_supervised_samples(make_frame(100), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)
    params = {
        "n_estimators": 10,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    _, _, fixed_audit = run_fixed_setting(
        samples,
        split,
        config,
        params,
        seed=42,
        checkpoint_path=tmp_path / "fixed.joblib",
    )
    expanding_predictions, _, expanding_audits, final_selected_columns = run_expanding_setting(
        samples,
        split.test_idx,
        config,
        params,
        seed=42,
        checkpoint_path=tmp_path / "expanding.joblib",
        limit_origins=2,
    )

    assert fixed_audit["validation_start_date"] <= fixed_audit["validation_end_date"]
    assert fixed_audit["selected_columns"]
    assert len(expanding_predictions) == 2
    assert len(expanding_audits) == 2
    assert all(audit["selected_columns_hash"] for audit in expanding_audits)
    assert final_selected_columns


def test_economic_metrics_only_keep_framework_metrics() -> None:
    returns = np.asarray([-0.10, 0.05])
    metrics = _economic_metrics(returns, horizon=2)

    assert metrics["max_drawdown"] == pytest.approx(0.10)
    assert set(metrics) == {
        "cumulative_return",
        "annualized_return",
        "sharpe",
        "max_drawdown",
    }


def test_worst_economic_aggregation_respects_metric_direction() -> None:
    frame = pd.DataFrame(
        {
            "actual_return": [-0.20, -0.05, 0.0, 0.0],
            "predicted_trend": [1, 1, 1, 1],
        }
    )
    metrics = _non_overlapping_economics(frame, horizon=2, transaction_cost_bps=[0])

    assert metrics["economic_0bp_worst_max_drawdown"] == pytest.approx(0.20)
    assert metrics["economic_0bp_worst_cumulative_return"] == pytest.approx(-0.20)


def test_coverage_validator_rejects_missing_formal_combinations() -> None:
    config = {
        "target": {"horizons": [5]},
        "lookback": {"candidates": [5]},
        "split": {
            "strategies": {
                "chronological_811": {},
                "chronological_712": {},
                "expanding_rolling_backtest": {},
            }
        },
        "formal": {"seeds": [42]},
    }
    expected_counts = {
        (5, 5, "chronological_811"): 2,
        (5, 5, "chronological_712"): 2,
        (5, 5, "expanding_rolling_backtest"): 2,
    }
    metric_rows = [
        {
            "horizon": 5,
            "lookback": 5,
            "split_strategy": strategy,
            "seed": 42,
            "n_predictions": 2,
        }
        for _, _, strategy in expected_counts
    ]
    prediction_rows = [
        {
            "horizon": 5,
            "lookback": 5,
            "split_strategy": strategy,
            "seed": 42,
            "anchor_date": date,
            "predicted_dce_corn_close": 2000.0,
        }
        for _, _, strategy in expected_counts
        for date in pd.bdate_range("2024-01-02", periods=2)
    ]

    validate_formal_coverage(
        pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), config, expected_counts
    )
    with pytest.raises(ExperimentContractError):
        validate_formal_coverage(
            pd.DataFrame(metric_rows[:-1]),
            pd.DataFrame(prediction_rows),
            config,
            expected_counts,
        )


def test_seed_stability_reports_mean_std_and_directional_worst() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["random_forest"] * 2,
            "feature_set": ["market_factors_lag1"] * 2,
            "horizon": [5, 5],
            "lookback": [5, 5],
            "split_strategy": ["chronological_811"] * 2,
            "seed": [42, 2024],
            "n_predictions": [2, 2],
            "price_rmse": [1.0, 3.0],
            "trend_direction_accuracy": [0.7, 0.5],
            "economic_0bp_mean_max_drawdown": [0.1, 0.3],
        }
    )
    stability = build_seed_stability(metrics)
    row = stability.iloc[0]

    assert row["price_rmse_mean"] == pytest.approx(2.0)
    assert row["price_rmse_worst"] == pytest.approx(3.0)
    assert row["trend_direction_accuracy_worst"] == pytest.approx(0.5)
    assert row["economic_0bp_mean_max_drawdown_worst"] == pytest.approx(0.3)


def test_expanding_run_resumes_without_duplicate_origins(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")
    config["model"]["fixed_params"]["n_jobs"] = 1
    samples = build_supervised_samples(make_frame(100), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)
    params = {
        "n_estimators": 10,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    progress = tmp_path / "progress"

    first_predictions, _, _, _ = run_expanding_setting(
        samples,
        split.test_idx,
        config,
        params,
        seed=42,
        checkpoint_path=tmp_path / "expanding.joblib",
        limit_origins=2,
        progress_dir=progress,
        flush_interval=1,
    )
    resumed_predictions, _, resumed_audits, _ = run_expanding_setting(
        samples,
        split.test_idx,
        config,
        params,
        seed=42,
        checkpoint_path=tmp_path / "expanding.joblib",
        progress_dir=progress,
        flush_interval=2,
    )

    assert len(first_predictions) == 2
    assert len(resumed_predictions) == len(split.test_idx)
    assert resumed_predictions["sample_position"].is_unique
    assert len(resumed_audits) == len(split.test_idx)
    state = json.loads((progress / "progress_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "COMPLETED"


def test_full_safe_applies_lag_only_to_unadjusted_external_fields() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates,
            "dce_corn_close": np.arange(100.0, 108.0),
            "price_momentum_1d": np.arange(10.0, 18.0),
            "cbot_corn_close": np.arange(200.0, 208.0),
            "basis_rate_level_lag1d": np.arange(300.0, 308.0),
        }
    )

    samples = build_supervised_samples(
        frame, horizon=1, lookback=1, feature_set="full_safe"
    )
    second = samples.X.iloc[1]

    assert second["price_momentum_1d__lag0"] == frame.iloc[1]["price_momentum_1d"]
    assert second["cbot_corn_close__lag0"] == frame.iloc[0]["cbot_corn_close"]
    assert (
        second["basis_rate_level_lag1d__lag0"]
        == frame.iloc[1]["basis_rate_level_lag1d"]
    )


def test_full_safe_rejects_future_or_target_features() -> None:
    frame = make_frame(20)
    frame["target_future_return"] = 1.0

    with pytest.raises(ValueError, match="Forbidden full_safe"):
        build_supervised_samples(
            frame, horizon=1, lookback=1, feature_set="full_safe"
        )


def test_fixed_setting_never_refits_with_validation_data(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "random_forest.yaml")
    config["model"]["fixed_params"]["n_jobs"] = 1
    samples = build_supervised_samples(make_frame(100), horizon=5, lookback=5)
    split = make_fixed_split(samples, ratios=(0.7, 0.1, 0.2), embargo=5)

    predictions, _, audit = run_fixed_setting(
        samples,
        split,
        config,
        {
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        seed=42,
        checkpoint_path=tmp_path / "fixed.joblib",
    )

    assert audit["n_train"] == len(split.train_idx)
    assert audit["train_end_date"] < samples.metadata.iloc[split.validation_idx[0]][
        "anchor_date"
    ]
    assert {
        "selected_threshold",
        "threshold_calibration_end_date",
        "n_threshold_samples",
    }.issubset(predictions.columns)


def test_threshold_selection_uses_ba_then_mcc_then_closest_to_zero() -> None:
    evaluation = importlib.import_module("scripts.evaluation.evaluate")
    select_threshold = getattr(evaluation, "select_trend_threshold")
    actual_return = np.asarray([0.02, -0.01, 0.03, -0.02])
    predicted_return = np.asarray([0.03, 0.005, 0.04, 0.005])

    selected, summary, audit = select_threshold(
        actual_return,
        predicted_return,
        candidates=[0.00, 0.01, 0.02],
    )

    assert selected == pytest.approx(0.01)
    assert summary["validation_ba_selected"] == pytest.approx(1.0)
    assert len(audit) == 3
    assert sum(bool(row["is_selected"]) for row in audit) == 1
    assert all(
        row["selection_rule"] == "balanced_accuracy,higher_mcc,closer_to_zero"
        for row in audit
    )


def test_evaluation_reports_only_framework_price_and_economic_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "anchor_date": pd.bdate_range("2024-01-02", periods=12),
            "close_t": np.full(12, 100.0),
            "actual_dce_corn_close": [102, 98] * 6,
            "predicted_dce_corn_close": [103, 101] * 6,
        }
    )

    metrics, enriched = evaluate_predictions(
        predictions,
        horizon=2,
        mase_scale=1.0,
        transaction_cost_bps=[0, 2],
        selected_threshold=0.01,
    )

    assert {"price_mae", "price_rmse", "price_r2"}.issubset(metrics)
    assert not any(
        key.startswith(("price_mape", "price_smape", "price_mase"))
        for key in metrics
    )
    assert {
        "predicted_trend_threshold_0",
        "predicted_trend_selected",
        "selected_threshold",
    }.issubset(enriched.columns)
    assert not any(
        token in key
        for key in metrics
        for token in ("sortino", "calmar", "profit_factor", "win_rate")
    )


def test_xgboost_outputs_prices_and_round_trips(tmp_path: Path) -> None:
    import xgboost

    module = importlib.import_module("models.machine_learning.XGBoost")
    model_class = getattr(module, "XGBoostPriceRegressor")
    X = np.arange(180, dtype=float).reshape(60, 3)
    y = 1800.0 + X[:, 0] * 0.5
    model = model_class(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X[:50], y[:50], validation_data=(X[50:], y[50:]))
    checkpoint = tmp_path / "xgboost.joblib"
    model.save(checkpoint)
    loaded = model_class.load(checkpoint)

    predictions = model.predict(X[50:])
    assert model_class.runtime_version == xgboost.__version__
    assert predictions.shape == (10,)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions, loaded.predict(X[50:]))


def test_xgboost_config_matches_confirmed_scope() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "xgboost.yaml")
    grid = make_parameter_grid(config["tuning"]["grid"])

    assert config["target"]["horizons"] == [5, 10, 15, 20]
    assert config["lookback"]["candidates"] == [5, 10, 15, 20, 30, 40, 60]
    assert list(config["split"]["strategies"]) == [
        "chronological_811",
        "chronological_712",
    ]
    assert config["feature_set"]["name"] == "full_safe"
    assert config["formal"]["seeds"] == [42]
    assert config["model"]["source_version"] == "3.2.0"
    assert config["model"]["fixed_params"]["device"] == "cuda"
    assert len(grid) == 8


def test_xgboost_runtime_version_must_match_config() -> None:
    import xgboost

    runner = importlib.import_module("scripts.experiments.run_experiment")
    validate_runtime = getattr(runner, "validate_model_runtime")
    config = load_config(PROJECT_ROOT / "configs" / "xgboost.yaml")
    config["model"]["source_version"] = "0.0.0"
    with pytest.raises(RuntimeError, match="XGBoost runtime version"):
        validate_runtime(config)

    config["model"]["source_version"] = xgboost.__version__
    validate_runtime(config)
