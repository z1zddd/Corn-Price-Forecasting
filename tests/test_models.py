import numpy as np
import pytest

from commodity_backtest.models.registry import create_model


def test_last_return_baseline_predicts_last_feature_change():
    x = np.array(
        [
            [[100.0, 101.0, 103.0]],
            [[103.0, 102.0, 101.0]],
        ]
    )
    y = np.array([1, 0])
    model = create_model({"name": "last_return", "type": "baseline"})

    model.fit(x, y)
    pred = model.predict(x)

    assert pred.tolist() == [1, 0]


def test_random_forest_model_predicts_binary_labels():
    x = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[3.0, 2.0, 1.0]],
            [[2.0, 3.0, 4.0]],
            [[4.0, 3.0, 2.0]],
        ]
    )
    y = np.array([1, 0, 1, 0])
    model = create_model(
        {
            "name": "random_forest",
            "type": "sklearn_random_forest",
            "params": {"n_estimators": 5, "max_depth": 2, "random_state": 42},
        }
    )

    model.fit(x, y)
    pred = model.predict(x)

    assert set(pred.tolist()).issubset({0, 1})


def test_logistic_regression_model_predicts_probabilities_from_string_config():
    x = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[3.0, 2.0, 1.0]],
            [[2.0, 3.0, 4.0]],
            [[4.0, 3.0, 2.0]],
            [[1.0, 3.0, 5.0]],
            [[5.0, 3.0, 1.0]],
        ]
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    model = create_model("logistic_regression")

    model.fit(x, y)
    pred = model.predict(x)
    prob = model.predict_proba(x)

    assert set(pred.tolist()).issubset({0, 1})
    assert prob.shape == (6,)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_name_only_random_forest_config_is_supported():
    model = create_model({"name": "random_forest", "params": {"n_estimators": 2, "random_state": 42}})

    assert model is not None


def test_optional_tree_model_missing_dependency_has_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", None)

    with pytest.raises(ImportError, match="LightGBM is not installed"):
        create_model({"name": "lightgbm", "params": {"n_estimators": 2}})