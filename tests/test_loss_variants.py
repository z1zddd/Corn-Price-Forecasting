from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from backtest.engine import run_backtest
from models.registry import create_model


def small_windows():
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
    y_class = np.array([1, 0, 1, 0, 1, 0])
    y_return = np.array([0.03, -0.02, 0.01, -0.01, 0.04, -0.03])
    return x, y_class, y_return


def test_regression_sign_loss_variants_fit_on_return_targets():
    x, y_class, y_return = small_windows()
    for name in ["regression_mse_sign", "regression_mae_sign", "regression_huber_sign"]:
        model = create_model({"name": name, "params": {"random_state": 42}})

        model.fit_with_targets(x, y_class, y_return)
        pred = model.predict(x)
        raw = model.predict_regression(x)

        assert set(pred.tolist()).issubset({0, 1})
        assert raw.shape == (len(x),)


def test_dual_head_mse_bce_combines_classifier_and_return_regressor():
    x, y_class, y_return = small_windows()
    model = create_model({"name": "dual_head_mse_bce", "params": {"random_state": 42}})

    model.fit_with_targets(x, y_class, y_return)
    pred = model.predict(x)
    prob = model.predict_proba(x)
    raw = model.predict_regression(x)

    assert set(pred.tolist()).issubset({0, 1})
    assert prob.shape == (len(x),)
    assert raw.shape == (len(x),)


def test_focal_logistic_model_predicts_probabilities():
    pytest.importorskip("torch")
    x, y_class, _ = small_windows()
    model = create_model({"name": "focal_logistic", "params": {"epochs": 3, "lr": 0.05, "random_state": 42}})

    model.fit(x, y_class)
    prob = model.predict_proba(x)

    assert prob.shape == (len(x),)
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_backtest_regression_sign_model_writes_r2_health(tmp_path: Path):
    cfg = yaml.safe_load(Path("configs/corn.yaml").read_text(encoding="utf-8"))
    cfg["models"] = ["last_return", {"name": "regression_mse_sign", "params": {"random_state": 42}}]
    cfg["lookback"]["candidates"] = [3]
    output_dir = tmp_path / "loss_variant_run"

    comparison = run_backtest(cfg, output_dir=output_dir)

    assert "R2_health" in comparison.columns
    metrics = pd.read_json(output_dir / "model_outputs" / "regression_mse_sign" / "metrics_summary.json", typ="series")
    assert "R2_health" in metrics.index
