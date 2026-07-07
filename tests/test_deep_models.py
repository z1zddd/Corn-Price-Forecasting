import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from backtest.engine import run_backtest
from models.registry import create_model


DEEP_MODEL_NAMES = ["lstm", "gru", "transformer", "patchtst", "itransformer", "dlinear", "dual_stream_lstm"]


def _tiny_deep_params():
    return {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.01,
        "patience": 1,
        "device": "cpu",
        "random_state": 42,
        "n_heads": 2,
        "patch_len": 2,
        "stride": 1,
    }


def _tiny_sequence_data():
    x = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0]],
            [[0.4, 0.3, 0.2, 0.1], [1.0, 1.0, 1.0, 1.0]],
            [[0.2, 0.3, 0.5, 0.7], [0.0, 0.1, 0.1, 0.2]],
            [[0.8, 0.7, 0.6, 0.5], [0.2, 0.1, 0.1, 0.0]],
            [[0.3, 0.4, 0.6, 0.9], [0.5, 0.5, 0.6, 0.6]],
            [[0.9, 0.6, 0.4, 0.2], [0.6, 0.6, 0.5, 0.5]],
            [[0.2, 0.4, 0.5, 0.8], [0.3, 0.4, 0.4, 0.5]],
            [[0.7, 0.5, 0.3, 0.1], [0.5, 0.4, 0.4, 0.3]],
        ],
        dtype=np.float32,
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    return x, y


def test_deep_model_names_are_registered():
    for name in DEEP_MODEL_NAMES:
        model = create_model({"name": name, "params": _tiny_deep_params()})

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")


def test_lstm_deep_classifier_trains_predicts_and_saves(tmp_path):
    x, y = _tiny_sequence_data()
    model = create_model({"name": "lstm", "params": _tiny_deep_params()})

    model.fit(x[:6], y[:6], x[6:], y[6:])
    prob = model.predict_proba(x)
    pred = model.predict(x)
    model_path = tmp_path / "model.pt"
    model.save(model_path)

    assert prob.shape == (8,)
    assert np.all(np.isfinite(prob))
    assert np.all((prob >= 0.0) & (prob <= 1.0))
    assert set(pred.tolist()).issubset({0, 1})
    assert model_path.exists()


def test_dual_stream_lstm_splits_pca_features_and_predicts():
    feature_cols = ["close", "volume", "basis", "pca_001", "PCA002"]
    x, y = _tiny_sequence_data()
    x = np.concatenate([x, x[:, :1, :] * 0.5, x[:, :2, :] * 0.25], axis=1)
    params = {
        **_tiny_deep_params(),
        "feature_cols": feature_cols,
        "hidden_size": 8,
        "attn_size": 4,
        "dense_size": 8,
    }

    model = create_model({"name": "dual_stream_lstm", "params": params})

    assert model.structured_indices == [0, 1, 2]
    assert model.news_indices == [3, 4]

    model.fit(x[:6], y[:6], x[6:], y[6:])
    prob = model.predict_proba(x)

    assert prob.shape == (8,)
    assert np.all(np.isfinite(prob))
    assert np.all((prob >= 0.0) & (prob <= 1.0))


def test_dual_stream_lstm_without_pca_features_trains_as_structured_only():
    feature_cols = ["close", "volume"]
    x, y = _tiny_sequence_data()
    params = {
        **_tiny_deep_params(),
        "feature_cols": feature_cols,
        "hidden_size": 8,
        "attn_size": 4,
        "dense_size": 8,
    }

    model = create_model({"name": "dual_stream_lstm", "params": params})

    assert model.structured_indices == [0, 1]
    assert model.news_indices == []

    model.fit(x[:6], y[:6], x[6:], y[6:])
    prob = model.predict_proba(x)

    assert prob.shape == (8,)
    assert np.all(np.isfinite(prob))


def test_backtest_passes_auto_numeric_feature_names_to_dual_stream_lstm(tmp_path):
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": [10, 11, 10, 12, 13, 12, 14, 15, 14, 16, 17, 18],
            "volume": [100, 120, 110, 130, 150, 140, 160, 180, 170, 190, 200, 210],
            "basis": [1.0, 1.1, 1.0, 1.2, 1.3, 1.2, 1.4, 1.5, 1.4, 1.6, 1.7, 1.8],
            "pca_001": [0.1, 0.2, 0.0, 0.3, 0.4, 0.2, 0.5, 0.6, 0.4, 0.7, 0.8, 0.9],
            "pca_002": [0.2, 0.1, 0.3, 0.0, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7],
        }
    )
    csv_path = tmp_path / "toy.csv"
    frame.to_csv(csv_path, index=False)
    config = {
        "commodity": {"name": "toy", "exchange": "TEST", "frequency": "monthly"},
        "data": {
            "csv_path": str(csv_path),
            "date_col": "date",
            "price_col": "close",
            "feature_cols": "auto_numeric",
            "exclude_feature_cols": [],
        },
        "target": {"horizon": 1, "mode": "classification", "spike_threshold": 0.0},
        "lookback": {"default": 2, "candidates": [2]},
        "train_window": {"mode": "expanding", "min_train_periods": 4, "stride_periods": 2},
        "split": {"val_ratio": 0.0},
        "evaluation": {"primary_metric": "direction_accuracy", "ci_level": 0.95, "ci_bootstrap_samples": 5},
        "models": [
            {
                "name": "dual_stream_lstm",
                "type": "dual_stream_lstm",
                "params": {
                    "hidden_size": 8,
                    "attn_size": 4,
                    "dense_size": 8,
                    "epochs": 1,
                    "batch_size": 4,
                    "lr": 0.01,
                    "patience": 1,
                    "device": "cpu",
                    "random_state": 42,
                },
            }
        ],
    }

    comparison = run_backtest(config, output_dir=tmp_path / "run")

    assert comparison["model"].tolist() == ["dual_stream_lstm"]
    assert (tmp_path / "run" / "model_outputs" / "dual_stream_lstm" / "predictions.csv").exists()
