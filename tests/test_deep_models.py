import numpy as np
import pytest

pytest.importorskip("torch")

from models.registry import create_model


DEEP_MODEL_NAMES = ["lstm", "gru", "transformer", "patchtst", "itransformer", "dlinear"]


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
