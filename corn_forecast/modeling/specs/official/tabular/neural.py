"""Sklearn neural-network official-pool model specs."""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..base import OfficialPoolSpec
from .common import make_spec


def build_neural_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("mlp_small_relu", "neural_sklearn", "sklearn", lambda s: MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), lambda s: MLPRegressor(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), "log_loss_mlp", "squared_error_mlp"),
        make_spec("mlp_small_tanh", "neural_sklearn", "sklearn", lambda s: MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), lambda s: MLPRegressor(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), "log_loss_mlp", "squared_error_mlp"),
    ]
