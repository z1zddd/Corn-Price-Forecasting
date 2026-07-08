"""Shared helpers for Keras official-pool specs."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .runtime import KerasSequenceClassifier, KerasSequenceRegressor


KERAS_EPOCHS = 12
KERAS_BATCH_SIZE = 16


def keras_sequence_pair(
    name: str,
    architecture: str,
    params: dict[str, object],
    epochs: int = KERAS_EPOCHS,
    batch_size: int = KERAS_BATCH_SIZE,
    package: str = "tensorflow.keras",
) -> OfficialPoolSpec:
    return OfficialPoolSpec(
        name=name,
        family="deep_sequence",
        package=package,
        classifier_factory=lambda seed: KerasSequenceClassifier(architecture, dict(params), epochs, batch_size, seed),
        regressor_factory=lambda seed: KerasSequenceRegressor(architecture, dict(params), epochs, batch_size, seed),
        classifier_loss="binary_crossentropy",
        regressor_loss="mean_squared_error",
        input_kind="keras_sequence",
        source_kind="official_keras_layers",
    )
