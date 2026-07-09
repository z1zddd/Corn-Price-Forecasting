"""Keras official-pool specs and runtime estimators."""

from corn_forecast.operator.model.families.official.base import OfficialPoolSpec
from corn_forecast.operator.model.families.official.keras.bilstm import build_bilstm_specs
from corn_forecast.operator.model.families.official.keras.common import keras_sequence_pair
from corn_forecast.operator.model.families.official.keras.gru import build_gru_specs
from corn_forecast.operator.model.families.official.keras.lstm import build_lstm_specs
from corn_forecast.operator.model.families.official.keras.runtime import (
    KerasSequenceClassifier,
    KerasSequenceRegressor,
    as_int_list,
    build_keras_sequence_model,
    configure_tensorflow_runtime,
    import_keras,
)
from corn_forecast.operator.model.families.official.keras.tcn import build_tcn_specs


def build_keras_sequence_specs() -> list[OfficialPoolSpec]:
    """Build Keras recurrent/TCN variants from official TensorFlow/Keras layers."""

    return [
        *build_lstm_specs(),
        *build_gru_specs(),
        *build_bilstm_specs(),
        *build_tcn_specs(),
    ]

__all__ = [
    "KerasSequenceClassifier",
    "KerasSequenceRegressor",
    "as_int_list",
    "build_bilstm_specs",
    "build_gru_specs",
    "build_keras_sequence_model",
    "build_keras_sequence_specs",
    "build_lstm_specs",
    "build_tcn_specs",
    "configure_tensorflow_runtime",
    "import_keras",
    "keras_sequence_pair",
]
