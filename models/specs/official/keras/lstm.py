"""LSTM official-pool model specs."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import keras_sequence_pair


def build_lstm_specs() -> list[OfficialPoolSpec]:
    return [
        keras_sequence_pair("keras_lstm_u16", "lstm", {"units": 16}),
        keras_sequence_pair("keras_lstm_u32", "lstm", {"units": 32}),
        keras_sequence_pair("keras_lstm_stack2_u32", "lstm", {"units": [32, 16]}),
    ]
