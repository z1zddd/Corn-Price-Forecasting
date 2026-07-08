"""Bidirectional LSTM official-pool model specs."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import keras_sequence_pair


def build_bilstm_specs() -> list[OfficialPoolSpec]:
    return [
        keras_sequence_pair("keras_bilstm_u16", "bilstm", {"units": 16}),
    ]
