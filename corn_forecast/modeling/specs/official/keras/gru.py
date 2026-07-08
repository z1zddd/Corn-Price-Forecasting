"""GRU official-pool model specs."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import keras_sequence_pair


def build_gru_specs() -> list[OfficialPoolSpec]:
    return [
        keras_sequence_pair("keras_gru_u16", "gru", {"units": 16}),
    ]
