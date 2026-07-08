"""TCN official-pool model specs."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import keras_sequence_pair


def build_tcn_specs() -> list[OfficialPoolSpec]:
    return [
        keras_sequence_pair("keras_tcn_filters8_k2_d1", "tcn", {"nb_filters": 8, "kernel_size": 2, "dilations": (1,)}, package="keras-tcn"),
        keras_sequence_pair("keras_tcn_filters16_k2_d1", "tcn", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,)}, package="keras-tcn"),
    ]
