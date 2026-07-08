"""Aeon FCN official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_DEEP_BATCH_SIZE, AEON_DEEP_EPOCHS, deep_pair


def build_deep_fcn_specs() -> list[OfficialPoolSpec]:
    return [
        deep_pair("aeon_deep_fcn", "FCNClassifier", "FCNRegressor", AEON_DEEP_EPOCHS, AEON_DEEP_BATCH_SIZE, {"n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]}, {"n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]}),
    ]
