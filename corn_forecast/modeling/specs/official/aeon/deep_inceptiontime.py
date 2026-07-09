"""Aeon InceptionTime official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_DEEP_BATCH_SIZE, AEON_DEEP_EPOCHS, deep_pair


def build_deep_inceptiontime_specs() -> list[OfficialPoolSpec]:
    return [
        deep_pair("aeon_deep_inceptiontime", "InceptionTimeClassifier", "InceptionTimeRegressor", AEON_DEEP_EPOCHS, AEON_DEEP_BATCH_SIZE, {"n_classifiers": 1, "n_filters": 16, "kernel_size": 3, "depth": 3, "bottleneck_size": 8}, {"n_regressors": 1, "n_filters": 16, "kernel_size": 3, "depth": 3, "bottleneck_size": 8}),
    ]
