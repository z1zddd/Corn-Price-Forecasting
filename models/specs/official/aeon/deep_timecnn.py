"""Aeon TimeCNN official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_DEEP_BATCH_SIZE, AEON_DEEP_EPOCHS, deep_pair


def build_deep_timecnn_specs() -> list[OfficialPoolSpec]:
    return [
        deep_pair("aeon_deep_timecnn", "TimeCNNClassifier", "TimeCNNRegressor", AEON_DEEP_EPOCHS, AEON_DEEP_BATCH_SIZE, {"kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16]}, {"kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16]}),
    ]
