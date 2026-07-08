"""Aeon deep MLP official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_DEEP_BATCH_SIZE, AEON_DEEP_EPOCHS, deep_pair


def build_deep_mlp_specs() -> list[OfficialPoolSpec]:
    return [
        deep_pair("aeon_deep_mlp", "MLPClassifier", "MLPRegressor", AEON_DEEP_EPOCHS, AEON_DEEP_BATCH_SIZE, {"n_layers": 2, "n_units": 64}, {"n_layers": 2, "n_units": 64}),
    ]
