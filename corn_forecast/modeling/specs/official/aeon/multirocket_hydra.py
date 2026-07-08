"""MultiRocketHydra official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import spec_pair


def build_multirocket_hydra_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_multirocket_hydra", "tsc_convolution", "MultiRocketHydraClassifier", "MultiRocketHydraRegressor", "aeon.classification.convolution_based", "aeon.regression.convolution_based", {"n_kernels": 8, "n_groups": 16, "n_jobs": 1}, {"n_kernels": 8, "n_groups": 16, "n_jobs": 1}, "multirocket_hydra_ridge", "multirocket_hydra_ridge", input_kind="aeon_collection_pad10"),
    ]
