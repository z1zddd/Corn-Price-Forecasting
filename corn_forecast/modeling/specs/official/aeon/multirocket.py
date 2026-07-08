"""MultiRocket official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_KERNELS, spec_pair


def build_multirocket_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_multirocket", "tsc_convolution", "MultiRocketClassifier", "MultiRocketRegressor", "aeon.classification.convolution_based", "aeon.regression.convolution_based", {"n_kernels": AEON_KERNELS, "n_features_per_kernel": 4, "n_jobs": 1}, {"n_kernels": AEON_KERNELS, "n_features_per_kernel": 4, "n_jobs": 1}, "multirocket_ridge", "multirocket_ridge", input_kind="aeon_collection_pad10"),
    ]
