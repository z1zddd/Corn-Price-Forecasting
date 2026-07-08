"""MiniRocket official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_KERNELS, spec_pair


def build_minirocket_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_minirocket", "tsc_convolution", "MiniRocketClassifier", "MiniRocketRegressor", "aeon.classification.convolution_based", "aeon.regression.convolution_based", {"n_kernels": AEON_KERNELS, "n_jobs": 1}, {"n_kernels": AEON_KERNELS, "n_jobs": 1}, "minirocket_ridge", "minirocket_ridge", input_kind="aeon_collection_pad10"),
    ]
