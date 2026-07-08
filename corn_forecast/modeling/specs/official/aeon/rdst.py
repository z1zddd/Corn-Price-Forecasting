"""RDST official-pool spec."""

from __future__ import annotations

import numpy as np

from ..base import OfficialPoolSpec
from .common import spec_pair


def build_rdst_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_rdst", "tsc_shapelet", "RDSTClassifier", "RDSTRegressor", "aeon.classification.shapelet_based", "aeon.regression.shapelet_based", {"max_shapelets": 256, "shapelet_lengths": np.array([2], dtype=np.int64), "n_jobs": 1}, {"max_shapelets": 256, "shapelet_lengths": np.array([2], dtype=np.int64), "n_jobs": 1}, "random_dilated_shapelet", "random_dilated_shapelet", input_kind="aeon_collection_pad10_float64"),
    ]
