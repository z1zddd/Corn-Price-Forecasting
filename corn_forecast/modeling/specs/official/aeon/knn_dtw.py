"""DTW KNN official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import spec_pair


def build_knn_dtw_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_knn_dtw", "tsc_distance", "KNeighborsTimeSeriesClassifier", "KNeighborsTimeSeriesRegressor", "aeon.classification.distance_based", "aeon.regression.distance_based", {"n_neighbors": 3, "distance": "dtw", "n_jobs": 1}, {"n_neighbors": 3, "distance": "dtw", "n_jobs": 1}, "dtw_vote", "dtw_mean"),
    ]
