"""Euclidean KNN official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import spec_pair


def build_knn_euclidean_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_knn_euclidean", "tsc_distance", "KNeighborsTimeSeriesClassifier", "KNeighborsTimeSeriesRegressor", "aeon.classification.distance_based", "aeon.regression.distance_based", {"n_neighbors": 5, "distance": "euclidean", "n_jobs": 1}, {"n_neighbors": 5, "distance": "euclidean", "n_jobs": 1}, "euclidean_vote", "euclidean_mean"),
    ]
