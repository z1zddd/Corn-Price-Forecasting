"""Nearest-neighbor official-pool model specs."""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid

from ..base import OfficialPoolSpec
from .common import make_spec


def build_neighbor_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("knn_3_uniform", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=3, weights="uniform"), lambda s: KNeighborsRegressor(n_neighbors=3, weights="uniform"), "vote", "neighbor_mean"),
        make_spec("knn_5_distance", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=5, weights="distance"), lambda s: KNeighborsRegressor(n_neighbors=5, weights="distance"), "vote_distance", "neighbor_distance"),
        make_spec("knn_9_distance", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=9, weights="distance"), lambda s: KNeighborsRegressor(n_neighbors=9, weights="distance"), "vote_distance", "neighbor_distance"),
        make_spec("nearest_centroid", "neighbors", "sklearn", lambda s: NearestCentroid(), lambda s: Ridge(alpha=1.0), "centroid_distance", "ridge_l2"),
    ]
