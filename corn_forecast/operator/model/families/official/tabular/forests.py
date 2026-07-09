"""Random-forest official-pool model specs."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..base import OfficialPoolSpec
from .common import make_spec


def build_forest_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("random_forest_100", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=s, n_jobs=1), "gini_bagging", "squared_error_bagging"),
        make_spec("random_forest_balanced", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=120, max_depth=None, min_samples_leaf=4, class_weight="balanced_subsample", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=120, max_depth=None, min_samples_leaf=4, random_state=s, n_jobs=1), "gini_balanced_bagging", "squared_error_bagging"),
        make_spec("random_forest_shallow", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=120, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=120, max_depth=3, min_samples_leaf=3, random_state=s, n_jobs=1), "gini_shallow", "squared_error_shallow"),
    ]
