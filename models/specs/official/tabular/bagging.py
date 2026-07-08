"""Bagging official-pool model specs."""

from __future__ import annotations

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

from ..base import OfficialPoolSpec
from .common import make_spec


def build_bagging_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("bagging_tree", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=s), n_estimators=80, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=s), n_estimators=80, max_samples=0.8, random_state=s, n_jobs=1), "bagged_tree_vote", "bagged_tree_regression"),
        make_spec("bagging_extra_tree", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=ExtraTreeClassifier(max_depth=4, class_weight="balanced", random_state=s), n_estimators=100, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=ExtraTreeRegressor(max_depth=4, random_state=s), n_estimators=100, max_samples=0.8, random_state=s, n_jobs=1), "bagged_extra_tree_vote", "bagged_extra_tree_regression"),
        make_spec("bagging_logistic", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), n_estimators=30, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=30, max_samples=0.8, random_state=s, n_jobs=1), "bagged_log_loss", "bagged_ridge"),
    ]
