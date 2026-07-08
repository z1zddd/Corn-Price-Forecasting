"""Boosting official-pool model specs."""

from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..base import OfficialPoolSpec
from .common import adaboost_tree_classifier_factory, make_spec


def build_boosting_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("gradient_boosting", "boosting", "sklearn", lambda s: GradientBoostingClassifier(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=s), lambda s: GradientBoostingRegressor(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=s), "deviance", "squared_error_boosting"),
        make_spec("hist_gradient_boosting", "boosting", "sklearn", lambda s: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=s), lambda s: HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=s), "log_loss_hist_gbdt", "squared_error_hist_gbdt"),
        make_spec("hist_gradient_boosting_l2", "boosting", "sklearn", lambda s: HistGradientBoostingClassifier(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=1.0, random_state=s), lambda s: HistGradientBoostingRegressor(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=1.0, random_state=s), "log_loss_hist_l2", "squared_error_hist_l2"),
        make_spec("ada_boost_tree", "boosting", "sklearn", adaboost_tree_classifier_factory(AdaBoostClassifier, DecisionTreeClassifier), lambda s: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=1, random_state=s), n_estimators=80, learning_rate=0.05, random_state=s), "samme", "adaboost_square"),
    ]
