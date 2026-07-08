"""Decision-tree official-pool model specs."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

from ..base import OfficialPoolSpec
from .common import make_spec


def build_tree_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("decision_tree_gini", "tree", "sklearn", lambda s: DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: DecisionTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=4, random_state=s), "gini", "squared_error"),
        make_spec("decision_tree_entropy", "tree", "sklearn", lambda s: DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: DecisionTreeRegressor(criterion="absolute_error", max_depth=4, min_samples_leaf=4, random_state=s), "entropy", "absolute_error"),
        make_spec("extra_tree_gini", "tree", "sklearn", lambda s: ExtraTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: ExtraTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=4, random_state=s), "gini_randomized", "squared_error"),
        make_spec("extra_tree_entropy", "tree", "sklearn", lambda s: ExtraTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: ExtraTreeRegressor(criterion="absolute_error", max_depth=4, min_samples_leaf=4, random_state=s), "entropy_randomized", "absolute_error"),
    ]
