"""Tabular official-pool specs grouped by estimator family."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .bagging import build_bagging_specs
from .boosting import build_boosting_specs
from .forests import build_forest_specs
from .gbdt import build_gbdt_specs
from .kernel import build_kernel_specs
from .linear import build_linear_specs
from .neighbors import build_neighbor_specs
from .neural import build_neural_specs
from .statistical import build_statistical_specs
from .svm import build_svm_specs
from .trees import build_tree_specs


def build_tabular_specs() -> list[OfficialPoolSpec]:
    """Build sklearn/LightGBM/XGBoost/CatBoost entries from package-native estimators."""

    return [
        *build_linear_specs(),
        *build_svm_specs(),
        *build_neighbor_specs(),
        *build_statistical_specs(),
        *build_tree_specs(),
        *build_forest_specs(),
        *build_boosting_specs(),
        *build_bagging_specs(),
        *build_neural_specs(),
        *build_kernel_specs(),
        *build_gbdt_specs(),
    ]
