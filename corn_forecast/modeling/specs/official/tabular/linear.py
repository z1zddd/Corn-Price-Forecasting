"""Linear official-pool model specs."""

from __future__ import annotations

from sklearn.linear_model import HuberRegressor, Lasso, LogisticRegression, PassiveAggressiveRegressor, Perceptron, SGDClassifier, SGDRegressor

from ..base import OfficialPoolSpec
from .common import make_spec


def build_linear_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec(
            "logistic_l1_liblinear",
            "linear",
            "sklearn",
            lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s, penalty="l1", solver="liblinear", C=0.5),
            lambda s: Lasso(alpha=0.01, max_iter=5000),
            "log_loss_l1",
            "lasso_l1",
        ),
        make_spec(
            "sgd_log_loss",
            "linear",
            "sklearn",
            lambda s: SGDClassifier(loss="log_loss", alpha=1e-3, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s),
            lambda s: SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3, max_iter=2000, tol=1e-4, random_state=s),
            "log_loss",
            "squared_error",
        ),
        make_spec(
            "sgd_modified_huber",
            "linear",
            "sklearn",
            lambda s: SGDClassifier(loss="modified_huber", alpha=1e-3, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s),
            lambda s: HuberRegressor(alpha=1e-3, max_iter=200),
            "modified_huber",
            "huber",
        ),
        make_spec(
            "perceptron",
            "linear",
            "sklearn",
            lambda s: Perceptron(class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s),
            lambda s: PassiveAggressiveRegressor(max_iter=2000, tol=1e-4, random_state=s),
            "perceptron",
            "pa",
        ),
    ]
