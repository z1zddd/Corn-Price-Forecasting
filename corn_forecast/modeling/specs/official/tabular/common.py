"""Shared helpers for tabular official-pool specs."""

from __future__ import annotations

import inspect

from ..base import Factory, OfficialPoolSpec


def make_spec(name, family, package, clf, reg, clf_loss="native", reg_loss="squared_error"):
    return OfficialPoolSpec(name, family, package, clf, reg, clf_loss, reg_loss, "tabular_flat", "official_tabular_package")


def adaboost_tree_classifier_factory(ada_cls, tree_cls) -> Factory:
    def factory(seed: int):
        params = {
            "estimator": tree_cls(max_depth=1, class_weight="balanced", random_state=seed),
            "n_estimators": 80,
            "learning_rate": 0.05,
            "random_state": seed,
        }
        if "algorithm" in inspect.signature(ada_cls).parameters:
            params["algorithm"] = "SAMME"
        return ada_cls(**params)

    return factory
