"""scikit-learn model adapters."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """Flatten [N, V, T] into [N, V*T]."""

    return x.reshape(x.shape[0], -1)


class SklearnClassifierAdapter:
    """Adapter around sklearn binary classifiers."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        self.estimator.fit(flatten_windows(x_train), y_train)
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return self.estimator.predict(flatten_windows(x_test)).astype(int)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        if not hasattr(self.estimator, "predict_proba"):
            return None
        proba = np.asarray(self.estimator.predict_proba(flatten_windows(x_test)), dtype=float)
        classes = getattr(self.estimator, "classes_", None)
        if classes is not None and len(classes) == 1:
            return np.ones(proba.shape[0], dtype=float) if int(classes[0]) == 1 else np.zeros(proba.shape[0], dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0]
        if classes is not None and 1 in classes:
            class_one_idx = list(classes).index(1)
            return proba[:, class_one_idx]
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.estimator, path)


def create_random_forest(params: dict) -> SklearnClassifierAdapter:
    """Create a random forest classifier adapter."""

    estimator = RandomForestClassifier(**params)
    return SklearnClassifierAdapter(estimator)


def create_logistic_regression(params: dict) -> SklearnClassifierAdapter:
    """Create a logistic regression classifier adapter."""

    defaults = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
    defaults.update(params)
    estimator = LogisticRegression(**defaults)
    return SklearnClassifierAdapter(estimator)


def create_lightgbm(params: dict) -> SklearnClassifierAdapter:
    """Create a LightGBM classifier adapter when the optional dependency exists."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("LightGBM is not installed. Install with: pip install -e .[trees]") from exc
    defaults = {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "random_state": 42, "n_jobs": -1}
    defaults.update(params)
    estimator = LGBMClassifier(**defaults)
    return SklearnClassifierAdapter(estimator)


def create_xgboost(params: dict) -> SklearnClassifierAdapter:
    """Create an XGBoost classifier adapter when the optional dependency exists."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost is not installed. Install with: pip install -e .[trees]") from exc
    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    defaults.update(params)
    estimator = XGBClassifier(**defaults)
    return SklearnClassifierAdapter(estimator)


def create_catboost(params: dict) -> SklearnClassifierAdapter:
    """Create a CatBoost classifier adapter when the optional dependency exists."""

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("CatBoost is not installed. Install with: pip install -e .[trees]") from exc
    defaults = {"iterations": 300, "learning_rate": 0.03, "depth": 4, "random_seed": 42, "verbose": False}
    defaults.update(params)
    estimator = CatBoostClassifier(**defaults)
    return SklearnClassifierAdapter(estimator)