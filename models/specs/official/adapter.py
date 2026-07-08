"""Training/prediction adapter for official-pool estimators."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .base import OfficialPoolSpec
from .io import format_input, positive_probability, sigmoid


class ConstantClassifier:
    """Fallback classifier for one-class rolling folds."""

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def fit(self, x, y):
        return self

    def predict_proba(self, x) -> np.ndarray:
        n = len(x)
        p = np.full(n, self.positive_probability, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class ConstantRegressor:
    """Fallback regressor for constant return folds."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, x, y):
        return self

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


class OfficialPoolAdapter:
    """Framework adapter around official sklearn, aeon, and Keras estimators."""

    def __init__(self, spec: OfficialPoolSpec, *, seed: int = 42) -> None:
        self.spec = spec
        self.seed = int(seed)
        self.classifier_ = None
        self.regressor_ = None
        self.model_family = spec.family
        self.package = spec.package
        self.input_kind = spec.input_kind
        self.source_kind = spec.source_kind

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        return self.fit_with_targets(x_train, y_train, np.asarray(y_train, dtype=float), x_val, y_val, None)

    def fit_with_targets(self, x_train, y_class_train, y_return_train, x_val=None, y_class_val=None, y_return_val=None):
        x = format_input(x_train, self.spec.input_kind)
        y_class = np.asarray(y_class_train, dtype=int).reshape(-1)
        y_return = np.asarray(y_return_train, dtype=float).reshape(-1)
        if self.spec.classifier_factory is not None:
            self.classifier_ = fit_classifier(self.spec, x, y_class, self.seed)
        if self.spec.regressor_factory is not None:
            self.regressor_ = fit_regressor(self.spec, x, y_return, self.seed)
        return self

    def predict_proba(self, x_test) -> np.ndarray:
        x = format_input(x_test, self.spec.input_kind)
        if self.classifier_ is not None:
            return positive_probability(self.classifier_, x)
        raw = self.predict_regression(x_test)
        if raw is None:
            raise RuntimeError(f"{self.spec.name} has neither fitted classifier nor fitted regressor")
        scale = float(np.nanstd(raw))
        scale = scale if scale > 1e-12 else 1.0
        return sigmoid(raw / scale)

    def predict_regression(self, x_test) -> np.ndarray | None:
        if self.regressor_ is None:
            return None
        x = format_input(x_test, self.spec.input_kind)
        return np.asarray(self.regressor_.predict(x), dtype=float).reshape(-1)

    def predict(self, x_test) -> np.ndarray:
        return (self.predict_proba(x_test) > 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "spec": self.spec,
                "seed": self.seed,
                "classifier": self.classifier_,
                "regressor": self.regressor_,
            },
            path,
        )


def create_official_pool_model(model_name: str, params: dict | None = None) -> OfficialPoolAdapter:
    """Create a framework adapter for one model in the official 57-model pool."""

    from .pool import build_official_model_pool

    params = dict(params or {})
    pool_model = str(params.pop("pool_model", model_name))
    seed = int(params.pop("seed", params.pop("random_state", 42)))
    if params:
        raise ValueError(f"Unsupported official_pool params for {pool_model}: {sorted(params)}")
    specs = {spec.name: spec for spec in build_official_model_pool()}
    if pool_model not in specs:
        raise ValueError(f"Unknown official_pool model: {pool_model}")
    return OfficialPoolAdapter(specs[pool_model], seed=seed)


def fit_classifier(spec: OfficialPoolSpec, x: np.ndarray, y: np.ndarray, seed: int):
    if len(np.unique(y)) < 2:
        return ConstantClassifier(float(np.mean(y)))
    if spec.classifier_factory is None:
        raise RuntimeError(f"{spec.name} does not define a classifier")
    model = spec.classifier_factory(seed)
    model.fit(x, y.astype(int))
    return model


def fit_regressor(spec: OfficialPoolSpec, x: np.ndarray, y: np.ndarray, seed: int):
    if np.nanstd(y) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y)))
    if spec.regressor_factory is None:
        return None
    model = spec.regressor_factory(seed)
    model.fit(x, y.astype(float))
    return model
