"""Benchmark layer-2 loss variant model adapters."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from corn_forecast.modeling.classical.sklearn import flatten_windows


class RegressionSignModel:
    """Fit a return regressor and trade on the predicted return sign."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self.estimator.fit(flatten_windows(x_train), np.asarray(y_train, dtype=float))
        return self

    def fit_with_targets(self, x_train, y_class_train, y_return_train, x_val=None, y_class_val=None, y_return_val=None):
        return self.fit(x_train, y_return_train, x_val, y_return_val)

    def predict_regression(self, x_test) -> np.ndarray:
        return np.asarray(self.estimator.predict(flatten_windows(x_test)), dtype=float).reshape(-1)

    def predict(self, x_test) -> np.ndarray:
        return (self.predict_regression(x_test) > 0.0).astype(int)

    def predict_proba(self, x_test) -> np.ndarray:
        raw = self.predict_regression(x_test)
        scale = np.std(raw) if len(raw) > 1 else 1.0
        scale = scale if scale > 1e-12 else 1.0
        return 1.0 / (1.0 + np.exp(-raw / scale))

    def save(self, path: str | Path) -> None:
        joblib.dump(self.estimator, path)


class DualHeadMseBceModel:
    """Combine a direction classifier with a return regressor."""

    def __init__(self, classifier, regressor, dual_weight: float = 0.5) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.dual_weight = float(dual_weight)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self.classifier.fit(flatten_windows(x_train), y_train)
        self.regressor.fit(flatten_windows(x_train), np.asarray(y_train, dtype=float))
        return self

    def fit_with_targets(self, x_train, y_class_train, y_return_train, x_val=None, y_class_val=None, y_return_val=None):
        self.classifier.fit(flatten_windows(x_train), y_class_train)
        self.regressor.fit(flatten_windows(x_train), np.asarray(y_return_train, dtype=float))
        return self

    def predict_regression(self, x_test) -> np.ndarray:
        return np.asarray(self.regressor.predict(flatten_windows(x_test)), dtype=float).reshape(-1)

    def predict_proba(self, x_test) -> np.ndarray:
        flat = flatten_windows(x_test)
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(flat)
            classes = list(getattr(self.classifier, "classes_", [0, 1]))
            cls_prob = proba[:, classes.index(1)] if 1 in classes and proba.ndim == 2 else proba.reshape(-1)
        else:
            cls_prob = self.classifier.predict(flat).astype(float)
        reg_raw = self.predict_regression(x_test)
        reg_prob = (reg_raw > 0.0).astype(float)
        return (1.0 - self.dual_weight) * cls_prob + self.dual_weight * reg_prob

    def predict(self, x_test) -> np.ndarray:
        return (self.predict_proba(x_test) > 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump({"classifier": self.classifier, "regressor": self.regressor, "dual_weight": self.dual_weight}, path)


class FocalLogisticModel:
    """Small torch-backed focal-loss logistic classifier."""

    def __init__(self, *, gamma: float = 2.0, epochs: int = 50, lr: float = 0.01, random_state: int = 42) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("torch is required for focal_logistic. Install with: pip install -e .[deep]") from exc
        self.torch = torch
        self.nn = nn
        self.gamma = float(gamma)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self.model = None

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        torch = self.torch
        torch.manual_seed(self.random_state)
        x = torch.as_tensor(flatten_windows(x_train), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y_train, dtype=float).reshape(-1, 1), dtype=torch.float32)
        self.model = self.nn.Linear(x.shape[1], 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            bce = self.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            p_t = torch.exp(-bce)
            loss = ((1.0 - p_t) ** self.gamma * bce).mean()
            loss.backward()
            optimizer.step()
        return self

    def predict_proba(self, x_test) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        torch = self.torch
        x = torch.as_tensor(flatten_windows(x_test), dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(x)).cpu().numpy().reshape(-1)

    def predict(self, x_test) -> np.ndarray:
        return (self.predict_proba(x_test) > 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        self.torch.save({"state_dict": self.model.state_dict(), "gamma": self.gamma}, path)


def create_regression_mse_sign(params: dict) -> RegressionSignModel:
    defaults = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    defaults.update(params)
    return RegressionSignModel(RandomForestRegressor(criterion="squared_error", **defaults))


def create_regression_mae_sign(params: dict) -> RegressionSignModel:
    defaults = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    defaults.update(params)
    return RegressionSignModel(RandomForestRegressor(criterion="absolute_error", **defaults))


def create_regression_huber_sign(params: dict) -> RegressionSignModel:
    defaults = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05, "random_state": 42, "loss": "huber"}
    defaults.update(params)
    return RegressionSignModel(GradientBoostingRegressor(**defaults))


def create_dual_head_mse_bce(params: dict) -> DualHeadMseBceModel:
    random_state = int(params.get("random_state", 42))
    dual_weight = float(params.get("dual_weight", 0.5))
    classifier = RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth", 5),
        random_state=random_state,
    )
    regressor = RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=params.get("max_depth", 5),
        random_state=random_state,
    )
    return DualHeadMseBceModel(classifier, regressor, dual_weight=dual_weight)


def create_focal_logistic(params: dict) -> FocalLogisticModel:
    return FocalLogisticModel(**params)
