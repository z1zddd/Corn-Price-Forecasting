"""sklearn-style wrappers following darts RegressionModel's flattened-window pattern."""

from __future__ import annotations

import numpy as np

from src.models.base import BaseModel
from src.eval.metrics import logit_np


class SklearnWindowModel(BaseModel):
    def __init__(self, model, task: str = "regression"):
        self.model = model
        self.task = task

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        y = np.asarray(y_train).reshape(-1)
        if self.task == "classification":
            y = y.astype(int)
        self.model.fit(self._flatten(X_train), y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.task == "classification":
            return self.predict_proba(X)
        return np.asarray(self.model.predict(self._flatten(X)), dtype="float32").reshape(-1)

    def predict_proba(self, X) -> np.ndarray:
        x = self._flatten(X)
        if self.task != "classification":
            return np.asarray(self.model.predict(x), dtype="float32").reshape(-1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1].astype("float32")
            return proba.reshape(-1).astype("float32")
        if hasattr(self.model, "decision_function"):
            decision = np.asarray(self.model.decision_function(x), dtype=float).reshape(-1)
            return (1.0 / (1.0 + np.exp(-np.clip(decision, -50.0, 50.0)))).astype("float32")
        return np.asarray(self.model.predict(x), dtype="float32").reshape(-1)

    def predict_logits(self, X) -> np.ndarray:
        x = self._flatten(X)
        if self.task == "classification" and hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(x), dtype="float32").reshape(-1)
        return logit_np(self.predict_proba(X)).astype("float32").reshape(-1)

    @staticmethod
    def _flatten(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        return arr.reshape(arr.shape[0], -1)

    def get_params(self) -> dict:
        return {"task": self.task, "model_params": getattr(self.model, "get_params", lambda: {})()}
