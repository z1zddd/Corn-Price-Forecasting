"""Baseline models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class LastReturnBaseline:
    """Predict up when the latest value in the first feature increased."""

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        first_feature = x_test[:, 0, :]
        return (first_feature[:, -1] > first_feature[:, 0]).astype(int)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        return self.predict(x_test).astype(float)

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": "LastReturnBaseline"}, path)


class MeanDirectionBaseline:
    """Predict the majority training class."""

    def __init__(self) -> None:
        self.prediction = 0

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        self.prediction = int(np.mean(y_train) >= 0.5)
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return np.full(len(x_test), self.prediction, dtype=int)

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        return np.full(len(x_test), float(self.prediction), dtype=float)

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": "MeanDirectionBaseline", "prediction": self.prediction}, path)