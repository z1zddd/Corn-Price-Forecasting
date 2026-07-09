"""Simple forecasting baselines required before complex models."""

from __future__ import annotations

import numpy as np

from src.models.base import BaseModel


class ZeroReturnBaseline(BaseModel):
    """Predict zero future return in the scaled target space."""

    def __init__(self, scaled_value: float = 0.0, **_):
        self.value_ = float(scaled_value)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.value_, dtype="float32")


class MeanReturnBaseline(BaseModel):
    """Predict train mean target."""

    def __init__(self, **_):
        self.value_ = 0.0

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.value_ = float(np.mean(y_train))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.value_, dtype="float32")


class LastReturnBaseline(BaseModel):
    """Predict the previous observed close-to-close return from the input window."""

    def __init__(
        self,
        close_feature_idx: int = 0,
        close_mean: float = 0.0,
        close_std: float = 1.0,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        **_,
    ):
        self.close_feature_idx = close_feature_idx
        self.close_mean = close_mean
        self.close_std = close_std
        self.y_mean = y_mean
        self.y_std = y_std

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        close = self._inverse_close(arr[:, self.close_feature_idx, :])
        prev = close[:, -2]
        last = close[:, -1]
        denom = np.where(np.abs(prev) < 1e-8, 1.0, prev)
        raw_return = (last / denom) - 1.0
        return self._scale_return(raw_return)

    def _inverse_close(self, close_scaled: np.ndarray) -> np.ndarray:
        return close_scaled * self.close_std + self.close_mean

    def _scale_return(self, raw_return: np.ndarray) -> np.ndarray:
        return ((raw_return - self.y_mean) / self.y_std).astype("float32")


class MovingAverageReturnBaseline(LastReturnBaseline):
    """Predict the trailing mean close-to-close return."""

    def __init__(self, window: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def predict(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        close = self._inverse_close(arr[:, self.close_feature_idx, :])
        prev = close[:, :-1]
        nxt = close[:, 1:]
        returns = nxt / np.where(np.abs(prev) < 1e-8, 1.0, prev) - 1.0
        raw_return = returns[:, -self.window :].mean(axis=1)
        return self._scale_return(raw_return)
