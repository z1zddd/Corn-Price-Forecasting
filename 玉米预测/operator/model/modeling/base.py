"""Unified model interface inspired by darts ForecastingModel and sklearn estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pickle

import numpy as np


class BaseModel(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "BaseModel":
        """Train. X is [N, V, T], y is [N] or [N, 1]."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predictions with shape [N]."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def get_params(self) -> dict:
        return {}
