"""Model interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class BaseModel(Protocol):
    """Protocol all model adapters follow."""

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        """Fit the model."""

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Return class labels or numeric predictions."""

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        """Return class-1 probabilities when available."""

    def save(self, path: str | Path) -> None:
        """Persist the model."""