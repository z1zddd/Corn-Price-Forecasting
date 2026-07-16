from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestPriceRegressor:
    """Price-regression adapter backed by scikit-learn's BSD-3-Clause implementation."""

    source = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.model = RandomForestRegressor(**self.params)

    def fit(
        self,
        X_train: np.ndarray,
        y_price_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "RandomForestPriceRegressor":
        del validation_data
        self.model.fit(X_train, np.asarray(y_price_train, dtype=float))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = np.asarray(self.model.predict(X_test), dtype=float).reshape(-1)
        if not np.isfinite(predictions).all():
            raise ValueError("Random Forest produced non-finite price predictions")
        return predictions

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "model": self.model}, destination)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestPriceRegressor":
        payload = joblib.load(Path(path))
        instance = cls(**payload["params"])
        instance.model = payload["model"]
        return instance

