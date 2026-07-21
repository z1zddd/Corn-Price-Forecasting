from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class SklearnPriceRegressor:
    """Common finite-price and persistence contract for tabular regressors."""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.model = self._make_model(dict(params))

    def _make_model(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def fit(
        self,
        X_train: np.ndarray,
        y_price_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "SklearnPriceRegressor":
        del validation_data
        self.model.fit(np.asarray(X_train), np.asarray(y_price_train, dtype=float))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = np.asarray(self.model.predict(np.asarray(X_test)), dtype=float).reshape(-1)
        if not np.isfinite(predictions).all():
            raise ValueError(f"{type(self).__name__} produced non-finite price predictions")
        return predictions

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "model": self.model}, destination)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnPriceRegressor":
        payload = joblib.load(Path(path))
        instance = cls(**payload["params"])
        instance.model = payload["model"]
        return instance

