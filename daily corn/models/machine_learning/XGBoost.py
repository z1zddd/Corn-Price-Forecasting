from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost
from xgboost import XGBRegressor


class XGBoostPriceRegressor:
    """Price-regression adapter for the Apache-2.0 XGBoost project."""

    source = "https://github.com/dmlc/xgboost/releases/tag/v3.2.0"
    source_version = "3.2.0"
    runtime_version = xgboost.__version__
    license = "Apache-2.0"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.model: XGBRegressor | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_price_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "XGBoostPriceRegressor":
        params = dict(self.params)
        if validation_data is None:
            params.pop("early_stopping_rounds", None)
        self.model = XGBRegressor(**params)
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if validation_data is not None:
            X_validation, y_validation = validation_data
            fit_kwargs["eval_set"] = [
                (np.asarray(X_validation), np.asarray(y_validation, dtype=float))
            ]
        self.model.fit(
            np.asarray(X_train),
            np.asarray(y_price_train, dtype=float),
            **fit_kwargs,
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostPriceRegressor must be fitted before predict")
        predictions = np.asarray(self.model.predict(np.asarray(X_test)), dtype=float).reshape(-1)
        if not np.isfinite(predictions).all():
            raise ValueError("XGBoost produced non-finite price predictions")
        return predictions

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "model": self.model}, destination)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostPriceRegressor":
        payload = joblib.load(Path(path))
        instance = cls(**payload["params"])
        instance.model = payload["model"]
        return instance
