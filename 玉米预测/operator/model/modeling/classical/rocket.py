"""sktime ROCKET-family classifiers.

These are external convolution-kernel time-series classifiers. The wrapper keeps
the project contract while delegating the transform/classifier to sktime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.eval.metrics import logit_np


class SktimeRocketClassifier(BaseModel):
    def __init__(
        self,
        task: str = "classification",
        rocket_transform: str = "rocket",
        num_kernels: int = 2000,
        n_jobs: int = 1,
        random_state: int | None = 42,
        **params,
    ):
        self.task = task
        self.rocket_transform = rocket_transform
        self.num_kernels = num_kernels
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        try:
            from sktime.classification.kernel_based import RocketClassifier
        except ImportError as exc:
            raise ImportError("SktimeRocketClassifier requires `pip install sktime==0.37.0`.") from exc
        self.model = RocketClassifier(
            num_kernels=self.num_kernels,
            rocket_transform=self.rocket_transform,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.params,
        )
        self.model.fit(to_nested_panel(X_train), np.asarray(y_train).reshape(-1).astype(int))
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(to_nested_panel(X))
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1].astype("float32")
            return proba.reshape(-1).astype("float32")
        pred = self.model.predict(to_nested_panel(X))
        return np.asarray(pred, dtype="float32").reshape(-1)

    def predict_logits(self, X) -> np.ndarray:
        return logit_np(self.predict_proba(X)).astype("float32").reshape(-1)

    def get_params(self) -> dict:
        return {
            "task": self.task,
            "rocket_transform": self.rocket_transform,
            "num_kernels": self.num_kernels,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            **self.params,
        }


def to_nested_panel(X) -> pd.DataFrame:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
    rows = []
    for sample in arr:
        rows.append([pd.Series(sample[var_idx, :]) for var_idx in range(sample.shape[0])])
    return pd.DataFrame(rows, columns=[f"var_{idx}" for idx in range(arr.shape[1])])
