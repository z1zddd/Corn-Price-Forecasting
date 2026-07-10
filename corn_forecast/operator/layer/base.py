"""Common layer protocol and array helpers."""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np


SelfLayer = TypeVar("SelfLayer", bound="BaseLayer")


class BaseLayer(Protocol):
    """Protocol for reusable train-only transformation layers."""

    is_fitted_: bool

    def fit(self: SelfLayer, x, y=None) -> SelfLayer:
        """Fit train-only layer state."""

    def transform(self, x) -> np.ndarray:
        """Transform windows with fitted state."""

    def fit_transform(self: SelfLayer, x, y=None) -> np.ndarray:
        """Fit layer state, then transform the same windows."""


class LayerMixin:
    """Small mixin implementing fit_transform for concrete layers."""

    is_fitted_: bool = False

    def fit_transform(self: SelfLayer, x, y=None) -> np.ndarray:
        self.fit(x, y=y)
        return self.transform(x)


def ensure_3d_windows(x, *, name: str = "x") -> np.ndarray:
    """Return finite float windows shaped [samples, nodes, lookback]."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected {name} with shape [n_samples, n_nodes, lookback], got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """Flatten [samples, nodes, lookback] windows to tabular rows."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got {arr.shape}")
    return arr.reshape(arr.shape[0], -1).astype(float)
