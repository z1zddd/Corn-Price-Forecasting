"""Train-only scaling for sequence windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceStandardizer:
    """Standardize [N, V, T] windows using train-only statistics."""

    x_median: np.ndarray | None = None
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0

    def fit(self, x, y=None) -> "SequenceStandardizer":
        """Fit per-feature statistics from training windows only."""

        arr = self._as_windows(x)
        self.x_median = np.nanmedian(arr, axis=(0, 2)).astype(float)
        self.x_median = np.where(np.isfinite(self.x_median), self.x_median, 0.0)
        filled = self.fill_x(arr)
        self.x_mean = filled.mean(axis=(0, 2)).astype(float)
        self.x_std = filled.std(axis=(0, 2)).astype(float)
        self.x_std = np.where(self.x_std < 1e-12, 1.0, self.x_std)
        if y is not None:
            y_arr = np.asarray(y, dtype=float)
            self.y_mean = float(np.mean(y_arr))
            self.y_std = float(np.std(y_arr))
            if self.y_std < 1e-12:
                self.y_std = 1.0
        return self

    def fill_x(self, x) -> np.ndarray:
        """Fill non-finite feature values with train medians."""

        if self.x_median is None:
            raise RuntimeError("SequenceStandardizer must be fitted before fill_x")
        arr = self._as_windows(x)
        return np.where(np.isfinite(arr), arr, self.x_median.reshape(1, -1, 1)).astype(float)

    def transform_x(self, x) -> np.ndarray:
        """Transform [N, V, T] windows."""

        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("SequenceStandardizer must be fitted before transform_x")
        filled = self.fill_x(x)
        return ((filled - self.x_mean.reshape(1, -1, 1)) / self.x_std.reshape(1, -1, 1)).astype(float)

    def transform_y(self, y) -> np.ndarray:
        """Transform a numeric regression target."""

        return ((np.asarray(y, dtype=float) - self.y_mean) / self.y_std).astype(float)

    def inverse_y(self, y_scaled) -> np.ndarray:
        """Invert a transformed numeric regression target."""

        return np.asarray(y_scaled, dtype=float) * self.y_std + self.y_mean

    def to_dict(self) -> dict:
        """Return JSON-serializable scaler parameters."""

        return {
            "x_median": None if self.x_median is None else self.x_median.tolist(),
            "x_mean": None if self.x_mean is None else self.x_mean.tolist(),
            "x_std": None if self.x_std is None else self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @staticmethod
    def _as_windows(x) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"Expected windows with shape [N, V, T], got {arr.shape}")
        return arr
