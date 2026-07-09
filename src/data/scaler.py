"""从 run_original_real_lstm_seq30_lr001.py 复制 Standardizer 逻辑。

原始脚本约定 X 形状为 [N, T, V]；本框架外部约定是 [N, V, T]，
所以 DataPipeline 会在调用本类前后做 transpose。
"""

from __future__ import annotations

import numpy as np


class Standardizer:
    def __init__(self) -> None:
        self.x_median: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "Standardizer":
        self.x_median = np.nanmedian(x, axis=(0, 1)).astype(np.float32)
        self.x_median = np.where(np.isfinite(self.x_median), self.x_median, 0.0)
        filled = self.fill(x)
        self.x_mean = filled.mean(axis=(0, 1)).astype(np.float32)
        self.x_std = filled.std(axis=(0, 1)).astype(np.float32)
        self.x_std = np.where(self.x_std < 1e-6, 1.0, self.x_std)
        if y is not None:
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y))
            if self.y_std < 1e-6:
                self.y_std = 1.0
        return self

    def fill(self, x: np.ndarray) -> np.ndarray:
        assert self.x_median is not None
        return np.where(np.isfinite(x), x, self.x_median.reshape(1, 1, -1)).astype(np.float32)

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        assert self.x_mean is not None and self.x_std is not None
        return ((self.fill(x) - self.x_mean.reshape(1, 1, -1)) / self.x_std.reshape(1, 1, -1)).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.y_std + self.y_mean

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.transform_x(x)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.inverse_y(y_scaled)

    def to_dict(self) -> dict:
        return {
            "x_median": None if self.x_median is None else self.x_median.tolist(),
            "x_mean": None if self.x_mean is None else self.x_mean.tolist(),
            "x_std": None if self.x_std is None else self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }
