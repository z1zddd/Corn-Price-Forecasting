"""Recurrence-plot features backed by pyts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows


def recurrence_plot_features(x: np.ndarray, *, mode: str = "stats", percentage: float = 20.0) -> np.ndarray:
    """Per-window recurrence plot features using pyts' official transformer."""

    try:
        from pyts.image import RecurrencePlot
    except ImportError as exc:
        raise ImportError("pyts is required for recurrence plot layers. Install with: pip install pyts") from exc
    windows = ensure_3d_windows(x)
    n_cases, n_nodes, n_time = windows.shape
    transformer = RecurrencePlot(threshold="point", percentage=percentage)
    outputs: list[np.ndarray] = []
    for node_idx in range(n_nodes):
        series = np.nan_to_num(windows[:, node_idx, :], nan=0.0, posinf=0.0, neginf=0.0)
        plots = np.asarray(transformer.fit_transform(series), dtype=float)
        if mode == "flat":
            outputs.append(plots.reshape(n_cases, -1))
            continue
        if mode != "stats":
            raise ValueError(f"Unknown recurrence mode: {mode}")
        density = plots.mean(axis=(1, 2))
        diag = np.asarray([np.mean(np.diag(plot)) for plot in plots], dtype=float)
        last_row = plots[:, -1, :].mean(axis=1)
        upper = np.asarray([plot[np.triu_indices(n_time, k=1)].mean() if n_time > 1 else 0.0 for plot in plots], dtype=float)
        row_std = plots.mean(axis=2).std(axis=1)
        outputs.append(np.column_stack([density, diag, last_row, upper, row_std]))
    return np.nan_to_num(np.concatenate(outputs, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RecurrencePlotLayer(LayerMixin):
    """Statistical or flat recurrence-plot representation for each node."""

    mode: str = "stats"
    percentage: float = 20.0
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        ensure_3d_windows(x)
        if self.mode not in {"stats", "flat"}:
            raise ValueError(f"Unknown recurrence mode: {self.mode}")
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("RecurrencePlotLayer is not fitted")
        return recurrence_plot_features(x, mode=self.mode, percentage=self.percentage)
