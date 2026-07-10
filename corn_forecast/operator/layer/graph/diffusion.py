"""Graph diffusion transforms for feature-node windows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows, flatten_windows
from corn_forecast.operator.layer.graph.adjacency import AdjacencyLayer


DIFFUSION_TRANSFORMS = {"smooth_only", "raw_smooth", "raw_residual", "raw_diffuse2"}


def smooth_only(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    smooth = _smooth(x, adjacency)
    return flatten_windows(smooth) if flatten else smooth


def raw_smooth(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    windows = ensure_3d_windows(x)
    smooth = _smooth(windows, adjacency)
    out = np.concatenate([windows, smooth], axis=1)
    return flatten_windows(out) if flatten else out


def raw_residual(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    windows = ensure_3d_windows(x)
    smooth = _smooth(windows, adjacency)
    out = np.concatenate([windows, windows - smooth], axis=1)
    return flatten_windows(out) if flatten else out


def raw_diffuse2(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    windows = ensure_3d_windows(x)
    smooth = _smooth(windows, adjacency)
    diffuse2 = _smooth(smooth, adjacency)
    out = np.concatenate([windows, smooth, diffuse2], axis=1)
    return flatten_windows(out) if flatten else out


def apply_diffusion(x: np.ndarray, adjacency: np.ndarray, *, method: str, flatten: bool = True) -> np.ndarray:
    """Apply a named diffusion transform."""

    if method == "smooth_only":
        return smooth_only(x, adjacency, flatten=flatten)
    if method == "raw_smooth":
        return raw_smooth(x, adjacency, flatten=flatten)
    if method == "raw_residual":
        return raw_residual(x, adjacency, flatten=flatten)
    if method == "raw_diffuse2":
        return raw_diffuse2(x, adjacency, flatten=flatten)
    raise ValueError(f"Unknown diffusion transform: {method}")


@dataclass
class DiffusionLayer(LayerMixin):
    """Fit an adjacency matrix, then diffuse windows through it."""

    method: str = "raw_smooth"
    graph_builder: str = "corr_abs_top5"
    feature_cols: list[str] | None = None
    flatten: bool = True
    adjacency_layer_: AdjacencyLayer = field(init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        if self.method not in DIFFUSION_TRANSFORMS:
            raise ValueError(f"Unknown diffusion transform: {self.method}")
        self.adjacency_layer_ = AdjacencyLayer(builder=self.graph_builder, feature_cols=self.feature_cols)
        self.adjacency_layer_.fit(x, y=y)
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("DiffusionLayer is not fitted")
        return apply_diffusion(x, self.adjacency_layer_.adjacency_, method=self.method, flatten=self.flatten)


def _smooth(x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    windows = ensure_3d_windows(x)
    adj = np.asarray(adjacency, dtype=float)
    if adj.shape != (windows.shape[1], windows.shape[1]):
        raise ValueError(f"Adjacency shape {adj.shape} does not match node count {windows.shape[1]}")
    return np.einsum("ij,bjt->bit", adj, windows).astype(float)
