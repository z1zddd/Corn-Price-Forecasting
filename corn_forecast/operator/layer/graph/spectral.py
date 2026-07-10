"""Graph spectral filters for feature-node windows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows, flatten_windows
from corn_forecast.operator.layer.graph.adjacency import AdjacencyLayer


SPECTRAL_TRANSFORMS = {"spectral_lowpass", "spectral_residual", "spectral_bands"}


def graph_spectral_bands(x: np.ndarray, adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split node signals into graph-Laplacian low/mid/high bands."""

    windows = ensure_3d_windows(x)
    n_nodes = windows.shape[1]
    if n_nodes <= 1:
        zeros = np.zeros_like(windows)
        return windows.copy(), zeros, zeros
    sym_adj = np.maximum(np.asarray(adjacency, dtype=float), np.asarray(adjacency, dtype=float).T)
    laplacian = np.eye(n_nodes, dtype=float) - sym_adj
    try:
        eigvals, basis = np.linalg.eigh(laplacian)
        order = np.argsort(eigvals)
        basis = basis[:, order]
    except np.linalg.LinAlgError:
        zeros = np.zeros_like(windows)
        return windows.copy(), zeros, zeros
    coeff = np.einsum("vi,bvt->bit", basis, windows)
    low_end = max(1, int(np.ceil(n_nodes * 0.25)))
    mid_end = max(low_end + 1, int(np.ceil(n_nodes * 0.60)))
    mid_end = min(mid_end, n_nodes)

    def reconstruct(start: int, end: int) -> np.ndarray:
        if start >= end:
            return np.zeros_like(windows)
        return np.einsum("vi,bit->bvt", basis[:, start:end], coeff[:, start:end, :])

    return tuple(
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        for arr in (reconstruct(0, low_end), reconstruct(low_end, mid_end), reconstruct(mid_end, n_nodes))
    )


def spectral_lowpass(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    low, _mid, _high = graph_spectral_bands(x, adjacency)
    return flatten_windows(low) if flatten else low


def spectral_residual(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    windows = ensure_3d_windows(x)
    low, _mid, _high = graph_spectral_bands(windows, adjacency)
    out = np.concatenate([windows, windows - low], axis=1)
    return flatten_windows(out) if flatten else out


def spectral_bands(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    low, mid, high = graph_spectral_bands(x, adjacency)
    out = np.concatenate([low, mid, high], axis=1)
    return flatten_windows(out) if flatten else out


def apply_spectral(x: np.ndarray, adjacency: np.ndarray, *, method: str, flatten: bool = True) -> np.ndarray:
    if method == "spectral_lowpass":
        return spectral_lowpass(x, adjacency, flatten=flatten)
    if method == "spectral_residual":
        return spectral_residual(x, adjacency, flatten=flatten)
    if method == "spectral_bands":
        return spectral_bands(x, adjacency, flatten=flatten)
    raise ValueError(f"Unknown spectral transform: {method}")


@dataclass
class SpectralLayer(LayerMixin):
    """Fit an adjacency matrix, then return spectral graph features."""

    method: str = "spectral_bands"
    graph_builder: str = "corr_abs_top5"
    feature_cols: list[str] | None = None
    flatten: bool = True
    adjacency_layer_: AdjacencyLayer = field(init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        if self.method not in SPECTRAL_TRANSFORMS:
            raise ValueError(f"Unknown spectral transform: {self.method}")
        self.adjacency_layer_ = AdjacencyLayer(builder=self.graph_builder, feature_cols=self.feature_cols)
        self.adjacency_layer_.fit(x, y=y)
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("SpectralLayer is not fitted")
        return apply_spectral(x, self.adjacency_layer_.adjacency_, method=self.method, flatten=self.flatten)
