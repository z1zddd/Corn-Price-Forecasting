"""Train-only feature graph builders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.neighbors import NearestNeighbors

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows


GRAPH_BUILDERS = {
    "corr_abs_top3",
    "corr_abs_top5",
    "corr_abs_top8",
    "spearman_abs_top5",
    "knn_cosine_top5",
    "knn_euclidean_top5",
    "glasso_alpha05",
    "glasso_alpha10",
    "group_prefix",
}


def corr_abs_topk(x_train_flat: np.ndarray, *, top_k: int) -> np.ndarray:
    """Absolute Pearson-correlation graph, symmetrized by row top-k."""

    return _corr_graph(x_train_flat, top_k=top_k, rank=False)


def spearman_abs_topk(x_train_flat: np.ndarray, *, top_k: int) -> np.ndarray:
    """Absolute Spearman-correlation graph, symmetrized by row top-k."""

    return _corr_graph(x_train_flat, top_k=top_k, rank=True)


def knn_cosine_topk(x_train_flat: np.ndarray, *, top_k: int) -> np.ndarray:
    """Feature-neighbor graph using cosine distance."""

    return _knn_feature_graph(x_train_flat, top_k=top_k, metric="cosine")


def knn_euclidean_topk(x_train_flat: np.ndarray, *, top_k: int) -> np.ndarray:
    """Feature-neighbor graph using Euclidean distance."""

    return _knn_feature_graph(x_train_flat, top_k=top_k, metric="euclidean")


def graphical_lasso_alpha(x_train_flat: np.ndarray, *, alpha: float, top_k: int = 5) -> np.ndarray:
    """Sparse precision graph from sklearn GraphicalLasso with correlation fallback."""

    try:
        model = GraphicalLasso(alpha=float(alpha), max_iter=100, assume_centered=False)
        model.fit(np.asarray(x_train_flat, dtype=float))
        weights = np.abs(model.precision_)
        np.fill_diagonal(weights, 0.0)
        return topk_symmetric(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), top_k)
    except Exception:
        return corr_abs_topk(x_train_flat, top_k=top_k)


def group_prefix(feature_cols: list[str] | None, n_features: int) -> np.ndarray:
    """Connect features with the same coarse prefix group."""

    names = list(feature_cols) if feature_cols and len(feature_cols) == n_features else [f"feature_{i}" for i in range(n_features)]
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(feature_group(name), []).append(idx)
    weights = np.zeros((n_features, n_features), dtype=float)
    for members in groups.values():
        for i in members:
            for j in members:
                if i != j:
                    weights[i, j] = 1.0
    return weights


def normalized_adjacency(weights: np.ndarray, *, add_self_loops: bool = True) -> np.ndarray:
    """Return symmetric D^-1/2 A D^-1/2 adjacency."""

    adjacency = np.nan_to_num(np.maximum(np.asarray(weights, dtype=float), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    if add_self_loops:
        adjacency = adjacency + np.eye(adjacency.shape[0], dtype=float)
    degree = adjacency.sum(axis=1)
    degree = np.where(degree > 1e-12, degree, 1.0)
    inv_sqrt = 1.0 / np.sqrt(degree)
    return (inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]).astype(float)


def graph_info(weights: np.ndarray) -> dict[str, float | int]:
    """Small graph audit summary without requiring a full model."""

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for graph_info. Install with: pip install networkx") from exc
    plain = np.maximum(np.asarray(weights, dtype=float), np.asarray(weights, dtype=float).T)
    np.fill_diagonal(plain, 0.0)
    graph = nx.from_numpy_array(plain)
    return {
        "graph_edges": int(graph.number_of_edges()),
        "graph_density": float(nx.density(graph)) if plain.shape[0] > 1 else 0.0,
        "graph_components": int(nx.number_connected_components(graph)) if plain.shape[0] else 0,
    }


def build_adjacency(
    x_train_flat: np.ndarray,
    *,
    builder: str,
    feature_cols: list[str] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Build a graph from train-only flattened feature rows."""

    match = re.fullmatch(r"corr_abs_top(\d+)", builder)
    if match:
        weights = corr_abs_topk(x_train_flat, top_k=int(match.group(1)))
    elif builder == "spearman_abs_top5":
        weights = spearman_abs_topk(x_train_flat, top_k=5)
    elif builder == "knn_cosine_top5":
        weights = knn_cosine_topk(x_train_flat, top_k=5)
    elif builder == "knn_euclidean_top5":
        weights = knn_euclidean_topk(x_train_flat, top_k=5)
    elif builder == "glasso_alpha05":
        weights = graphical_lasso_alpha(x_train_flat, alpha=0.05, top_k=5)
    elif builder == "glasso_alpha10":
        weights = graphical_lasso_alpha(x_train_flat, alpha=0.10, top_k=5)
    elif builder == "group_prefix":
        weights = group_prefix(feature_cols, np.asarray(x_train_flat).shape[1])
    else:
        raise ValueError(f"Unknown graph builder: {builder}")
    return normalized_adjacency(weights) if normalize else weights.astype(float)


@dataclass
class AdjacencyLayer(LayerMixin):
    """Fit a train-only adjacency matrix from feature-node windows."""

    builder: str = "corr_abs_top5"
    feature_cols: list[str] | None = None
    normalize: bool = True
    adjacency_: np.ndarray | None = field(default=None, init=False)
    graph_info_: dict[str, float | int] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        windows = ensure_3d_windows(x)
        train_flat = np.transpose(windows, (0, 2, 1)).reshape(-1, windows.shape[1])
        self.adjacency_ = build_adjacency(
            train_flat,
            builder=self.builder,
            feature_cols=self.feature_cols,
            normalize=self.normalize,
        )
        self.graph_info_ = graph_info(self.adjacency_)
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if self.adjacency_ is None:
            raise RuntimeError("AdjacencyLayer is not fitted")
        _ = ensure_3d_windows(x)
        return self.adjacency_.copy()


def _corr_graph(x_train_flat: np.ndarray, *, top_k: int, rank: bool) -> np.ndarray:
    x = np.asarray(x_train_flat, dtype=float)
    if rank:
        x = pd.DataFrame(x).rank(axis=0).to_numpy(dtype=float)
    corr = np.corrcoef(x, rowvar=False)
    weights = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(weights, 0.0)
    return topk_symmetric(weights, top_k)


def _knn_feature_graph(x_train_flat: np.ndarray, *, top_k: int, metric: str) -> np.ndarray:
    features_as_samples = np.asarray(x_train_flat, dtype=float).T
    n_features = features_as_samples.shape[0]
    if n_features <= 1:
        return np.zeros((n_features, n_features), dtype=float)
    n_neighbors = min(top_k + 1, n_features)
    nearest = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nearest.fit(features_as_samples)
    distances, indices = nearest.kneighbors(features_as_samples)
    finite_dist = distances[np.isfinite(distances)]
    scale = float(np.nanmedian(finite_dist)) if len(finite_dist) else 1.0
    scale = scale if scale > 1e-12 else 1.0
    weights = np.zeros((n_features, n_features), dtype=float)
    for i in range(n_features):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            weight = max(0.0, 1.0 - float(dist)) if metric == "cosine" else float(np.exp(-float(dist) / scale))
            weights[i, int(j)] = weight
    return np.maximum(weights, weights.T)


def topk_symmetric(weights: np.ndarray, top_k: int) -> np.ndarray:
    """Keep row-wise top-k positive weights, then symmetrize."""

    w = np.asarray(weights, dtype=float).copy()
    np.fill_diagonal(w, 0.0)
    keep = np.zeros_like(w)
    n_features = w.shape[0]
    for i in range(n_features):
        row = w[i]
        if n_features <= 1:
            continue
        candidates = np.argsort(row)[-(min(top_k, n_features - 1)) :]
        for j in candidates:
            if i != int(j) and row[j] > 0:
                keep[i, int(j)] = row[j]
    return np.maximum(keep, keep.T)


def feature_group(name: str) -> str:
    lower = str(name).lower()
    if lower.startswith("pca_"):
        return "pca"
    for prefix in (
        "dce_corn_starch",
        "dce_corn",
        "cbot_corn",
        "hlj_",
        "jilin_",
        "inner_mongolia_",
        "liaoning_",
        "ne_",
        "china_",
        "us_",
    ):
        if lower.startswith(prefix):
            return prefix.strip("_")
    return lower.split("_")[0]
