"""Graph statistics and centrality pooling layers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows, flatten_windows
from corn_forecast.operator.layer.graph.adjacency import AdjacencyLayer, graph_info


def graph_centrality_weights(adjacency: np.ndarray) -> np.ndarray:
    """Return degree, pagerank, and eigenvector centrality weights."""

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for centrality pooling. Install with: pip install networkx") from exc
    raw = np.asarray(adjacency, dtype=float)
    graph = nx.from_numpy_array(np.maximum(raw - np.eye(raw.shape[0]), 0.0))
    degree = np.asarray([value for _, value in nx.degree_centrality(graph).items()], dtype=float)
    try:
        pagerank_dict = nx.pagerank(graph, weight="weight", max_iter=100)
        pagerank = np.asarray([pagerank_dict[i] for i in range(raw.shape[0])], dtype=float)
    except Exception:
        pagerank = degree.copy()
    try:
        eigen_dict = nx.eigenvector_centrality_numpy(graph, weight="weight")
        eigen = np.asarray([eigen_dict[i] for i in range(raw.shape[0])], dtype=float)
    except Exception:
        eigen = degree.copy()
    weights = np.column_stack([degree, pagerank, eigen])
    denom = np.sum(np.abs(weights), axis=0, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    return weights / denom


def centrality_pool(x: np.ndarray, adjacency: np.ndarray, *, flatten: bool = True) -> np.ndarray:
    """Pool node windows by learned centrality summaries."""

    windows = ensure_3d_windows(x)
    weights = graph_centrality_weights(adjacency)
    pooled = np.einsum("bvt,vk->bkt", windows, weights)
    out = np.concatenate([windows, pooled], axis=1)
    return flatten_windows(out) if flatten else out


def graph_stats(x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Return compact raw/smoothed graph-signal statistics."""

    windows = ensure_3d_windows(x)
    smooth = np.einsum("ij,bjt->bit", np.asarray(adjacency, dtype=float), windows)
    delta = windows - smooth
    return np.concatenate(
        [
            windows.mean(axis=2),
            windows[:, :, -1],
            smooth.mean(axis=2),
            delta.mean(axis=2),
            delta.std(axis=2),
        ],
        axis=1,
    ).astype(float)


@dataclass
class GraphCentralityLayer(LayerMixin):
    """Fit an adjacency matrix, then return graph stats or centrality pools."""

    method: str = "centrality_pool"
    graph_builder: str = "corr_abs_top5"
    feature_cols: list[str] | None = None
    flatten: bool = True
    adjacency_layer_: AdjacencyLayer = field(init=False)
    graph_info_: dict[str, float | int] = field(default_factory=dict, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        if self.method not in {"centrality_pool", "graph_stats"}:
            raise ValueError(f"Unknown centrality/stat transform: {self.method}")
        self.adjacency_layer_ = AdjacencyLayer(builder=self.graph_builder, feature_cols=self.feature_cols)
        self.adjacency_layer_.fit(x, y=y)
        self.graph_info_ = graph_info(self.adjacency_layer_.adjacency_)
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("GraphCentralityLayer is not fitted")
        adjacency = self.adjacency_layer_.adjacency_
        if self.method == "graph_stats":
            return graph_stats(x, adjacency)
        return centrality_pool(x, adjacency, flatten=self.flatten)
