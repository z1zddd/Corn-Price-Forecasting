"""Name mapping for reusable operator layers."""

from __future__ import annotations

from collections.abc import Callable

from corn_forecast.operator.layer.decomposition.vmd import VMDLayer
from corn_forecast.operator.layer.graph.adjacency import AdjacencyLayer
from corn_forecast.operator.layer.graph.centrality import GraphCentralityLayer
from corn_forecast.operator.layer.graph.diffusion import DiffusionLayer
from corn_forecast.operator.layer.graph.spectral import SpectralLayer
from corn_forecast.operator.layer.recurrence.pyts_recurrence import RecurrencePlotLayer
from corn_forecast.operator.layer.visibility.ts2vg_visibility import VisibilityGraphLayer


LAYER_REGISTRY: dict[str, Callable[..., object]] = {
    "adjacency": AdjacencyLayer,
    "diffusion": DiffusionLayer,
    "spectral": SpectralLayer,
    "centrality": GraphCentralityLayer,
    "recurrence_plot": RecurrencePlotLayer,
    "visibility_graph": VisibilityGraphLayer,
    "vmd": VMDLayer,
}


def available_layers() -> list[str]:
    """Return registered layer names."""

    return sorted(LAYER_REGISTRY)


def create_layer(name: str, params: dict | None = None):
    """Create a layer by name without touching model registries."""

    try:
        layer_cls = LAYER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown operator layer: {name}") from exc
    return layer_cls(**dict(params or {}))
