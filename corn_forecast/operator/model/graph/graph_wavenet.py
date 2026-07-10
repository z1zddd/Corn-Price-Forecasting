"""GraphWaveNet graph-temporal model wrapper using Torch Spatiotemporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class GraphWaveNetGraphModel(OfficialGraphForecastAdapter):
    """Complete GraphWaveNet wrapper backed by `tsl.nn.models.GraphWaveNetModel`."""

    architecture = "graph_wavenet"

    def _build_model(self, *, n_nodes: int):
        try:
            from tsl.nn.models import GraphWaveNetModel
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("Torch Spatiotemporal", "pip install torch-spatiotemporal", exc)
        return GraphWaveNetModel(
            input_size=1,
            output_size=1,
            horizon=1,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            temporal_kernel_size=self.temporal_kernel_size,
            spatial_kernel_size=self.spatial_kernel_size,
            n_nodes=n_nodes,
        )

    def _forward_official(self, model, x_tensor):
        out = model(x_tensor, edge_index=self.edge_index_, edge_weight=self.edge_weight_)
        return ensure_graph_output(self.torch, out, n_nodes=x_tensor.shape[2])


def create_graph_wavenet(params: dict | None = None) -> GraphWaveNetGraphModel:
    return GraphWaveNetGraphModel(**graph_model_params(params))
