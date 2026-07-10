"""DCRNN graph-temporal model wrapper using Torch Spatiotemporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class DCRNNGraphModel(OfficialGraphForecastAdapter):
    """Complete DCRNN wrapper backed by `tsl.nn.models.DCRNNModel`."""

    architecture = "dcrnn"

    def _build_model(self, *, n_nodes: int):
        try:
            from tsl.nn.models import DCRNNModel
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("Torch Spatiotemporal", "pip install torch-spatiotemporal", exc)
        return DCRNNModel(
            input_size=1,
            output_size=1,
            horizon=1,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            kernel_size=self.kernel_size,
        )

    def _forward_official(self, model, x_tensor):
        out = model(x_tensor, edge_index=self.edge_index_, edge_weight=self.edge_weight_)
        return ensure_graph_output(self.torch, out, n_nodes=x_tensor.shape[2])


def create_dcrnn(params: dict | None = None) -> DCRNNGraphModel:
    return DCRNNGraphModel(**graph_model_params(params))
