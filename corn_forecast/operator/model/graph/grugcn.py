"""GRUGCN graph-temporal model wrapper using Torch Spatiotemporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class GRUGCNGraphModel(OfficialGraphForecastAdapter):
    """Complete GRUGCN wrapper backed by `tsl.nn.models.GRUGCNModel`."""

    architecture = "grugcn"

    def _build_model(self, *, n_nodes: int):
        try:
            from tsl.nn.models import GRUGCNModel
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("Torch Spatiotemporal", "pip install torch-spatiotemporal", exc)
        return GRUGCNModel(
            input_size=1,
            hidden_size=self.hidden_size,
            output_size=1,
            horizon=1,
            exog_size=0,
            enc_layers=self.n_layers,
            gcn_layers=1,
            norm="mean",
        )

    def _forward_official(self, model, x_tensor):
        out = model(x_tensor, edge_index=self.edge_index_, edge_weight=self.edge_weight_)
        return ensure_graph_output(self.torch, out, n_nodes=x_tensor.shape[2])


def create_grugcn(params: dict | None = None) -> GRUGCNGraphModel:
    return GRUGCNGraphModel(**graph_model_params(params))
