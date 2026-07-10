"""AGCRN graph-temporal model wrapper using Torch Spatiotemporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class AGCRNGraphModel(OfficialGraphForecastAdapter):
    """Complete AGCRN wrapper backed by `tsl.nn.models.AGCRNModel`."""

    architecture = "agcrn"

    def _build_model(self, *, n_nodes: int):
        try:
            from tsl.nn.models import AGCRNModel
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("Torch Spatiotemporal", "pip install torch-spatiotemporal", exc)
        return AGCRNModel(
            input_size=1,
            output_size=1,
            horizon=1,
            n_nodes=n_nodes,
            hidden_size=self.hidden_size,
            emb_size=min(10, max(2, self.hidden_size // 2)),
            exog_size=0,
            n_layers=self.n_layers,
        )

    def _forward_official(self, model, x_tensor):
        out = model(x_tensor)
        return ensure_graph_output(self.torch, out, n_nodes=x_tensor.shape[2])


def create_agcrn(params: dict | None = None) -> AGCRNGraphModel:
    return AGCRNGraphModel(**graph_model_params(params))
