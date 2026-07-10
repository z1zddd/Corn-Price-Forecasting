"""ASTGCN graph-temporal model wrapper using PyTorch Geometric Temporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class ASTGCNGraphModel(OfficialGraphForecastAdapter):
    """Complete ASTGCN wrapper backed by `torch_geometric_temporal.nn.attention.ASTGCN`."""

    architecture = "astgcn"

    def _build_model(self, *, n_nodes: int):
        try:
            from torch import nn
            from torch_geometric_temporal.nn.attention import ASTGCN
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("PyTorch Geometric Temporal", "pip install torch-geometric-temporal", exc)

        torch = self.torch
        hidden_size = self.hidden_size
        spatial_kernel_size = self.spatial_kernel_size
        lookback = int(self.input_shape_[1])

        class ASTGCNCore(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = ASTGCN(
                    nb_block=1,
                    in_channels=1,
                    K=spatial_kernel_size,
                    nb_chev_filter=hidden_size,
                    nb_time_filter=hidden_size,
                    time_strides=1,
                    num_for_predict=1,
                    len_input=lookback,
                    num_of_vertices=n_nodes,
                )

            def forward(self, x, edge_index):
                out = self.net(x.permute(0, 2, 3, 1), edge_index)
                return ensure_graph_output(torch, out, n_nodes=n_nodes)

        return ASTGCNCore()

    def _forward_official(self, model, x_tensor):
        return model(x_tensor, self.edge_index_)


def create_astgcn(params: dict | None = None) -> ASTGCNGraphModel:
    return ASTGCNGraphModel(**graph_model_params(params))
