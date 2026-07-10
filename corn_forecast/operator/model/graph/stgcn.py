"""STGCN/STConv graph-temporal model wrapper using PyTorch Geometric Temporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class STGCNGraphModel(OfficialGraphForecastAdapter):
    """Complete STGCN wrapper backed by `torch_geometric_temporal.nn.attention.STConv`."""

    architecture = "stgcn"

    def _build_model(self, *, n_nodes: int):
        lookback = int(self.input_shape_[1])
        min_lookback = 2 * (self.temporal_kernel_size - 1) + 1
        if lookback < min_lookback:
            raise ValueError(
                f"STGCN requires lookback >= {min_lookback} for temporal_kernel_size={self.temporal_kernel_size}; "
                f"got {lookback}"
            )
        try:
            from torch import nn
            from torch_geometric_temporal.nn.attention import STConv
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("PyTorch Geometric Temporal", "pip install torch-geometric-temporal", exc)

        torch = self.torch
        hidden_size = self.hidden_size
        kernel_size = self.temporal_kernel_size
        spatial_kernel_size = self.spatial_kernel_size

        class STGCNCore(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = STConv(
                    num_nodes=n_nodes,
                    in_channels=1,
                    hidden_channels=hidden_size,
                    out_channels=1,
                    kernel_size=kernel_size,
                    K=spatial_kernel_size,
                )

            def forward(self, x, edge_index, edge_weight=None):
                out = self.net(x, edge_index=edge_index, edge_weight=edge_weight)
                out = out.mean(dim=1, keepdim=True)
                return ensure_graph_output(torch, out, n_nodes=n_nodes)

        return STGCNCore()

    def _forward_official(self, model, x_tensor):
        return model(x_tensor, self.edge_index_, self.edge_weight_)


def create_stgcn(params: dict | None = None) -> STGCNGraphModel:
    return STGCNGraphModel(**graph_model_params(params))
