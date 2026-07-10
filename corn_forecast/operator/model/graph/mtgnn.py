"""MTGNN graph-temporal model wrapper using PyTorch Geometric Temporal."""

from __future__ import annotations

from corn_forecast.operator.model.graph.base import OfficialGraphForecastAdapter, graph_model_params
from corn_forecast.operator.model.graph.utils import ensure_graph_output, optional_dependency_error


class MTGNNGraphModel(OfficialGraphForecastAdapter):
    """Complete MTGNN wrapper backed by `torch_geometric_temporal.nn.attention.MTGNN`."""

    architecture = "mtgnn"

    def _build_model(self, *, n_nodes: int):
        try:
            from torch import nn
            from torch_geometric_temporal.nn.attention import MTGNN
        except (ImportError, OSError) as exc:
            raise optional_dependency_error("PyTorch Geometric Temporal", "pip install torch-geometric-temporal", exc)

        torch = self.torch
        hidden_size = self.hidden_size
        ff_size = self.ff_size
        kernel_size = self.temporal_kernel_size
        lookback = int(self.input_shape_[1])
        dropout = self.dropout
        n_layers = self.n_layers

        class MTGNNCore(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = MTGNN(
                    gcn_true=True,
                    build_adj=False,
                    gcn_depth=2,
                    num_nodes=n_nodes,
                    kernel_set=[kernel_size],
                    kernel_size=kernel_size,
                    dropout=dropout,
                    subgraph_size=min(max(2, n_nodes), 20),
                    node_dim=max(4, min(hidden_size, 16)),
                    dilation_exponential=1,
                    conv_channels=hidden_size,
                    residual_channels=hidden_size,
                    skip_channels=max(ff_size, hidden_size),
                    end_channels=max(ff_size, hidden_size),
                    seq_length=lookback,
                    in_dim=1,
                    out_dim=1,
                    layers=n_layers,
                    propalpha=0.05,
                    tanhalpha=3,
                    layer_norm_affline=True,
                )

            def forward(self, x, adjacency):
                out = self.net(x.permute(0, 3, 2, 1), A_tilde=adjacency)
                return ensure_graph_output(torch, out, n_nodes=n_nodes)

        return MTGNNCore()

    def _forward_official(self, model, x_tensor):
        return model(x_tensor, self.adjacency_tensor_)


def create_mtgnn(params: dict | None = None) -> MTGNNGraphModel:
    return MTGNNGraphModel(**graph_model_params(params))
