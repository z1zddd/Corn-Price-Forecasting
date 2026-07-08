"""GRU sequence classifier."""

from __future__ import annotations

from torch import nn

from wrappers.torch import TorchSequenceClassifierAdapter


class GRUClassifier(nn.Module):
    def __init__(self, *, n_vars: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(n_vars, hidden_size, num_layers=num_layers, dropout=recurrent_dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        seq = x.transpose(1, 2)
        _, hidden = self.gru(seq)
        return self.head(hidden[-1])


def _build_gru(*, n_vars: int, lookback: int, params: dict):
    return GRUClassifier(
        n_vars=n_vars,
        hidden_size=int(params.get("hidden_size", 32)),
        num_layers=int(params.get("num_layers", 1)),
        dropout=float(params.get("dropout", 0.0)),
    )


def create_gru(params: dict) -> TorchSequenceClassifierAdapter:
    return TorchSequenceClassifierAdapter(model_name="gru", network_builder=_build_gru, params=params)
