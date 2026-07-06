"""Transformer encoder sequence classifier."""

from __future__ import annotations

from torch import nn

from models.deep.base import TorchSequenceClassifierAdapter


class TransformerClassifier(nn.Module):
    def __init__(self, *, n_vars: int, hidden_size: int, n_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            n_heads = 1
        self.input_projection = nn.Linear(n_vars, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, n_heads),
            dim_feedforward=max(hidden_size * 2, 8),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        seq = x.transpose(1, 2)
        encoded = self.encoder(self.input_projection(seq))
        return self.head(encoded.mean(dim=1))


def _build_transformer(*, n_vars: int, lookback: int, params: dict):
    return TransformerClassifier(
        n_vars=n_vars,
        hidden_size=int(params.get("hidden_size", 32)),
        n_heads=int(params.get("n_heads", 2)),
        num_layers=int(params.get("num_layers", 1)),
        dropout=float(params.get("dropout", 0.0)),
    )


def create_transformer(params: dict) -> TorchSequenceClassifierAdapter:
    return TorchSequenceClassifierAdapter(model_name="transformer", network_builder=_build_transformer, params=params)

