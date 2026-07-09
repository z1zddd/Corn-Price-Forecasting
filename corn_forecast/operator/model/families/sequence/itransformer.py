"""iTransformer-style inverted variable projection classifier."""

from __future__ import annotations

from torch import nn

from corn_forecast.operator.model.wrappers.torch import TorchSequenceClassifierAdapter


class ITransformerClassifier(nn.Module):
    def __init__(self, *, lookback: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.variable_projection = nn.Linear(lookback, hidden_size)
        self.net = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        inverted_features = self.variable_projection(x)
        pooled = inverted_features.mean(dim=1)
        return self.net(pooled)


def _build_itransformer(*, n_vars: int, lookback: int, params: dict):
    return ITransformerClassifier(
        lookback=lookback,
        hidden_size=int(params.get("hidden_size", 32)),
        dropout=float(params.get("dropout", 0.0)),
    )


def create_itransformer(params: dict) -> TorchSequenceClassifierAdapter:
    return TorchSequenceClassifierAdapter(model_name="itransformer", network_builder=_build_itransformer, params=params)
