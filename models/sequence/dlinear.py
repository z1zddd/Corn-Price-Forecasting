"""DLinear-style direct linear classifier."""

from __future__ import annotations

from torch import nn

from wrappers.torch import TorchSequenceClassifierAdapter


class DLinearClassifier(nn.Module):
    def __init__(self, *, n_vars: int, lookback: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_vars * lookback, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


def _build_dlinear(*, n_vars: int, lookback: int, params: dict):
    return DLinearClassifier(
        n_vars=n_vars,
        lookback=lookback,
        hidden_size=int(params.get("hidden_size", 32)),
        dropout=float(params.get("dropout", 0.0)),
    )


def create_dlinear(params: dict) -> TorchSequenceClassifierAdapter:
    return TorchSequenceClassifierAdapter(model_name="dlinear", network_builder=_build_dlinear, params=params)
