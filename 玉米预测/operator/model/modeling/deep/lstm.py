"""LSTM model following the head specified in the supervision document."""

from __future__ import annotations

from torch import nn

from src.models.deep._torch_forecaster import TorchForecaster


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int = 96, num_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMForecaster(TorchForecaster):
    network_cls = LSTMRegressor
