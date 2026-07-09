"""GRU variant copied from the LSTM wrapper pattern with nn.GRU replacing nn.LSTM."""

from __future__ import annotations

from torch import nn

from src.models.deep._torch_forecaster import TorchForecaster


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int = 96, num_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class GRUForecaster(TorchForecaster):
    network_cls = GRURegressor

