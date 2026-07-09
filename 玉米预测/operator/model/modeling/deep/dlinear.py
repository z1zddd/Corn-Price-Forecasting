"""DLinear adapted from Time-Series-Library models/DLinear.py."""

from __future__ import annotations

import torch
from torch import nn

from layers.Autoformer_EncDec import series_decomp
from src.models.deep._torch_forecaster import TorchForecaster


class DLinearRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
        moving_avg: int = 25,
        individual: bool = False,
        **_,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.channels = input_size
        self.individual = individual
        self.decompsition = series_decomp(moving_avg)
        if individual:
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(seq_len, seq_len) for _ in range(input_size)])
            self.Linear_Trend = nn.ModuleList([nn.Linear(seq_len, seq_len) for _ in range(input_size)])
            for i in range(input_size):
                self.Linear_Seasonal[i].weight = nn.Parameter((1 / seq_len) * torch.ones([seq_len, seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter((1 / seq_len) * torch.ones([seq_len, seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, seq_len)
            self.Linear_Trend = nn.Linear(seq_len, seq_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / seq_len) * torch.ones([seq_len, seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / seq_len) * torch.ones([seq_len, seq_len]))
        self.head = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros_like(seasonal_init)
            trend_output = torch.zeros_like(trend_init)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)

    def forward(self, x):
        enc_out = self.encoder(x)
        return self.head(enc_out[:, -1, :])


class DLinearForecaster(TorchForecaster):
    network_cls = DLinearRegressor

