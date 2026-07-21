from __future__ import annotations

import torch
from torch import nn


def compatible_heads(hidden_size: int, requested: int) -> int:
    return max(heads for heads in range(1, min(hidden_size, requested) + 1) if hidden_size % heads == 0)


class RecurrentNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(values)
        return self.output(encoded[:, -1]).squeeze(-1)


class LiquidNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.candidate = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        state = values.new_zeros((len(values), self.output.in_features))
        for step in range(values.shape[1]):
            projected = torch.tanh(self.input_projection(values[:, step]))
            joined = torch.cat((projected, state), dim=-1)
            gate = torch.sigmoid(self.gate(joined))
            candidate = torch.tanh(self.candidate(joined))
            state = gate * state + (1.0 - gate) * candidate
        return self.output(state).squeeze(-1)


class TransformerNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 2) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            compatible_heads(hidden_size, num_heads),
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(self.projection(values))[:, -1]).squeeze(-1)


class TemporalConvNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.network(values.transpose(1, 2))[:, :, : values.shape[1]]
        return self.output(encoded[:, :, -1]).squeeze(-1)


class RAFTNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.residual = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = torch.relu(self.projection(values))
        weights = torch.softmax(self.score(encoded), dim=1)
        summary = (weights * encoded).sum(dim=1)
        return self.output(summary + self.residual(summary)).squeeze(-1)


class MAFSNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, input_size), nn.Sigmoid()
        )
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weights = self.feature_gate(values.mean(dim=1)).unsqueeze(1)
        encoded, _ = self.encoder(values * weights)
        return self.output(encoded[:, -1]).squeeze(-1)


class DLinearNetwork(nn.Module):
    """DLinear decomposition with a scalar future-price regression head."""

    def __init__(
        self,
        lookback: int,
        input_size: int,
        kernel_size: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DLinear kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.individual = bool(individual)
        self.input_size = int(input_size)
        self.moving_average = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)
        if self.individual:
            self.trend = nn.ModuleList(nn.Linear(lookback, 1) for _ in range(input_size))
            self.seasonal = nn.ModuleList(nn.Linear(lookback, 1) for _ in range(input_size))
        else:
            self.trend = nn.Linear(lookback, 1)
            self.seasonal = nn.Linear(lookback, 1)
        self.output = nn.Linear(input_size * 2, 1)

    def _decompose(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transposed = values.transpose(1, 2)
        padding = (self.kernel_size - 1) // 2
        padded = torch.nn.functional.pad(transposed, (padding, padding), mode="replicate")
        trend = self.moving_average(padded)
        return transposed - trend, trend

    def _project(self, values: torch.Tensor, layers: nn.Module) -> torch.Tensor:
        if not self.individual:
            return layers(values)
        return torch.cat(
            [layer(values[:, index, :]).unsqueeze(1) for index, layer in enumerate(layers)],
            dim=1,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self._decompose(values)
        combined = torch.cat(
            (self._project(trend, self.trend), self._project(seasonal, self.seasonal)), dim=1
        )
        return self.output(combined.squeeze(-1)).squeeze(-1)


class XLinearNetwork(nn.Module):
    def __init__(self, lookback: int, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lookback = lookback
        self.projection = nn.Sequential(
            nn.Linear(input_size * 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        short = values[:, -max(1, self.lookback // 4) :].mean(dim=1)
        medium = values[:, -max(1, self.lookback // 2) :].mean(dim=1)
        long = values.mean(dim=1)
        return self.projection(torch.cat((short, medium, long), dim=-1)).squeeze(-1)


class PatchTransformerNetwork(nn.Module):
    def __init__(self, lookback: int, input_size: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = min(max(1, patch_size), lookback)
        self.projection = nn.Linear(input_size * self.patch_size, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, compatible_heads(hidden_size, 2), hidden_size * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        patches = values.unfold(1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 1, 3, 2).reshape(len(values), patches.shape[1], -1)
        encoded = self.encoder(self.projection(patches)).mean(dim=1)
        return self.output(encoded).squeeze(-1)


class InvertedTransformerNetwork(nn.Module):
    def __init__(self, lookback: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(lookback, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, compatible_heads(hidden_size, 2), hidden_size * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(values.transpose(1, 2))
        return self.output(self.encoder(tokens).mean(dim=1)).squeeze(-1)
