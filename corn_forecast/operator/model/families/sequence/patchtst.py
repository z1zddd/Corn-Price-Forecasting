"""PatchTST-style patch MLP classifier."""

from __future__ import annotations

from torch import nn

from corn_forecast.operator.model.wrappers.torch import TorchSequenceClassifierAdapter


class PatchTSTClassifier(nn.Module):
    def __init__(self, *, n_vars: int, lookback: int, patch_len: int, stride: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = max(1, min(int(patch_len), lookback))
        self.stride = max(1, int(stride))
        patch_count = max(1, 1 + (lookback - self.patch_len) // self.stride)
        input_size = n_vars * patch_count * self.patch_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        return self.net(patches)


def _build_patchtst(*, n_vars: int, lookback: int, params: dict):
    return PatchTSTClassifier(
        n_vars=n_vars,
        lookback=lookback,
        patch_len=int(params.get("patch_len", 4)),
        stride=int(params.get("stride", 1)),
        hidden_size=int(params.get("hidden_size", 32)),
        dropout=float(params.get("dropout", 0.0)),
    )


def create_patchtst(params: dict) -> TorchSequenceClassifierAdapter:
    return TorchSequenceClassifierAdapter(model_name="patchtst", network_builder=_build_patchtst, params=params)
