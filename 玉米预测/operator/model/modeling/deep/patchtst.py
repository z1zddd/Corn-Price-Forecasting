"""PatchTST adapted from Time-Series-Library models/PatchTST.py."""

from __future__ import annotations

import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.deep._torch_forecaster import TorchForecaster


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        out = x.transpose(*self.dims)
        return out.contiguous() if self.contiguous else out


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
        d_ff: int | None = None,
        factor: int = 5,
        activation: str = "gelu",
        patch_len: int = 16,
        stride: int = 8,
        **_,
    ):
        super().__init__()
        d_ff = d_ff or hidden_size * 4
        patch_len = min(patch_len, seq_len)
        stride = max(1, min(stride, patch_len))
        self.patch_embedding = PatchEmbedding(hidden_size, patch_len, stride, stride, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        pooled = enc_out.mean(dim=(1, 2))
        return self.head(pooled)


class PatchTSTForecaster(TorchForecaster):
    network_cls = PatchTSTRegressor

