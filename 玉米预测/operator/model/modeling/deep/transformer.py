"""Transformer adapted from Time-Series-Library models/Transformer.py.

Changes: remove x_mark/x_dec arguments, keep encoder blocks, and replace the
task-specific projection with the local LayerNorm -> Linear -> SiLU head.
"""

from __future__ import annotations

from torch import nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.deep._torch_forecaster import TorchForecaster


class TransformerRegressor(nn.Module):
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
        **_,
    ):
        super().__init__()
        d_ff = d_ff or hidden_size * 4
        self.enc_embedding = DataEmbedding(input_size, hidden_size, "fixed", "d", dropout)
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
            norm_layer=nn.LayerNorm(hidden_size),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.head(enc_out[:, -1, :])


class TransformerForecaster(TorchForecaster):
    network_cls = TransformerRegressor

