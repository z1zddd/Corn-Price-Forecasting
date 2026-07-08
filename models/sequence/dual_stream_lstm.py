"""Dual-stream LSTM classifier for structured and PCA/news features."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from wrappers.torch import TorchSequenceClassifierAdapter


_PCA_PATTERN = re.compile(r"^pca_?\d+$", re.IGNORECASE)


def split_feature_indices(
    feature_cols: Sequence[str],
    *,
    news_prefix: str = "pca_",
) -> tuple[list[int], list[int]]:
    """Split feature columns into structured and PCA/news branches."""

    news_prefix_lower = news_prefix.lower()
    news_indices = [
        idx
        for idx, column in enumerate(feature_cols)
        if str(column).lower().startswith(news_prefix_lower) or _PCA_PATTERN.match(str(column))
    ]
    news_set = set(news_indices)
    structured_indices = [idx for idx in range(len(feature_cols)) if idx not in news_set]

    if not structured_indices:
        structured_indices = list(range(len(feature_cols)))
        news_indices = []
    return structured_indices, news_indices


class DualStreamLSTMClassifier(nn.Module):
    """Two-branch LSTM: market/structured features plus PCA/news features."""

    def __init__(
        self,
        *,
        structured_indices: Sequence[int],
        news_indices: Sequence[int],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        attn_size: int,
        dense_size: int,
    ) -> None:
        super().__init__()
        self.structured_indices = list(structured_indices)
        self.news_indices = list(news_indices)
        recurrent_dropout = dropout if num_layers > 1 else 0.0

        self.structured_lstm = nn.LSTM(
            len(self.structured_indices),
            hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        if self.news_indices:
            self.news_lstm = nn.LSTM(
                len(self.news_indices),
                hidden_size,
                num_layers=num_layers,
                dropout=recurrent_dropout,
                batch_first=True,
            )
            self.q_proj = nn.Linear(hidden_size, attn_size)
            self.k_proj = nn.Linear(hidden_size, attn_size)
            self.v_proj = nn.Linear(hidden_size, attn_size)
            fused_size = hidden_size + attn_size
        else:
            self.news_lstm = None
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            fused_size = hidden_size

        self.head = nn.Sequential(
            nn.Linear(fused_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x):
        structured_seq = x[:, self.structured_indices, :].transpose(1, 2)
        _, (structured_hidden, _) = self.structured_lstm(structured_seq)
        h_structured = structured_hidden[-1]

        if self.news_lstm is None:
            fused = h_structured
        else:
            news_seq = x[:, self.news_indices, :].transpose(1, 2)
            news_hidden, _ = self.news_lstm(news_seq)
            q = self.q_proj(news_hidden)
            k = self.k_proj(news_hidden)
            v = self.v_proj(news_hidden)
            scores = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])
            weights = scores.softmax(dim=-1)
            h_news = (weights @ v).mean(dim=1)
            fused = torch.cat([h_structured, h_news], dim=1)

        return self.head(fused)


class DualStreamLSTMAdapter(TorchSequenceClassifierAdapter):
    """Adapter that derives branch feature indices from column names."""

    def __init__(self, params: dict) -> None:
        super().__init__(model_name="dual_stream_lstm", network_builder=_build_dual_stream_lstm, params=params)
        self.feature_cols = self._normalize_feature_cols(self.params.get("feature_cols"))
        self.news_prefix = str(self.params.get("news_prefix", "pca_"))
        self.structured_indices: list[int] = []
        self.news_indices: list[int] = []
        if self.feature_cols is not None:
            self.structured_indices, self.news_indices = split_feature_indices(
                self.feature_cols,
                news_prefix=self.news_prefix,
            )

    @staticmethod
    def _normalize_feature_cols(feature_cols) -> list[str] | None:
        if feature_cols is None:
            return None
        if isinstance(feature_cols, str):
            return [feature_cols]
        return [str(column) for column in feature_cols]

    def _build_model(self, x_train: np.ndarray):
        n_vars = int(x_train.shape[1])
        lookback = int(x_train.shape[2])
        self.input_shape = (n_vars, lookback)

        if self.feature_cols is None:
            news_feature_count = int(self.params.get("news_feature_count", 0))
            if news_feature_count < 0 or news_feature_count > n_vars:
                raise ValueError("news_feature_count must be between 0 and the number of features")
            split_at = n_vars - news_feature_count
            self.structured_indices = list(range(split_at))
            self.news_indices = list(range(split_at, n_vars))
            if not self.structured_indices:
                self.structured_indices = list(range(n_vars))
                self.news_indices = []
        else:
            if len(self.feature_cols) != n_vars:
                raise ValueError(
                    "dual_stream_lstm params.feature_cols length must match training features "
                    f"({len(self.feature_cols)} != {n_vars})"
                )
            self.structured_indices, self.news_indices = split_feature_indices(
                self.feature_cols,
                news_prefix=self.news_prefix,
            )

        return _build_dual_stream_lstm(n_vars=n_vars, lookback=lookback, params=self.params, adapter=self)


def _build_dual_stream_lstm(
    *,
    n_vars: int,
    lookback: int,
    params: dict,
    adapter: DualStreamLSTMAdapter | None = None,
):
    if adapter is None:
        feature_cols = DualStreamLSTMAdapter._normalize_feature_cols(params.get("feature_cols"))
        if feature_cols is not None and len(feature_cols) == n_vars:
            structured_indices, news_indices = split_feature_indices(
                feature_cols,
                news_prefix=str(params.get("news_prefix", "pca_")),
            )
        else:
            news_feature_count = int(params.get("news_feature_count", 0))
            split_at = max(0, n_vars - news_feature_count)
            structured_indices = list(range(split_at)) or list(range(n_vars))
            news_indices = list(range(split_at, n_vars)) if split_at > 0 else []
    else:
        structured_indices = adapter.structured_indices
        news_indices = adapter.news_indices

    return DualStreamLSTMClassifier(
        structured_indices=structured_indices,
        news_indices=news_indices,
        hidden_size=int(params.get("hidden_size", 32)),
        num_layers=int(params.get("num_layers", 1)),
        dropout=float(params.get("dropout", 0.0)),
        attn_size=int(params.get("attn_size", 16)),
        dense_size=int(params.get("dense_size", 32)),
    )


def create_dual_stream_lstm(params: dict) -> DualStreamLSTMAdapter:
    return DualStreamLSTMAdapter(params)
