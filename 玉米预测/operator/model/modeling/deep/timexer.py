"""TimeXer classifier adapted from the official THUML TimeXer implementation.

Official source: https://github.com/thuml/TimeXer/blob/main/models/TimeXer.py
The backbone keeps TimeXer's endogenous patch embedding, exogenous inverted
embedding, and endogenous-to-exogenous cross-attention design. The forecasting
head is used as a one-step score head and trained with BCE logits for the spike
classification task.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from src.models.base import BaseModel


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class TimeXerEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class TimeXerEncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        batch_size, _, dim = cross.shape
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (batch_size, -1, dim))
        x_glb_attn = self.dropout(
            self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn,
            (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2]),
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + x_glb_attn)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class TimeXerScoreNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int = 128,
        d_ff: int | None = None,
        e_layers: int = 2,
        n_heads: int = 4,
        factor: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        patch_len: int = 4,
        embed: str = "fixed",
        freq: str = "m",
        use_norm: bool = True,
    ):
        super().__init__()
        if seq_len < patch_len:
            raise ValueError(f"seq_len={seq_len} must be >= patch_len={patch_len}.")
        if seq_len % patch_len != 0:
            raise ValueError("TimeXer requires seq_len divisible by patch_len for this reproduction.")
        self.input_size = input_size
        self.seq_len = seq_len
        self.use_norm = use_norm
        self.patch_len = patch_len
        self.patch_num = int(seq_len // patch_len)
        self.n_vars = 1
        self.en_embedding = EnEmbedding(self.n_vars, d_model, patch_len, dropout)
        self.ex_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        d_ff = d_ff or d_model
        self.encoder = TimeXerEncoder(
            [
                TimeXerEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(1, self.head_nf, 1, head_dropout=dropout)

    def forward(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], None)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        score = self.head(enc_out)
        return score[:, 0, 0]


class TimeXerClassifier(BaseModel):
    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        d_model: int = 128,
        d_ff: int | None = None,
        e_layers: int = 2,
        n_heads: int = 4,
        factor: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        patch_len: int = 4,
        use_norm: bool = True,
        epochs: int = 80,
        batch_size: int = 16,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        patience: int = 12,
        min_delta: float = 1e-5,
        grad_clip: float = 1.0,
        device: str | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.patch_len = patch_len
        self.use_norm = use_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: TimeXerScoreNet | None = None
        self.history: dict | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "TimeXerClassifier":
        x_train = self._to_btv(X_train)
        x_val = self._to_btv(X_val) if X_val is not None else x_train
        y_val = y_train if y_val is None else y_val
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = self._build_network().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        train_loader = self._loader(x_train, y_train, shuffle=True)
        valid_x = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        valid_y = torch.as_tensor(y_val, dtype=torch.float32, device=self.device).view(-1)

        best_state = deepcopy(self.model.state_dict())
        best_valid = float("inf")
        stale = 0
        history = {"epoch": [], "train_loss": [], "val_loss": []}
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).view(-1)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            self.model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model(valid_x), valid_y).detach().cpu())
            history["epoch"].append(epoch)
            history["train_loss"].append(float(np.mean(losses)))
            history["val_loss"].append(val_loss)
            if val_loss < best_valid - self.min_delta:
                best_valid = val_loss
                best_state = deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.history = history
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)

    def predict_logits(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        x = torch.as_tensor(self._to_btv(X), dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                logits.append(self.model(x[start : start + self.batch_size]).detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1).astype("float32")

    def predict_proba(self, X) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-np.clip(self.predict_logits(X), -50.0, 50.0)))).astype("float32")

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "params": self._params()}, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        obj = cls(**payload["params"])
        obj.model = obj._build_network()
        obj.model.load_state_dict(payload["state_dict"])
        return obj

    def _build_network(self) -> TimeXerScoreNet:
        return TimeXerScoreNet(
            input_size=int(self.input_size),
            seq_len=int(self.seq_len),
            d_model=self.d_model,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            factor=self.factor,
            dropout=self.dropout,
            activation=self.activation,
            patch_len=self.patch_len,
            use_norm=self.use_norm,
        )

    def _loader(self, x, y, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _params(self) -> dict:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "n_heads": self.n_heads,
            "factor": self.factor,
            "dropout": self.dropout,
            "activation": self.activation,
            "patch_len": self.patch_len,
            "use_norm": self.use_norm,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "grad_clip": self.grad_clip,
            "device": self.device,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")
