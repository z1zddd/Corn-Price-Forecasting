"""iTransformer adapted from Time-Series-Library models/iTransformer.py."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from src.models.base import BaseModel
from src.models.deep._torch_forecaster import TorchForecaster


class ITransformerRegressor(nn.Module):
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
        self.enc_embedding = DataEmbedding_inverted(seq_len, hidden_size, "fixed", "d", dropout)
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
        return self.head(enc_out.mean(dim=1))


class ITransformerForecaster(TorchForecaster):
    network_cls = ITransformerRegressor


class ITransformerClassifier(BaseModel):
    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
        d_ff: int | None = None,
        factor: int = 5,
        activation: str = "gelu",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 8,
        min_delta: float = 1e-5,
        grad_clip: float = 1.0,
        device: str | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.factor = factor
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ITransformerRegressor | None = None
        self.history: dict | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ITransformerClassifier":
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
                logits = self.model(xb).view(-1)
                loss = criterion(logits, yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            self.model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model(valid_x).view(-1), valid_y).detach().cpu())
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

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        x = torch.as_tensor(self._to_btv(X), dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                logits = self.model(x[start : start + self.batch_size]).view(-1)
                probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(probs, axis=0).reshape(-1).astype("float32")

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

    def _build_network(self) -> ITransformerRegressor:
        return ITransformerRegressor(
            input_size=int(self.input_size),
            seq_len=int(self.seq_len),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            factor=self.factor,
            activation=self.activation,
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
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "factor": self.factor,
            "activation": self.activation,
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
