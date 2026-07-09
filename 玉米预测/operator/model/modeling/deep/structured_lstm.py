"""Single-stream LSTM classifier for structured-only spike ablations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class StructuredLSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dense_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dense_dim, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x_seq)
        z = self.dropout(torch.relu(self.fc1(hidden[-1])))
        return self.out(z).squeeze(-1)


class StructuredLSTMClassifier(BaseModel):
    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        dropout: float = 0.3,
        epochs: int = 120,
        batch_size: int = 16,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        patience: int = 15,
        min_delta: float = 1e-5,
        grad_clip: float = 1.0,
        device: str | None = None,
        attn_dim: int | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_dim = attn_dim
        self.model: StructuredLSTMNet | None = None
        self.history: dict | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "StructuredLSTMClassifier":
        x_train = self._to_btv(X_train)
        x_val = self._to_btv(X_val) if X_val is not None else x_train
        y_val = y_train if y_val is None else y_val
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = StructuredLSTMNet(
            input_dim=self.input_size,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            dropout=self.dropout,
        ).to(self.device)

        dense_params = list(self.model.fc1.parameters()) + list(self.model.out.parameters())
        dense_ids = {id(param) for param in dense_params}
        other_params = [param for param in self.model.parameters() if id(param) not in dense_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "weight_decay": 0.0},
                {"params": dense_params, "weight_decay": self.weight_decay},
            ],
            lr=self.lr,
        )
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

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        x = torch.as_tensor(self._to_btv(X), dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                logits = self.model(x[start : start + self.batch_size])
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
        obj.model = StructuredLSTMNet(
            input_dim=int(obj.input_size),
            hidden_dim=obj.hidden_dim,
            dense_dim=obj.dense_dim,
            dropout=obj.dropout,
        )
        obj.model.load_state_dict(payload["state_dict"])
        return obj

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
            "hidden_dim": self.hidden_dim,
            "dense_dim": self.dense_dim,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "grad_clip": self.grad_clip,
            "device": self.device,
            "attn_dim": self.attn_dim,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")
