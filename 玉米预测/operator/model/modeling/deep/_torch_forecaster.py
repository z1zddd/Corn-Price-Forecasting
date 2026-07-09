"""Shared PyTorch BaseModel adapter for sequence forecasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.base import BaseModel
from src.train.trainer import Trainer


class TorchForecaster(BaseModel):
    network_cls = None

    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 8,
        grad_clip: float = 1.0,
        monitor: str = "val_loss",
        device: str | None = None,
        **kwargs,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.grad_clip = grad_clip
        self.monitor = monitor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
        self.model: nn.Module | None = None
        self.history: dict | None = None
        self._y_inverse_fn = None

    def set_y_inverse(self, y_inverse_fn) -> None:
        self._y_inverse_fn = y_inverse_fn

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        x_train = self._to_btv(X_train)
        x_val = self._to_btv(X_val) if X_val is not None else None
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = self._build_network()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        trainer = Trainer(
            model=self.model,
            loss_fn=nn.MSELoss(),
            optimizer=optimizer,
            device=self.device,
            batch_size=self.batch_size,
            epochs=self.epochs,
            patience=self.patience,
            grad_clip=self.grad_clip,
            monitor=self.monitor,
            y_inverse_fn=self._y_inverse_fn,
        )
        self.model, self.history = trainer.train(x_train, y_train, x_val, y_val)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        x = torch.as_tensor(self._to_btv(X), dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                preds.append(self.model(x[start : start + self.batch_size]).detach().cpu().numpy())
        return np.concatenate(preds, axis=0).reshape(-1).astype("float32")

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "params": {
                    "input_size": self.input_size,
                    "seq_len": self.seq_len,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "patience": self.patience,
                    "grad_clip": self.grad_clip,
                    "monitor": self.monitor,
                    "device": self.device,
                    **self.kwargs,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        obj = cls(**payload["params"])
        obj.model = obj._build_network()
        obj.model.load_state_dict(payload["state_dict"])
        return obj

    def _build_network(self) -> nn.Module:
        if self.network_cls is None:
            raise NotImplementedError("network_cls must be set.")
        return self.network_cls(
            input_size=int(self.input_size),
            seq_len=int(self.seq_len),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **self.kwargs,
        )

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")
