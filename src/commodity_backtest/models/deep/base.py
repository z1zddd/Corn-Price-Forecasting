"""Shared adapter for optional torch sequence classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from commodity_backtest.train.trainer import train_binary_classifier


class TorchSequenceClassifierAdapter:
    """Adapter exposing fit/predict_proba/predict/save for torch sequence models."""

    def __init__(self, *, model_name: str, network_builder: Callable, params: dict) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for deep sequence models. Install with: pip install -e .[deep]") from exc
        self.torch = torch
        self.model_name = model_name
        self.network_builder = network_builder
        self.params = dict(params)
        self.hidden_size = int(self.params.get("hidden_size", 32))
        self.num_layers = int(self.params.get("num_layers", 1))
        self.dropout = float(self.params.get("dropout", 0.0))
        self.epochs = int(self.params.get("epochs", 20))
        self.batch_size = int(self.params.get("batch_size", 32))
        self.lr = float(self.params.get("lr", 0.001))
        self.patience = int(self.params.get("patience", 5))
        self.random_state = int(self.params.get("random_state", 42))
        requested_device = str(self.params.get("device", "cpu"))
        self.device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model = None
        self.input_shape: tuple[int, int] | None = None
        self.model_family = "deep_sequence"
        self.disabled_by_default = True
        self.min_train_samples = int(self.params.get("min_train_samples", 30))

    def _build_model(self, x_train: np.ndarray):
        n_vars = int(x_train.shape[1])
        lookback = int(x_train.shape[2])
        self.input_shape = (n_vars, lookback)
        return self.network_builder(n_vars=n_vars, lookback=lookback, params=self.params)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        self.torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = self._build_model(np.asarray(x_train, dtype=np.float32))
        self.model = train_binary_classifier(
            self.model,
            np.asarray(x_train, dtype=np.float32),
            np.asarray(y_train, dtype=int),
            x_val=np.asarray(x_val, dtype=np.float32) if x_val is not None else None,
            y_val=np.asarray(y_val, dtype=int) if y_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            device=self.device,
        )
        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        x = self.torch.as_tensor(np.asarray(x_test, dtype=np.float32), dtype=self.torch.float32, device=self.device)
        self.model.eval()
        with self.torch.no_grad():
            logits = self.model(x)
            return self.torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x_test) > 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        self.torch.save(
            {
                "model_name": self.model_name,
                "state_dict": self.model.state_dict(),
                "params": self.params,
                "input_shape": self.input_shape,
            },
            path,
        )
