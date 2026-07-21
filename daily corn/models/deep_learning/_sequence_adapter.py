from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchSequencePriceRegressor:
    """Shared 2D-flattened-window adapter for sequence price regressors."""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        if "lookback" not in params:
            raise ValueError("lookback is required for sequence models")
        self.lookback = int(params["lookback"])
        self.hidden_size = int(params.get("hidden_size", 32))
        self.batch_size = int(params.get("batch_size", 64))
        self.max_epochs = int(params.get("max_epochs", 60))
        self.patience = int(params.get("patience", 8))
        self.learning_rate = float(params.get("learning_rate", 1e-3))
        self.weight_decay = float(params.get("weight_decay", 1e-5))
        self.gradient_clip = float(params.get("gradient_clip", 1.0))
        self.random_state = int(params.get("random_state", 42))
        requested_device = str(params.get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(requested_device)
        if min(self.lookback, self.hidden_size, self.batch_size, self.max_epochs, self.patience) <= 0:
            raise ValueError("Sequence model size and training parameters must be positive")
        self.model: nn.Module | None = None
        self.input_size_: int | None = None
        self.input_mean_: np.ndarray | None = None
        self.input_std_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_std_: float | None = None
        self.best_epoch_ = 0
        self.best_validation_rmse_ = np.nan

    def _build_network(self, input_size: int) -> nn.Module:
        raise NotImplementedError

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _reshape(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] % self.lookback != 0:
            raise ValueError(
                "Sequence input must be a 2D flattened window whose width is divisible by lookback"
            )
        input_size = array.shape[1] // self.lookback
        if self.input_size_ is not None and input_size != self.input_size_:
            raise ValueError("Input feature count differs from the fitted sequence model")
        return array.reshape(len(array), self.lookback, input_size)

    def _transform_X(self, values: np.ndarray, fit: bool) -> np.ndarray:
        sequence = self._reshape(values)
        if fit:
            self.input_size_ = sequence.shape[2]
            self.input_mean_ = sequence.mean(axis=(0, 1), keepdims=True)
            self.input_std_ = sequence.std(axis=(0, 1), keepdims=True)
            self.input_std_[self.input_std_ < 1e-6] = 1.0
        if self.input_mean_ is None or self.input_std_ is None:
            raise RuntimeError("Sequence input scaler is not fitted")
        transformed = (sequence - self.input_mean_) / self.input_std_
        if not np.isfinite(transformed).all():
            raise ValueError("Sequence input scaling produced non-finite values")
        return transformed.astype(np.float32)

    def _transform_y(self, values: np.ndarray, fit: bool) -> np.ndarray:
        target = np.asarray(values, dtype=np.float32).reshape(-1)
        if fit:
            self.target_mean_ = float(target.mean())
            standard_deviation = float(target.std())
            self.target_std_ = standard_deviation if standard_deviation >= 1e-6 else 1.0
        if self.target_mean_ is None or self.target_std_ is None:
            raise RuntimeError("Sequence target scaler is not fitted")
        return ((target - self.target_mean_) / self.target_std_).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_price_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "TorchSequencePriceRegressor":
        self._set_seed()
        X_scaled = self._transform_X(X_train, fit=True)
        y_scaled = self._transform_y(y_price_train, fit=True)
        if len(X_scaled) != len(y_scaled):
            raise ValueError("Training inputs and targets have different lengths")
        self.model = self._build_network(int(self.input_size_)).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_scaled)),
            batch_size=min(self.batch_size, len(X_scaled)),
            shuffle=False,
        )
        validation_tensors = None
        if validation_data is not None:
            X_validation, y_validation = validation_data
            validation_tensors = (
                torch.from_numpy(self._transform_X(X_validation, fit=False)).to(self.device),
                np.asarray(y_validation, dtype=float).reshape(-1),
            )
        best_loss = np.inf
        best_state = None
        stale_epochs = 0
        loss_function = nn.MSELoss()
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(
                    self.model(X_batch.to(self.device)), y_batch.to(self.device)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                optimizer.step()
            if validation_tensors is None:
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                continue
            self.model.eval()
            with torch.no_grad():
                scaled = self.model(validation_tensors[0]).cpu().numpy()
            prices = scaled * float(self.target_std_) + float(self.target_mean_)
            monitored_loss = float(np.sqrt(np.mean((validation_tensors[1] - prices) ** 2)))
            if monitored_loss < best_loss - 1e-8:
                best_loss = monitored_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("Sequence training did not produce a valid checkpoint")
        self.model.load_state_dict(best_state)
        self.best_validation_rmse_ = best_loss if validation_data is not None else np.nan
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.target_mean_ is None or self.target_std_ is None:
            raise RuntimeError("Sequence model must be fitted before prediction")
        values = torch.from_numpy(self._transform_X(X_test, fit=False)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(values).cpu().numpy().astype(float)
        prices = scaled * self.target_std_ + self.target_mean_
        if not np.isfinite(prices).all():
            raise ValueError(f"{type(self).__name__} produced non-finite price predictions")
        return prices.reshape(-1)

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_size_ is None:
            raise RuntimeError("Cannot save an unfitted sequence model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "params": self.params,
                "state_dict": self.model.state_dict(),
                "input_size": self.input_size_,
                "input_mean": self.input_mean_,
                "input_std": self.input_std_,
                "target_mean": self.target_mean_,
                "target_std": self.target_std_,
                "best_epoch": self.best_epoch_,
                "best_validation_rmse": self.best_validation_rmse_,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TorchSequencePriceRegressor":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        params = dict(payload["params"])
        params["device"] = "cpu"
        instance = cls(**params)
        instance.input_size_ = int(payload["input_size"])
        instance.input_mean_ = np.asarray(payload["input_mean"], dtype=np.float32)
        instance.input_std_ = np.asarray(payload["input_std"], dtype=np.float32)
        instance.target_mean_ = float(payload["target_mean"])
        instance.target_std_ = float(payload["target_std"])
        instance.best_epoch_ = int(payload["best_epoch"])
        instance.best_validation_rmse_ = float(payload["best_validation_rmse"])
        instance.model = instance._build_network(instance.input_size_)
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device)
        return instance
