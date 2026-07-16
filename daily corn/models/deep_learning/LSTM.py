from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.output(encoded[:, -1, :]).squeeze(-1)


class LSTMPriceRegressor:
    """Price-regression adapter backed by PyTorch's BSD-3-Clause LSTM."""

    source = "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.lookback = int(self.params["lookback"])
        self.hidden_size = int(self.params.get("hidden_size", 32))
        self.num_layers = int(self.params.get("num_layers", 1))
        self.dropout = float(self.params.get("dropout", 0.0))
        self.learning_rate = float(self.params.get("learning_rate", 1e-3))
        self.batch_size = int(self.params.get("batch_size", 64))
        self.max_epochs = int(self.params.get("max_epochs", 60))
        self.patience = int(self.params.get("patience", 8))
        self.weight_decay = float(self.params.get("weight_decay", 1e-5))
        self.gradient_clip = float(self.params.get("gradient_clip", 1.0))
        self.random_state = int(self.params.get("random_state", 42))
        requested_device = str(self.params.get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(requested_device)
        self.model: _LSTMNetwork | None = None
        self.input_mean_: np.ndarray | None = None
        self.input_std_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_std_: float | None = None
        self.input_size_: int | None = None
        self.best_epoch_ = 0
        self.best_validation_rmse_ = np.nan
        self._validate_params()

    def _validate_params(self) -> None:
        positive = {
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"LSTM parameters must be positive: {invalid}")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid LSTM optimization parameters")

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
                "LSTM input must be a 2D flattened sequence whose width is "
                "divisible by lookback"
            )
        input_size = array.shape[1] // self.lookback
        if self.input_size_ is not None and input_size != self.input_size_:
            raise ValueError("LSTM input feature count differs from the fitted model")
        return array.reshape(array.shape[0], self.lookback, input_size)

    def _scale_X(self, values: np.ndarray, fit: bool) -> np.ndarray:
        sequence = self._reshape(values)
        if fit:
            self.input_size_ = sequence.shape[2]
            self.input_mean_ = sequence.mean(axis=(0, 1), keepdims=True)
            self.input_std_ = sequence.std(axis=(0, 1), keepdims=True)
            self.input_std_[self.input_std_ < 1e-6] = 1.0
        if self.input_mean_ is None or self.input_std_ is None:
            raise RuntimeError("LSTM input scaler is not fitted")
        scaled = (sequence - self.input_mean_) / self.input_std_
        if not np.isfinite(scaled).all():
            raise ValueError("LSTM input scaling produced non-finite values")
        return scaled.astype(np.float32)

    def _scale_y(self, values: np.ndarray, fit: bool) -> np.ndarray:
        target = np.asarray(values, dtype=np.float32).reshape(-1)
        if fit:
            self.target_mean_ = float(target.mean())
            target_std = float(target.std())
            self.target_std_ = target_std if target_std >= 1e-6 else 1.0
        if self.target_mean_ is None or self.target_std_ is None:
            raise RuntimeError("LSTM target scaler is not fitted")
        return ((target - self.target_mean_) / self.target_std_).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_price_train: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "LSTMPriceRegressor":
        self._set_seed()
        X_scaled = self._scale_X(X_train, fit=True)
        y_scaled = self._scale_y(y_price_train, fit=True)
        if len(X_scaled) != len(y_scaled):
            raise ValueError("Training inputs and targets have different lengths")

        self.model = _LSTMNetwork(
            input_size=int(self.input_size_),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.MSELoss()
        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_scaled))
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=False,
            generator=generator,
        )

        validation_tensors: tuple[torch.Tensor, np.ndarray] | None = None
        if validation_data is not None:
            X_validation, y_validation = validation_data
            validation_tensors = (
                torch.from_numpy(self._scale_X(X_validation, fit=False)).to(self.device),
                np.asarray(y_validation, dtype=float).reshape(-1),
            )

        best_loss = np.inf
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(self.model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                optimizer.step()

            if validation_tensors is None:
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                continue
            else:
                self.model.eval()
                with torch.no_grad():
                    predicted_scaled = self.model(validation_tensors[0]).cpu().numpy()
                predicted_price = (
                    predicted_scaled * float(self.target_std_) + float(self.target_mean_)
                )
                monitored_loss = float(
                    np.sqrt(np.mean((validation_tensors[1] - predicted_price) ** 2))
                )

            if monitored_loss < best_loss - 1e-8:
                best_loss = monitored_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if validation_tensors is not None and epochs_without_improvement >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("LSTM training did not produce a valid checkpoint")
        self.model.load_state_dict(best_state)
        self.best_validation_rmse_ = best_loss if validation_data is not None else np.nan
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.target_mean_ is None or self.target_std_ is None:
            raise RuntimeError("LSTM model must be fitted before prediction")
        X_scaled = torch.from_numpy(self._scale_X(X_test, fit=False)).to(self.device)
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_scaled), self.batch_size):
                batch = X_scaled[start : start + self.batch_size]
                predictions.append(self.model(batch).cpu().numpy())
        scaled = np.concatenate(predictions).astype(float)
        prices = scaled * self.target_std_ + self.target_mean_
        if not np.isfinite(prices).all():
            raise ValueError("LSTM produced non-finite price predictions")
        return prices.reshape(-1)

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_size_ is None:
            raise RuntimeError("Cannot save an unfitted LSTM model")
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
    def load(cls, path: str | Path) -> "LSTMPriceRegressor":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        instance = cls(**payload["params"])
        instance.device = torch.device("cpu")
        instance.input_size_ = int(payload["input_size"])
        instance.input_mean_ = np.asarray(payload["input_mean"], dtype=np.float32)
        instance.input_std_ = np.asarray(payload["input_std"], dtype=np.float32)
        instance.target_mean_ = float(payload["target_mean"])
        instance.target_std_ = float(payload["target_std"])
        instance.best_epoch_ = int(payload["best_epoch"])
        instance.best_validation_rmse_ = float(payload["best_validation_rmse"])
        instance.model = _LSTMNetwork(
            input_size=instance.input_size_,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout,
        )
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.to(instance.device)
        return instance
