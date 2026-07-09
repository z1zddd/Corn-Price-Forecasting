"""Trainer skeleton adapted from lightning-hydra-template and train_tcn_multivar.py."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device: str | torch.device = "cpu",
        callbacks: list | None = None,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 8,
        grad_clip: float = 1.0,
        monitor: str = "val_loss",
        y_inverse_fn=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.callbacks = callbacks or []
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.monitor = monitor
        self.y_inverse_fn = y_inverse_fn

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.to(self.device)
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_x = torch.as_tensor(X_val if X_val is not None else X_train, dtype=torch.float32, device=self.device)
        valid_y = torch.as_tensor(y_val if y_val is not None else y_train, dtype=torch.float32, device=self.device).view(-1, 1)

        best_state = deepcopy(self.model.state_dict())
        best_valid = float("inf")
        stale = 0
        history = {"epoch": [], "train_loss": [], "val_loss": []}
        if self.monitor == "val_mae_raw":
            if self.y_inverse_fn is None:
                raise ValueError("monitor='val_mae_raw' requires y_inverse_fn.")
            history["val_mae"] = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(valid_x)
                val_loss = float(self.loss_fn(val_pred, valid_y).detach().cpu())
            train_loss = float(np.mean(losses))
            monitor_value = val_loss
            row_extra = {}
            if self.monitor == "val_mae_raw":
                pred_raw = self.y_inverse_fn(val_pred.detach().cpu().numpy().reshape(-1))
                true_raw = self.y_inverse_fn(valid_y.detach().cpu().numpy().reshape(-1))
                val_mae = float(np.mean(np.abs(np.asarray(true_raw) - np.asarray(pred_raw))))
                monitor_value = val_mae
                row_extra["val_mae"] = val_mae
            elif self.monitor != "val_loss":
                raise ValueError(f"Unknown monitor: {self.monitor}")

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            for key, value in row_extra.items():
                history[key].append(value)
            if monitor_value < best_valid:
                best_valid = monitor_value
                best_state = deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        return self.model, history
