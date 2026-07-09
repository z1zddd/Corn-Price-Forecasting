"""CNN-family classifiers for short multivariate time-series windows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel


class CNNClassifier(BaseModel):
    network_cls = None

    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        hidden_size: int = 64,
        dropout: float = 0.2,
        epochs: int = 80,
        batch_size: int = 16,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        patience: int = 12,
        min_delta: float = 1e-5,
        grad_clip: float = 1.0,
        pos_weight: str | float | None = "auto",
        device: str | None = None,
        **kwargs,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.pos_weight = pos_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
        self.model: nn.Module | None = None
        self.history: dict | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        x_train = self._to_btv(X_train)
        x_val = self._to_btv(X_val) if X_val is not None else x_train
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        y_val = y_train if y_val is None else np.asarray(y_val, dtype=np.float32).reshape(-1)
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = self._build_network().to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor(y_train))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        valid_x = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        valid_y = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)

        best_state = deepcopy(self.model.state_dict())
        best_valid = float("inf")
        stale = 0
        history = {"epoch": [], "train_loss": [], "val_loss": []}
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
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
        logits = self.predict_logits(X)
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))).astype("float32")

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

    def _build_network(self) -> nn.Module:
        if self.network_cls is None:
            raise NotImplementedError("network_cls must be set.")
        return self.network_cls(
            input_size=int(self.input_size),
            seq_len=int(self.seq_len),
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            **self.kwargs,
        )

    def _pos_weight_tensor(self, y: np.ndarray) -> torch.Tensor | None:
        if self.pos_weight is None:
            return None
        if self.pos_weight == "auto":
            pos = float(np.sum(y == 1))
            neg = float(np.sum(y == 0))
            value = 1.0 if pos < 1 else max(neg / pos, 1e-3)
        else:
            value = float(self.pos_weight)
        return torch.as_tensor([value], dtype=torch.float32, device=self.device)

    def _params(self) -> dict:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "grad_clip": self.grad_clip,
            "pos_weight": self.pos_weight,
            "device": self.device,
            **self.kwargs,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same", dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FCNNet(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int = 128, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(input_size, hidden_size, 8, dropout=dropout),
            ConvBNReLU(hidden_size, hidden_size * 2, 5, dropout=dropout),
            ConvBNReLU(hidden_size * 2, hidden_size, 3, dropout=dropout),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_btv):
        x = x_btv.transpose(1, 2)
        z = self.net(x).mean(dim=-1)
        return self.head(z).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 8, dropout=dropout)
        self.conv2 = ConvBNReLU(out_channels, out_channels, 5, dropout=dropout)
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv3(self.conv2(self.conv1(x))) + self.shortcut(x))


class ResNet1DNet(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int = 64, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(input_size, hidden_size, dropout=dropout),
            ResidualBlock(hidden_size, hidden_size * 2, dropout=dropout),
            ResidualBlock(hidden_size * 2, hidden_size * 2, dropout=dropout),
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x_btv):
        z = self.net(x_btv.transpose(1, 2)).mean(dim=-1)
        return self.head(z).squeeze(-1)


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 32, bottleneck_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) if use_bottleneck else nn.Identity()
        conv_in = bottleneck_channels if use_bottleneck else in_channels
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(conv_in, out_channels, kernel, dropout=dropout)
                for kernel in (9, 5, 3)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        xb = self.bottleneck(x)
        return torch.cat([branch(xb) for branch in self.branches] + [self.pool_branch(x)], dim=1)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        modules = []
        channels = in_channels
        for _ in range(depth):
            modules.append(InceptionModule(channels, out_channels=out_channels, dropout=dropout))
            channels = out_channels * 4
        self.modules_ = nn.ModuleList(modules)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels))
            if in_channels != channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = x
        for module in self.modules_:
            y = module(y)
        return self.relu(y + self.shortcut(x))


class InceptionTimeNet(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int = 32, dropout: float = 0.2, blocks: int = 2, **kwargs):
        super().__init__()
        layers = []
        channels = input_size
        for _ in range(blocks):
            block = InceptionBlock(channels, out_channels=hidden_size, depth=3, dropout=dropout)
            layers.append(block)
            channels = hidden_size * 4
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x_btv):
        z = self.net(x_btv.transpose(1, 2)).mean(dim=-1)
        return self.head(z).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class TCNNet(nn.Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int = 64, dropout: float = 0.2, levels: int = 4, kernel_size: int = 3, **kwargs):
        super().__init__()
        layers = []
        in_channels = input_size
        for level in range(levels):
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size=kernel_size, dilation=2**level, dropout=dropout))
            in_channels = hidden_size
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_btv):
        z = self.net(x_btv.transpose(1, 2))[:, :, -1]
        return self.head(z).squeeze(-1)


class FCNClassifier(CNNClassifier):
    network_cls = FCNNet


class ResNet1DClassifier(CNNClassifier):
    network_cls = ResNet1DNet


class InceptionTimeClassifier(CNNClassifier):
    network_cls = InceptionTimeNet


class TCNClassifier(CNNClassifier):
    network_cls = TCNNet
