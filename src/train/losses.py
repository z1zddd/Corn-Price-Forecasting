"""Loss functions: PyTorch built-ins plus MADL for direction-aware returns."""

from __future__ import annotations

import torch
from torch import nn


def madl_loss(pred_returns: torch.Tensor, actual_returns: torch.Tensor, prev_prices: torch.Tensor | None = None) -> torch.Tensor:
    """Mean Absolute Directional Loss from the supervision document."""

    pred_dir = torch.sign(pred_returns)
    return -torch.mean(pred_dir * actual_returns)


def get_loss_fn(task: str):
    if task == "regression":
        return nn.MSELoss()
    if task == "classification":
        return nn.BCEWithLogitsLoss()
    if task == "madl":
        return madl_loss
    raise ValueError(f"Unknown loss task: {task}")

