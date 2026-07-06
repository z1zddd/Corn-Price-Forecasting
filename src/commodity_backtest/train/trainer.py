"""Small training loop shared by optional torch classifiers."""

from __future__ import annotations

import copy

import numpy as np


def train_binary_classifier(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 5,
    device: str = "cpu",
):
    """Fit a torch binary classifier on [N, V, T] sequence windows."""

    import torch
    from torch.nn import functional as F

    model.to(device)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(y_train, dtype=float).reshape(-1, 1), dtype=torch.float32, device=device)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=device) if x_val is not None else None
    y_val_tensor = (
        torch.as_tensor(np.asarray(y_val, dtype=float).reshape(-1, 1), dtype=torch.float32, device=device)
        if y_val is not None
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    bad_epochs = 0
    batch_size = max(1, int(batch_size))

    for _ in range(max(1, int(epochs))):
        model.train()
        order = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if x_val_tensor is not None and y_val_tensor is not None and len(y_val_tensor):
                eval_loss = F.binary_cross_entropy_with_logits(model(x_val_tensor), y_val_tensor).item()
            else:
                eval_loss = F.binary_cross_entropy_with_logits(model(x), y).item()
        if eval_loss < best_loss - 1e-9:
            best_loss = eval_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= max(1, int(patience)):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model
