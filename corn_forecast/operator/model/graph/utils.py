"""Utilities shared by graph-temporal model wrappers."""

from __future__ import annotations

import numpy as np


def optional_dependency_error(package: str, install_hint: str, exc: BaseException) -> ImportError:
    """Return a consistent optional dependency error."""

    error = ImportError(f"{package} is required for this graph temporal model. Install with: {install_hint}")
    error.__cause__ = exc
    return error


def sigmoid(x) -> np.ndarray:
    """Numerically stable sigmoid as numpy."""

    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50.0, 50.0)))


def as_windows(x, *, name: str = "x") -> np.ndarray:
    """Return finite float windows shaped [samples, nodes, lookback]."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected {name} with shape [n_samples, n_nodes, lookback], got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def ensure_graph_output(torch_module, output, *, n_nodes: int):
    """Coerce upstream outputs to [batch, horizon, nodes, channels]."""

    out = output
    if out.ndim == 2:
        out = out.reshape(out.shape[0], 1, n_nodes, 1)
    elif out.ndim == 3:
        if out.shape[1] == n_nodes:
            out = out.permute(0, 2, 1).unsqueeze(-1)
        elif out.shape[2] == n_nodes:
            out = out.unsqueeze(-1)
        else:
            out = out.reshape(out.shape[0], 1, n_nodes, -1)
    elif out.ndim == 4:
        if out.shape[1] == n_nodes and out.shape[2] != n_nodes:
            out = out.permute(0, 2, 1, 3)
        elif out.shape[2] != n_nodes and out.shape[1] != 1:
            out = out.reshape(out.shape[0], 1, n_nodes, -1)
    else:
        out = out.reshape(out.shape[0], 1, n_nodes, -1)
    if out.shape[1] != 1:
        out = out.mean(dim=1, keepdim=True)
    if out.shape[-1] != 1:
        out = out.mean(dim=-1, keepdim=True)
    return out.to(dtype=torch_module.float32)
