"""Torch loss helpers for optional deep models."""

from __future__ import annotations


def binary_focal_loss(logits, targets, *, gamma: float = 2.0):
    """Compute focal BCE loss for binary logits."""

    import torch
    from torch.nn import functional as F

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return ((1.0 - p_t) ** float(gamma) * bce).mean()

