"""Time split logic adapted from Time-Series-Library Dataset_Custom."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    dates,
    method: str = "fixed_date",
    test_start: str | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices for time-ordered samples."""

    date_index = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    n = len(date_index)
    if n < 3:
        raise ValueError("Need at least 3 samples for train/val/test split.")

    if method == "fixed_date":
        if not test_start:
            raise ValueError("test_start is required for fixed_date splitting.")
        test_mask = date_index >= pd.Timestamp(test_start)
        train_val_idx = np.flatnonzero(~test_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(test_idx) == 0:
            raise ValueError(f"test_start={test_start} produced an empty test split.")
        if len(train_val_idx) < 2:
            raise ValueError("fixed_date split leaves fewer than two train/val samples.")
        val_count = max(1, int(round(len(train_val_idx) * val_ratio)))
        train_idx = train_val_idx[:-val_count]
        val_idx = train_val_idx[-val_count:]
    elif method == "ratio":
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
        val_end = min(val_end, n - 1)
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
    else:
        raise ValueError(f"Unknown split method: {method}")

    return train_idx, val_idx, test_idx

