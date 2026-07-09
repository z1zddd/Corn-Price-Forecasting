"""Rolling-origin cross-validation helpers.

Inspired by Nixtla cross_validation, Darts historical forecasts, sktime
evaluate splitters, and sklearn TimeSeriesSplit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingWindow:
    window_id: int
    cutoff_date: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def rolling_origin_splits(
    target_dates,
    h: int,
    n_windows: int,
    step_size: int | None = None,
    min_train_size: int | None = None,
    max_train_size: int | None = None,
) -> list[RollingWindow]:
    """Create rolling-origin train/test splits over target dates.

    Parameters mirror common forecasting libraries:
    - `h`: test window length / forecast horizon window.
    - `n_windows`: number of historical evaluation windows.
    - `step_size`: distance between cutoffs. Defaults to `h`.
    """

    dates = pd.to_datetime(pd.Series(target_dates)).reset_index(drop=True)
    n = len(dates)
    if h <= 0:
        raise ValueError("h must be positive.")
    if n_windows <= 0:
        raise ValueError("n_windows must be positive.")
    step = int(step_size or h)
    min_train = int(min_train_size or max(h * 3, 30))

    last_test_end = n
    windows: list[RollingWindow] = []
    for raw_window_id in range(n_windows):
        test_end = last_test_end - raw_window_id * step
        test_start = test_end - h
        train_end = test_start
        if test_start < min_train:
            break
        train_start = 0 if max_train_size is None else max(0, train_end - int(max_train_size))
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < min_train or len(test_idx) != h:
            continue
        windows.append(
            RollingWindow(
                window_id=len(windows),
                cutoff_date=dates.iloc[train_idx[-1]],
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )

    windows.reverse()
    return [RollingWindow(i, w.cutoff_date, w.train_idx, w.test_idx) for i, w in enumerate(windows)]


def assert_temporal_holdout(meta: pd.DataFrame, train_idx, val_idx, test_idx) -> None:
    """Fail fast on common leakage mistakes."""

    date_col = "target_date" if "target_date" in meta else "date"
    dates = pd.to_datetime(meta[date_col]).reset_index(drop=True)
    train_max = dates.iloc[np.asarray(train_idx)].max()
    val_min = dates.iloc[np.asarray(val_idx)].min()
    val_max = dates.iloc[np.asarray(val_idx)].max()
    test_min = dates.iloc[np.asarray(test_idx)].min()
    if train_max >= val_min:
        raise ValueError(f"Temporal leakage: train max {train_max} >= val min {val_min}.")
    if val_max >= test_min:
        raise ValueError(f"Temporal leakage: val max {val_max} >= test min {test_min}.")

