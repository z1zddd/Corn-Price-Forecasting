"""Backtest split generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestWindow:
    """One chronological backtest window."""

    window_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray | None
    test_idx: np.ndarray
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def make_backtest_windows(
    dates,
    *,
    mode: str,
    min_train_periods: int,
    stride_periods: int = 1,
    window_size_periods: int | None = None,
    max_train_periods: int | None = None,
) -> list[BacktestWindow]:
    """Create expanding, rolling, or capped expanding windows."""

    parsed_dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    n = len(parsed_dates)
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be >= 1")
    if stride_periods < 1:
        raise ValueError("stride_periods must be >= 1")
    if min_train_periods >= n:
        raise ValueError("min_train_periods must leave at least one test point")

    windows: list[BacktestWindow] = []
    window_id = 0
    for test_start in range(min_train_periods, n, stride_periods):
        test_end = test_start + 1
        train_end = test_start
        if mode == "expanding":
            train_start = 0
        elif mode == "rolling":
            if window_size_periods is None:
                raise ValueError("rolling mode requires window_size_periods")
            if window_size_periods < min_train_periods:
                raise ValueError("window_size_periods must be >= min_train_periods")
            train_start = max(0, train_end - window_size_periods)
        elif mode == "expanding_with_cap":
            if max_train_periods is None:
                raise ValueError("expanding_with_cap mode requires max_train_periods")
            if max_train_periods < min_train_periods:
                raise ValueError("max_train_periods must be >= min_train_periods")
            train_start = max(0, train_end - max_train_periods)
        else:
            raise ValueError("mode must be expanding, rolling, or expanding_with_cap")

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        windows.append(
            BacktestWindow(
                window_id=window_id,
                train_idx=train_idx,
                val_idx=None,
                test_idx=test_idx,
                train_start_date=parsed_dates.iloc[train_idx[0]],
                train_end_date=parsed_dates.iloc[train_idx[-1]],
                test_start_date=parsed_dates.iloc[test_idx[0]],
                test_end_date=parsed_dates.iloc[test_idx[-1]],
            )
        )
        window_id += 1
    return windows


def assert_temporal_holdout(dates, train_idx, val_idx, test_idx) -> None:
    """Raise if chronological holdout is violated."""

    parsed_dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    train_dates = parsed_dates.iloc[np.asarray(train_idx)]
    test_dates = parsed_dates.iloc[np.asarray(test_idx)]
    if train_dates.max() >= test_dates.min():
        raise ValueError(f"Temporal leakage: train max {train_dates.max()} >= test min {test_dates.min()}")

    if val_idx is not None and len(val_idx) > 0:
        val_dates = parsed_dates.iloc[np.asarray(val_idx)]
        if train_dates.max() >= val_dates.min():
            raise ValueError(f"Temporal leakage: train max {train_dates.max()} >= val min {val_dates.min()}")
        if val_dates.max() >= test_dates.min():
            raise ValueError(f"Temporal leakage: val max {val_dates.max()} >= test min {test_dates.min()}")
