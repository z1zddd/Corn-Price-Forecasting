"""Sliding window construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Make [N, V, T] windows from a sorted dataframe."""

    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    for column in [*feature_cols, target_col, date_col]:
        if column not in df.columns:
            raise ValueError(f"Column not found for windowing: {column}")

    values = df[feature_cols].to_numpy(dtype=float)
    target = df[target_col].to_numpy()
    dates = pd.to_datetime(df[date_col]).reset_index(drop=True)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict] = []
    for end_idx in range(lookback - 1, len(df)):
        start_idx = end_idx - lookback + 1
        window = values[start_idx : end_idx + 1].T
        x_rows.append(window)
        y_rows.append(target[end_idx])
        meta_rows.append(
            {
                "row_idx": end_idx,
                "date": dates.iloc[end_idx],
                "window_start_date": dates.iloc[start_idx],
                "window_end_date": dates.iloc[end_idx],
            }
        )

    if not x_rows:
        raise ValueError("No windows produced; reduce lookback or add more rows")
    return np.stack(x_rows), np.asarray(y_rows), pd.DataFrame(meta_rows)