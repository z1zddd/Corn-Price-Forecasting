"""Sliding-window extraction adapted from the project's existing window scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int = 30,
    horizon: int = 30,
    include_today: bool = True,
    date_col: str = "date",
    today_col: str = "dce_corn_close",
    series_id: str = "corn",
    target_alignment: str = "anchor",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create X as [N, V, T] and y as [N].

    The anchor date is the last date visible to the model. Target columns are
    usually expected to be aligned to the anchor row, e.g. target_return_fwd at
    row t is the return from t to t + horizon. Some raw label columns are stored
    on the future row itself; set target_alignment="target_idx" for those.
    """
    if target_alignment not in {"anchor", "target_idx"}:
        raise ValueError("target_alignment must be 'anchor' or 'target_idx'.")

    values = df[feature_cols].to_numpy(dtype="float32")
    targets = df[target_col].to_numpy(dtype="float32")
    today = df[today_col].to_numpy(dtype="float32")
    dates = pd.to_datetime(df[date_col]).reset_index(drop=True)
    offset = 0 if include_today else 1

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict[str, object]] = []
    first_anchor = seq_len - 1 + offset
    last_anchor = len(df) - horizon - 1
    for anchor in range(first_anchor, last_anchor + 1):
        end = anchor + 1 - offset
        start = end - seq_len
        if start < 0:
            continue
        target_idx = anchor + horizon
        window = values[start:end]
        target = targets[target_idx] if target_alignment == "target_idx" else targets[anchor]
        if np.isnan(window).all() or np.isnan(target):
            continue
        x_rows.append(window.T)
        y_rows.append(float(target))
        meta_rows.append(
            {
                "sample_idx": len(x_rows) - 1,
                "series_id": series_id,
                "anchor_idx": anchor,
                "target_idx": target_idx,
                "date": dates.iloc[anchor],
                "target_date": dates.iloc[target_idx],
                "horizon": horizon,
                "today_close": float(today[anchor]),
            }
        )

    if not x_rows:
        raise ValueError("No valid windows were produced; check seq_len, horizon, and missing values.")
    return np.stack(x_rows).astype("float32"), np.asarray(y_rows, dtype="float32"), pd.DataFrame(meta_rows)
