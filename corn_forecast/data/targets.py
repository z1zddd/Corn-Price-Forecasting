"""Forward target generation."""

from __future__ import annotations

import pandas as pd


def add_forward_targets(
    df: pd.DataFrame,
    *,
    price_col: str,
    horizon: int,
    spike_threshold: float = 0.0,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Generate future price, return, and direction targets from a price column."""

    if price_col not in df.columns:
        raise ValueError(f"price_col not found in dataframe: {price_col}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if date_col is not None and date_col not in df.columns:
        raise ValueError(f"date_col not found in dataframe: {date_col}")

    out = df.copy()
    out["target_price_fwd"] = out[price_col].shift(-horizon)
    out["target_return_fwd"] = out["target_price_fwd"] / out[price_col] - 1.0
    out["target_direction_fwd"] = (out["target_return_fwd"] > spike_threshold).astype(int)
    required = ["target_price_fwd", "target_return_fwd", "target_direction_fwd"]
    if date_col is not None:
        out["target_date_fwd"] = out[date_col].shift(-horizon)
        required.append("target_date_fwd")
    out = out.dropna(subset=required).reset_index(drop=True)
    return out
