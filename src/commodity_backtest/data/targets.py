"""Forward target generation."""

from __future__ import annotations

import pandas as pd


def add_forward_targets(
    df: pd.DataFrame,
    *,
    price_col: str,
    horizon: int,
    spike_threshold: float = 0.0,
) -> pd.DataFrame:
    """Generate future price, return, and direction targets from a price column."""

    if price_col not in df.columns:
        raise ValueError(f"price_col not found in dataframe: {price_col}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    out = df.copy()
    out["target_price_fwd"] = out[price_col].shift(-horizon)
    out["target_return_fwd"] = out["target_price_fwd"] / out[price_col] - 1.0
    out["target_direction_fwd"] = (out["target_return_fwd"] > spike_threshold).astype(int)
    out = out.dropna(subset=["target_price_fwd", "target_return_fwd", "target_direction_fwd"]).reset_index(drop=True)
    return out