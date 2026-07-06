"""Data diagnosis helpers for CSV onboarding."""

from __future__ import annotations

import pandas as pd


PRICE_HINTS = ("close", "settle", "price", "收盘", "结算", "价格")
DATE_HINTS = ("date", "month", "time", "日期", "月份")
TARGET_HINTS = ("target", "spike", "label", "direction", "next")


def diagnose_frame(df: pd.DataFrame) -> dict:
    """Return a machine-readable diagnosis summary."""

    lower_names = {column: str(column).lower() for column in df.columns}
    date_candidates = [column for column, lowered in lower_names.items() if any(hint in lowered for hint in DATE_HINTS)]
    price_candidates = [
        column
        for column, lowered in lower_names.items()
        if any(hint in lowered for hint in PRICE_HINTS) and pd.api.types.is_numeric_dtype(df[column])
    ]
    target_like_columns = [column for column, lowered in lower_names.items() if any(hint in lowered for hint in TARGET_HINTS)]
    numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    missing_ratio = {column: float(df[column].isna().mean()) for column in df.columns}
    negative_numeric = {column: int((df[column] < 0).sum()) for column in numeric_cols}
    warnings: list[str] = []
    if not date_candidates:
        warnings.append("No date-like column detected.")
    if not price_candidates:
        warnings.append("No numeric price-like column detected.")
    if len(df) < 3:
        warnings.append("Too few rows for a time-series backtest.")
    high_missing = [column for column, ratio in missing_ratio.items() if ratio > 0.0]
    if high_missing:
        warnings.append(f"Missing values detected in columns: {', '.join(map(str, high_missing))}.")
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate rows detected.")

    if not date_candidates or not price_candidates or len(df) < 3 or not numeric_cols:
        status = "unusable"
    elif warnings:
        status = "usable_with_warnings"
    else:
        status = "usable"

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "status": status,
        "warnings": warnings,
        "date_candidates": date_candidates,
        "price_candidates": price_candidates,
        "target_like_columns": target_like_columns,
        "numeric_columns": numeric_cols,
        "p_over_n": float(len(numeric_cols) / len(df)) if len(df) else 0.0,
        "missing_ratio": missing_ratio,
        "duplicate_rows": duplicate_rows,
        "negative_numeric_counts": negative_numeric,
    }
