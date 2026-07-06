"""CSV loading and feature selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_TARGET_EXCLUDES = {
    "target_price_fwd",
    "target_return_fwd",
    "target_direction_fwd",
}


def load_commodity_csv(
    csv_path: str | Path,
    *,
    date_col: str,
    encodings: list[str] | tuple[str, ...] = ("utf-8", "gbk", "gb18030"),
) -> tuple[pd.DataFrame, str]:
    """Load a commodity CSV with encoding fallbacks and sorted dates."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    errors: list[str] = []
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            if date_col not in df.columns:
                raise ValueError(f"date_col not found in CSV: {date_col}")
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            return df, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    joined = "; ".join(errors)
    raise ValueError(f"Unable to decode {path}. Tried: {joined}")


def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: str | list[str],
    *,
    date_col: str,
    exclude_feature_cols: list[str] | tuple[str, ...] | None = None,
    max_missing_ratio: float | None = None,
) -> list[str]:
    """Select model feature columns."""

    excluded = {date_col, *DEFAULT_TARGET_EXCLUDES, *(exclude_feature_cols or [])}
    if feature_cols == "auto_numeric":
        selected: list[str] = []
        for column in df.columns:
            if column in excluded:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            missing_ratio = float(df[column].isna().mean())
            if max_missing_ratio is not None and missing_ratio > max_missing_ratio:
                continue
            selected.append(column)
    else:
        selected = list(feature_cols)

    missing = [column for column in selected if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if not selected:
        raise ValueError("No feature columns selected")
    return selected