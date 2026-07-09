"""Prepare the monthly corn dataset without news PCA features.

Source data follows the GitHub dual-head LSTM monthly table, then removes the
news PCA columns (`pca_*`) and normalizes the date fields for easier reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path("玉米预测/datasets/processed/corn_monthly_dual_stream_spike_github.csv")
DEFAULT_OUTPUT = Path("玉米预测/datasets/processed/corn_monthly_no_news_pca.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly corn dataset without news PCA columns.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_month(values: pd.Series) -> pd.Series:
    raw = values.astype(str)
    parsed = pd.to_datetime(raw, format="%y-%b", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.isna().any():
        bad = raw[parsed.isna()].head(3).tolist()
        raise ValueError(f"Could not parse month values: {bad}")
    return parsed.dt.strftime("%Y-%m")


def normalize_date(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise ValueError(f"Could not parse date values: {bad}")
    return parsed.dt.strftime("%Y-%m-%d")


def build_dataset(source: Path, output: Path) -> dict[str, object]:
    df = pd.read_csv(source)
    pca_cols = [col for col in df.columns if col.startswith("pca_")]
    if not pca_cols:
        raise ValueError("No pca_* columns found in source data.")

    out = df.drop(columns=pca_cols).copy()
    out["month"] = normalize_month(out["month"])
    for date_col in ("first_trade_date", "last_trade_date"):
        if date_col in out.columns:
            out[date_col] = normalize_date(out[date_col])

    missing_total = int(out.isna().sum().sum())
    if missing_total:
        missing_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Output still has {missing_total} missing values in columns: {missing_cols}")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    return {
        "source": str(source),
        "output": str(output),
        "rows": int(out.shape[0]),
        "columns": int(out.shape[1]),
        "dropped_pca_columns": len(pca_cols),
        "month_min": str(out["month"].iloc[0]),
        "month_max": str(out["month"].iloc[-1]),
        "missing_values": missing_total,
    }


def main() -> None:
    args = parse_args()
    summary = build_dataset(args.source, args.output)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
