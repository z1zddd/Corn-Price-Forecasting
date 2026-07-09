#!/usr/bin/env python3
"""Prepare monthly corn spike data from the new daily China corn file.

The source file contains daily market features plus forward-looking helper
columns such as ``future_return_*`` and ``trend_*``.  Those columns are useful
as labels in their original daily task, but are excluded from monthly features
here to keep the rolling backtest leak-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAK_PREFIXES = ("future_return_", "trend_")
CORE_AGG = {
    "dce_corn_open": "first",
    "dce_corn_close": "last",
    "dce_corn_high": "max",
    "dce_corn_low": "min",
    "dce_corn_volume": "sum",
    "dce_corn_amount": "sum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--price-col", default="dce_corn_close")
    return parser.parse_args()


def safe_numeric_columns(df: pd.DataFrame, date_col: str) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        col
        for col in numeric
        if col != date_col and not any(col.startswith(prefix) for prefix in LEAK_PREFIXES)
    ]


def build_monthly(df: pd.DataFrame, date_col: str, price_col: str) -> tuple[pd.DataFrame, dict]:
    if date_col not in df.columns:
        raise ValueError(f"date column not found: {date_col}")
    if price_col not in df.columns:
        raise ValueError(f"price column not found: {price_col}")

    raw_columns = list(df.columns)
    leak_columns = [col for col in raw_columns if any(col.startswith(prefix) for prefix in LEAK_PREFIXES)]

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    work["month"] = work[date_col].dt.to_period("M").astype(str)

    safe_numeric = safe_numeric_columns(work, date_col)
    grouped = work.groupby("month", sort=True)

    monthly_parts: dict[str, pd.Series] = {
        "month_start": grouped[date_col].min(),
        "month_end": grouped[date_col].max(),
        "daily_rows": grouped.size(),
    }

    for col, agg in CORE_AGG.items():
        if col in safe_numeric:
            monthly_parts[col] = grouped[col].agg(agg)

    for col in safe_numeric:
        if col in CORE_AGG:
            continue
        monthly_parts[f"{col}_last"] = grouped[col].last()
        monthly_parts[f"{col}_mean"] = grouped[col].mean()

    monthly = pd.DataFrame(monthly_parts).reset_index()
    monthly[price_col] = monthly[price_col].astype(float)
    monthly["corn_return_1m"] = monthly[price_col].pct_change()
    monthly["corn_return_2m"] = monthly[price_col].pct_change(2)
    monthly["corn_ma3_monthly"] = monthly[price_col].rolling(3, min_periods=1).mean()
    monthly["corn_ma6_monthly"] = monthly[price_col].rolling(6, min_periods=1).mean()
    monthly["corn_ma12_monthly"] = monthly[price_col].rolling(12, min_periods=1).mean()
    monthly["corn_vol3_monthly"] = monthly["corn_return_1m"].rolling(3, min_periods=2).std()
    monthly["corn_vol6_monthly"] = monthly["corn_return_1m"].rolling(6, min_periods=2).std()
    monthly["next_month_close"] = monthly[price_col].shift(-1)
    monthly["next_month_return"] = monthly["next_month_close"] / monthly[price_col] - 1.0
    monthly["spike"] = (monthly["next_month_return"] > 0.0).astype("Int64")

    monthly = monthly.loc[monthly["next_month_close"].notna()].copy()
    monthly["spike"] = monthly["spike"].astype(int)

    audit = {
        "source_rows": int(len(df)),
        "parsed_rows": int(len(work)),
        "source_columns": int(len(raw_columns)),
        "leak_columns_excluded": leak_columns,
        "safe_numeric_source_columns": int(len(safe_numeric)),
        "monthly_rows": int(len(monthly)),
        "monthly_columns": int(len(monthly.columns)),
        "date_min": str(work[date_col].min().date()),
        "date_max": str(work[date_col].max().date()),
        "month_min": str(monthly["month"].min()),
        "month_max": str(monthly["month"].max()),
        "target_definition": "spike = next monthly close return > 0",
        "missing_cells": int(monthly.isna().sum().sum()),
        "missing_columns_top20": monthly.isna().sum().sort_values(ascending=False).head(20).astype(int).to_dict(),
    }
    return monthly, audit


def main() -> None:
    args = parse_args()
    daily_path = Path(args.daily_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    audit_json = Path(args.audit_json).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(daily_path)
    monthly, audit = build_monthly(df, args.date_col, args.price_col)
    monthly.to_csv(out_csv, index=False)
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
