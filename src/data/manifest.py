"""Data manifest helpers for reproducible runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def build_data_manifest(csv_path: str | Path, feature_cols: list[str], bundle, data_config: dict) -> dict:
    path = Path(csv_path)
    stat = path.stat() if path.exists() else None
    columns = pd.read_csv(path, nrows=0).columns.tolist() if path.exists() else []
    return {
        "csv_path": str(path),
        "file_size": None if stat is None else stat.st_size,
        "file_mtime": None if stat is None else stat.st_mtime,
        "columns_hash": hashlib.sha256(json.dumps(columns, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "feature_cols": feature_cols,
        "feature_cols_hash": hashlib.sha256(json.dumps(feature_cols, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "target_mode": data_config.get("target_mode"),
        "horizon": data_config.get("horizon"),
        "seq_len": data_config.get("seq_len"),
        "include_today": data_config.get("include_today"),
        "splits": {
            "train": split_range(bundle.meta_train),
            "val": split_range(bundle.meta_val),
            "test": split_range(bundle.meta_test),
        },
    }


def split_range(meta) -> dict:
    date_col = "target_date" if "target_date" in meta else "date"
    return {
        "rows": int(len(meta)),
        "date_min": str(pd.to_datetime(meta["date"]).min().date()) if len(meta) else None,
        "date_max": str(pd.to_datetime(meta["date"]).max().date()) if len(meta) else None,
        "target_date_min": str(pd.to_datetime(meta[date_col]).min().date()) if len(meta) else None,
        "target_date_max": str(pd.to_datetime(meta[date_col]).max().date()) if len(meta) else None,
    }

