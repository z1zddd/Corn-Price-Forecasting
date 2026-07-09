"""DataPipeline combines loader, splitter, and scaler into one contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.loader import load_and_window
from src.data.scaler import Standardizer
from src.data.splitter import time_split
from src.data.cv import assert_temporal_holdout


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Standardizer
    y_scaler: Standardizer
    meta_train: object
    meta_val: object
    meta_test: object
    feature_cols: list[str]


class DataPipeline:
    def __init__(self, config: dict):
        self.config = config

    def run(self) -> DataBundle:
        x, y, meta = load_and_window(self.config)
        split_dates = meta["target_date"] if "target_date" in meta else meta["date"]
        train_idx, val_idx, test_idx = time_split(
            split_dates,
            method=self.config.get("split_method", "fixed_date"),
            test_start=self.config.get("test_start", "2024-01-01"),
            train_ratio=float(self.config.get("train_ratio", 0.8)),
            val_ratio=float(self.config.get("val_ratio", 0.1)),
        )
        if self.config.get("smoke_train_tail_n"):
            train_idx = train_idx[-int(self.config["smoke_train_tail_n"]) :]
        if self.config.get("smoke_val_n"):
            val_idx = val_idx[: int(self.config["smoke_val_n"])]
        if self.config.get("smoke_test_n"):
            test_idx = test_idx[: int(self.config["smoke_test_n"])]
        return build_bundle(x, y, meta, train_idx, val_idx, test_idx, self.config)


def build_bundle(x, y, meta, train_idx, val_idx, test_idx, config: dict) -> DataBundle:
    assert_temporal_holdout(meta, train_idx, val_idx, test_idx)

    scaler = Standardizer()
    x_ntv = x.transpose(0, 2, 1)
    scaler.fit(x_ntv[train_idx], y[train_idx])
    if config.get("scale_x", True):
        x_out = scaler.transform_x(x_ntv)
    else:
        x_out = scaler.fill(x_ntv)
    x_scaled = x_out.transpose(0, 2, 1).astype("float32")

    if config.get("scale_y", True):
        y_out = scaler.transform_y(y).reshape(-1)
    else:
        y_out = y.astype("float32")

    feature_cols = meta.attrs.get("feature_cols", config.get("feature_cols", []))
    return DataBundle(
        X_train=x_scaled[train_idx],
        y_train=y_out[train_idx],
        X_val=x_scaled[val_idx],
        y_val=y_out[val_idx],
        X_test=x_scaled[test_idx],
        y_test=y_out[test_idx],
        scaler=scaler,
        y_scaler=scaler,
        meta_train=meta.iloc[train_idx].reset_index(drop=True),
        meta_val=meta.iloc[val_idx].reset_index(drop=True),
        meta_test=meta.iloc[test_idx].reset_index(drop=True),
        feature_cols=list(feature_cols),
    )
