from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class SupervisedSamples:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class FixedSplit:
    train_idx: np.ndarray
    validation_idx: np.ndarray
    refit_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class ExpandingOrigin:
    train_idx: np.ndarray
    prediction_idx: int
    prediction_anchor_date: pd.Timestamp


def load_daily_data(
    path: str | Path,
    date_col: str = "date",
    target_col: str = "dce_corn_close",
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {date_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.sort_values(date_col).reset_index(drop=True)
    if frame[date_col].duplicated().any():
        raise ValueError("Duplicate dates are not allowed")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="raise")
    if frame[target_col].isna().any():
        raise ValueError("Target column contains missing values")
    for column in frame.columns.difference([date_col]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_supervised_samples(
    frame: pd.DataFrame,
    horizon: int,
    lookback: int,
    date_col: str = "date",
    target_col: str = "dce_corn_close",
    external_lag: int = 1,
) -> SupervisedSamples:
    if horizon <= 0 or lookback <= 0:
        raise ValueError("horizon and lookback must be positive")
    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col], errors="raise")
    ordered = ordered.sort_values(date_col).reset_index(drop=True)
    if ordered[date_col].duplicated().any():
        raise ValueError("Duplicate dates are not allowed")
    if target_col not in ordered:
        raise ValueError(f"Target column not found: {target_col}")

    feature_columns = [c for c in ordered.columns if c != date_col]
    external_columns = [c for c in feature_columns if c != target_col]
    features = ordered[feature_columns].apply(pd.to_numeric, errors="coerce")
    if external_columns and external_lag:
        features.loc[:, external_columns] = features[external_columns].shift(external_lag)

    flattened_columns = [
        f"{column}__lag{lag}"
        for lag in range(lookback - 1, -1, -1)
        for column in feature_columns
    ]
    rows: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, object]] = []
    last_anchor = len(ordered) - horizon
    for anchor in range(lookback - 1, last_anchor):
        window = features.iloc[anchor - lookback + 1 : anchor + 1]
        rows.append(window.to_numpy(dtype=float).reshape(-1))
        target_position = anchor + horizon
        targets.append(float(ordered.iloc[target_position][target_col]))
        metadata.append(
            {
                "sample_position": anchor,
                "anchor_date": ordered.iloc[anchor][date_col],
                "target_date": ordered.iloc[target_position][date_col],
                "close_t": float(ordered.iloc[anchor][target_col]),
            }
        )

    if not rows:
        raise ValueError("No eligible samples for the selected horizon/lookback")
    return SupervisedSamples(
        X=pd.DataFrame(rows, columns=flattened_columns),
        y=pd.Series(targets, name="actual_dce_corn_close", dtype=float),
        metadata=pd.DataFrame(metadata),
    )


class FoldPreprocessor:
    def __init__(
        self,
        max_missing_rate: float = 0.5,
        add_missing_indicators: bool = True,
        preserve_lag_groups: bool = False,
    ) -> None:
        if not 0 <= max_missing_rate < 1:
            raise ValueError("max_missing_rate must be in [0, 1)")
        self.max_missing_rate = max_missing_rate
        self.add_missing_indicators = bool(add_missing_indicators)
        self.preserve_lag_groups = bool(preserve_lag_groups)
        self.selected_columns: list[str] = []
        self.imputer: SimpleImputer | None = None

    def fit(self, X_train: pd.DataFrame) -> "FoldPreprocessor":
        missing_rate = X_train.isna().mean()
        eligible_columns = [
            column
            for column in X_train.columns
            if missing_rate[column] <= self.max_missing_rate
            and X_train[column].notna().any()
        ]
        if self.preserve_lag_groups:
            groups: dict[str, list[str]] = {}
            for column in X_train.columns:
                base_name = column.rsplit("__lag", maxsplit=1)[0]
                groups.setdefault(base_name, []).append(column)
            eligible = set(eligible_columns)
            retained_groups = {
                base_name
                for base_name, columns in groups.items()
                if all(column in eligible for column in columns)
            }
            self.selected_columns = [
                column
                for column in X_train.columns
                if column.rsplit("__lag", maxsplit=1)[0] in retained_groups
            ]
        else:
            self.selected_columns = eligible_columns
        if not self.selected_columns:
            raise ValueError("No features remain after training-fold missingness filtering")
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=self.add_missing_indicators
        )
        self.imputer.fit(X_train[self.selected_columns])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        transformed = np.asarray(
            self.imputer.transform(X[self.selected_columns]), dtype=np.float32
        )
        if not np.isfinite(transformed).all():
            raise ValueError("Preprocessing produced non-finite values")
        return transformed

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


def make_fixed_split(
    samples: SupervisedSamples,
    ratios: Sequence[float],
    embargo: int,
) -> FixedSplit:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain train/validation/test values summing to 1")
    n_samples = len(samples.X)
    train_end = int(np.floor(n_samples * ratios[0]))
    validation_end = int(np.floor(n_samples * (ratios[0] + ratios[1])))
    validation_start = min(train_end + embargo, validation_end)
    test_start = min(validation_end + embargo, n_samples)
    validation_idx = np.arange(validation_start, validation_end, dtype=int)
    test_idx = np.arange(test_start, n_samples, dtype=int)
    if validation_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Embargo leaves an empty validation or test segment")

    first_validation_anchor = samples.metadata.iloc[validation_idx[0]]["anchor_date"]
    first_test_anchor = samples.metadata.iloc[test_idx[0]]["anchor_date"]
    train_candidates = np.arange(0, train_end, dtype=int)
    refit_candidates = np.arange(0, validation_end, dtype=int)
    train_idx = train_candidates[
        (
            samples.metadata.iloc[train_candidates]["target_date"]
            <= first_validation_anchor
        ).to_numpy()
    ]
    refit_idx = refit_candidates[
        (
            samples.metadata.iloc[refit_candidates]["target_date"]
            <= first_test_anchor
        ).to_numpy()
    ]
    if train_idx.size == 0 or refit_idx.size == 0:
        raise ValueError("Purging leaves no training samples")
    return FixedSplit(train_idx, validation_idx, refit_idx, test_idx)


def assert_no_temporal_leakage(
    train_metadata: pd.DataFrame, prediction_anchor_date: pd.Timestamp
) -> None:
    if train_metadata.empty:
        raise AssertionError("Training metadata is empty")
    max_target_date = pd.to_datetime(train_metadata["target_date"]).max()
    if max_target_date > pd.Timestamp(prediction_anchor_date):
        raise AssertionError(
            f"Temporal leakage: train target {max_target_date} exceeds "
            f"prediction anchor {prediction_anchor_date}"
        )


def iter_expanding_origins(
    samples: SupervisedSamples, test_idx: Sequence[int]
) -> Iterator[ExpandingOrigin]:
    target_dates = pd.to_datetime(samples.metadata["target_date"])
    anchor_dates = pd.to_datetime(samples.metadata["anchor_date"])
    for prediction_idx in test_idx:
        prediction_idx = int(prediction_idx)
        prediction_anchor = anchor_dates.iloc[prediction_idx]
        eligible = np.flatnonzero(
            (target_dates <= prediction_anchor).to_numpy()
            & (anchor_dates < prediction_anchor).to_numpy()
        )
        assert_no_temporal_leakage(samples.metadata.iloc[eligible], prediction_anchor)
        yield ExpandingOrigin(eligible, prediction_idx, prediction_anchor)
