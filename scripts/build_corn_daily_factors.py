#!/usr/bin/env python3
"""Build the leakage-aware daily_v1 corn factor matrix and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = (
    ROOT
    / "corn_forecast"
    / "datasets"
    / "corn"
    / "raw"
    / "玉米价格原始数据.csv"
)
FACTOR_ROOT = ROOT / "corn_forecast" / "datasets" / "corn" / "factors"
MATRIX_PATH = FACTOR_ROOT / "matrix" / "corn_factors_daily_v1.csv"
DEFINITION_PATH = FACTOR_ROOT / "library" / "daily_v1" / "factor_set.yaml"
MANIFEST_PATH = FACTOR_ROOT / "daily_v1_manifest.json"
SOURCE_VERSION = "corn_daily_factors_v1"


FACTOR_FAMILIES = {
    "price_trend": {
        "group": "price",
        "availability": "DCE day-t close is available after the day-t close.",
        "inputs": ["dce_corn_close"],
        "outputs": {
            "price_momentum_1d": "close_t / close_t_minus_1 - 1",
            "price_momentum_5d": "close_t / close_t_minus_5 - 1",
            "price_momentum_20d": "close_t / close_t_minus_20 - 1",
            "price_ma_gap_5d": "close_t / mean(close_t_minus_4_to_t) - 1",
            "price_ma_gap_20d": "close_t / mean(close_t_minus_19_to_t) - 1",
            "price_ma_gap_60d": "close_t / mean(close_t_minus_59_to_t) - 1",
        },
    },
    "risk_volatility": {
        "group": "risk",
        "availability": "DCE day-t OHLC is available after the day-t close.",
        "inputs": ["dce_corn_high", "dce_corn_low", "dce_corn_close"],
        "outputs": {
            "price_range_1d": "(high_t - low_t) / close_t",
            "price_volatility_5d": "std(daily_close_return_t_minus_4_to_t)",
            "price_volatility_20d": "std(daily_close_return_t_minus_19_to_t)",
            "volatility_ratio_5d_20d": "volatility_5d / volatility_20d",
        },
    },
    "market_activity": {
        "group": "market",
        "availability": "DCE day-t volume and open interest are available after the close.",
        "inputs": ["dce_corn_volume", "dce_corn_open_interest"],
        "outputs": {
            "volume_log_change_1d": "log1p(volume_t) - log1p(volume_t_minus_1)",
            "open_interest_log_change_1d": "log(open_interest_t) - log(open_interest_t_minus_1)",
        },
    },
    "basis_tightness": {
        "group": "basis",
        "availability": "Publication time is undocumented; use the prior DCE row.",
        "publication_lag_dce_rows": 1,
        "inputs": ["corn_basis_rate"],
        "outputs": {
            "basis_rate_level_lag1d": "basis_rate_t_minus_1",
            "basis_rate_change_1d_lag1d": "basis_rate_t_minus_1 - basis_rate_t_minus_2",
            "basis_rate_zscore_20d_lag1d": "zscore(basis_rate_t_minus_1, trailing_20_rows)",
        },
    },
    "term_structure": {
        "group": "term_structure",
        "availability": "100PPI publication time is undocumented; use the prior DCE row.",
        "publication_lag_dce_rows": 1,
        "inputs": [
            "corn_100ppi_nearby_futures_price_cny_t",
            "corn_100ppi_main_futures_price_cny_t",
        ],
        "outputs": {
            "nearby_main_spread_ratio_lag1d": "nearby_t_minus_1 / main_t_minus_1 - 1",
            "nearby_main_spread_change_1d_lag1d": "spread_ratio_t_minus_1 - spread_ratio_t_minus_2",
        },
    },
    "processing_spread_proxy": {
        "group": "processing",
        "availability": "Both DCE prices are available after the day-t close.",
        "inputs": ["cs_c_spread_close", "dce_corn_close"],
        "outputs": {
            "starch_corn_spread_ratio": "starch_minus_corn_spread_t / corn_close_t",
            "starch_corn_spread_change_5d": "spread_ratio_t - spread_ratio_t_minus_5",
        },
        "notes": ["This is a futures spread proxy, not a measured processing margin."],
    },
    "cross_market_support": {
        "group": "external",
        "availability": "The US close follows the DCE close; use the prior DCE row.",
        "publication_lag_dce_rows": 1,
        "inputs": ["cbot_corn_close", "cbot_wheat_corn_ratio", "dce_corn_close"],
        "outputs": {
            "cbot_corn_momentum_1d_lag1d": "cbot_close_t_minus_1 / cbot_close_t_minus_2 - 1",
            "cbot_corn_momentum_5d_lag1d": "cbot_close_t_minus_1 / cbot_close_t_minus_6 - 1",
            "domestic_minus_cbot_momentum_1d_lag1d": "dce_return_t - lagged_cbot_return_t",
            "cbot_wheat_corn_ratio_change_1d_lag1d": "ratio_t_minus_1 / ratio_t_minus_2 - 1",
        },
    },
    "realized_weather": {
        "group": "weather",
        "availability": "Realized day-t weather is used only after day t ends.",
        "inputs": ["ne_avg_precip_mm", "ne_avg_t2m_c"],
        "outputs": {
            "precipitation_sum_5d": "sum(precipitation_t_minus_4_to_t)",
            "precipitation_week_vs_month": "mean(precipitation_5d) / mean(precipitation_20d) - 1",
            "temperature_deviation_20d": "temperature_t - mean(temperature_t_minus_19_to_t)",
        },
    },
    "calendar": {
        "group": "calendar",
        "availability": "Calendar fields are known in advance.",
        "inputs": ["date"],
        "outputs": {
            "day_of_year_sin": "sin(2*pi*day_of_year/365.25)",
            "day_of_year_cos": "cos(2*pi*day_of_year/365.25)",
            "day_of_week_sin": "sin(2*pi*day_of_week/5)",
            "day_of_week_cos": "cos(2*pi*day_of_week/5)",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--factor-root", type=Path, default=FACTOR_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_columns() -> list[str]:
    return [
        factor_id
        for family in FACTOR_FAMILIES.values()
        for factor_id in family["outputs"]
    ]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std()
    return (values - mean) / std.replace(0.0, np.nan)


def build_factors(raw: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(raw["date"], errors="raise")
    close = numeric(raw, "dce_corn_close")
    returns = close.pct_change(fill_method=None)
    factors = pd.DataFrame(index=raw.index)

    factors["price_momentum_1d"] = returns
    factors["price_momentum_5d"] = close / close.shift(5) - 1.0
    factors["price_momentum_20d"] = close / close.shift(20) - 1.0
    factors["price_ma_gap_5d"] = close / close.rolling(5, min_periods=5).mean() - 1.0
    factors["price_ma_gap_20d"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    factors["price_ma_gap_60d"] = close / close.rolling(60, min_periods=60).mean() - 1.0

    factors["price_range_1d"] = (
        numeric(raw, "dce_corn_high") - numeric(raw, "dce_corn_low")
    ) / close.replace(0.0, np.nan)
    volatility_5d = returns.rolling(5, min_periods=5).std()
    volatility_20d = returns.rolling(20, min_periods=20).std()
    factors["price_volatility_5d"] = volatility_5d
    factors["price_volatility_20d"] = volatility_20d
    factors["volatility_ratio_5d_20d"] = volatility_5d / volatility_20d.replace(0.0, np.nan)

    volume = numeric(raw, "dce_corn_volume")
    open_interest = numeric(raw, "dce_corn_open_interest")
    factors["volume_log_change_1d"] = np.log1p(volume).diff()
    factors["open_interest_log_change_1d"] = np.log(
        open_interest.where(open_interest > 0)
    ).diff()

    basis_lag1 = numeric(raw, "corn_basis_rate").shift(1)
    factors["basis_rate_level_lag1d"] = basis_lag1
    factors["basis_rate_change_1d_lag1d"] = basis_lag1.diff()
    factors["basis_rate_zscore_20d_lag1d"] = rolling_zscore(basis_lag1, 20)

    nearby = numeric(raw, "corn_100ppi_nearby_futures_price_cny_t")
    main = numeric(raw, "corn_100ppi_main_futures_price_cny_t")
    term_lag1 = (nearby / main.replace(0.0, np.nan) - 1.0).shift(1)
    factors["nearby_main_spread_ratio_lag1d"] = term_lag1
    factors["nearby_main_spread_change_1d_lag1d"] = term_lag1.diff()

    spread_ratio = numeric(raw, "cs_c_spread_close") / close.replace(0.0, np.nan)
    factors["starch_corn_spread_ratio"] = spread_ratio
    factors["starch_corn_spread_change_5d"] = spread_ratio.diff(5)

    cbot_lag1 = numeric(raw, "cbot_corn_close").shift(1)
    cbot_return_lag1 = cbot_lag1.pct_change(fill_method=None)
    factors["cbot_corn_momentum_1d_lag1d"] = cbot_return_lag1
    factors["cbot_corn_momentum_5d_lag1d"] = cbot_lag1 / cbot_lag1.shift(5) - 1.0
    factors["domestic_minus_cbot_momentum_1d_lag1d"] = returns - cbot_return_lag1
    wheat_corn_lag1 = numeric(raw, "cbot_wheat_corn_ratio").shift(1)
    factors["cbot_wheat_corn_ratio_change_1d_lag1d"] = wheat_corn_lag1.pct_change(
        fill_method=None
    )

    precipitation = numeric(raw, "ne_avg_precip_mm")
    temperature = numeric(raw, "ne_avg_t2m_c")
    precipitation_mean_5d = precipitation.rolling(5, min_periods=5).mean()
    precipitation_mean_20d = precipitation.rolling(20, min_periods=20).mean()
    factors["precipitation_sum_5d"] = precipitation.rolling(5, min_periods=5).sum()
    factors["precipitation_week_vs_month"] = (
        precipitation_mean_5d / precipitation_mean_20d.replace(0.0, np.nan) - 1.0
    )
    factors["temperature_deviation_20d"] = temperature - temperature.rolling(
        20, min_periods=20
    ).mean()

    day_of_year = dates.dt.dayofyear.astype(float)
    day_of_week = dates.dt.dayofweek.astype(float)
    factors["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    factors["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    factors["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 5.0)
    factors["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 5.0)
    return factors.replace([np.inf, -np.inf], np.nan)


def build_matrix(raw: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(raw["date"], errors="raise")
    matrix = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "factor_cutoff": "end_of_day",
            "strict_backtest_eligible": factors.notna().all(axis=1),
            "is_latest_observation": False,
            "available_factor_count": factors.notna().sum(axis=1).astype(int),
            "candidate_factor_count": len(factors.columns),
            "source_version": SOURCE_VERSION,
        }
    )
    matrix.loc[matrix.index[-1], "is_latest_observation"] = True
    return pd.concat([matrix, factors], axis=1)


def validate(raw: pd.DataFrame, factors: pd.DataFrame, matrix: pd.DataFrame) -> None:
    dates = pd.to_datetime(raw["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise AssertionError("Raw daily dates must be unique and sorted")
    if list(factors.columns) != output_columns():
        raise AssertionError("Generated factor order differs from the factor definition")
    forbidden = ("target", "next_day", "spike")
    if any(token in column.lower() for column in matrix for token in forbidden):
        raise AssertionError("The daily factor matrix must not contain target-like columns")
    if len(matrix) != len(raw) or not matrix["date"].equals(dates.dt.strftime("%Y-%m-%d")):
        raise AssertionError("The factor matrix must preserve raw daily coverage")
    constant = [column for column in factors if factors[column].dropna().nunique() <= 1]
    if constant:
        raise AssertionError(f"Constant factors are forbidden: {constant}")
    if factors["price_ma_gap_60d"].iloc[:59].notna().any():
        raise AssertionError("The 60-day moving-average factor must preserve its warm-up gap")
    if factors["price_volatility_20d"].iloc[:20].notna().any():
        raise AssertionError("The 20-return volatility factor must preserve its warm-up gap")
    if matrix["is_latest_observation"].sum() != 1 or not matrix.iloc[-1]["is_latest_observation"]:
        raise AssertionError("Exactly the final row must be marked as the latest observation")


def write_definition(path: Path) -> None:
    definition = {
        "schema_version": 1,
        "id": "corn_daily_v1",
        "version": "daily_v1",
        "status": "candidate",
        "instrument": "corn",
        "frequency": "daily_dce_trading_rows",
        "source": "corn_forecast/datasets/corn/raw/玉米价格原始数据.csv",
        "prediction_contract": {
            "factor_row": "DCE trading day t",
            "target_row": "next actual DCE trading day after t",
            "cutoff": "after day t ends",
            "uses_future_target": False,
        },
        "missing_value_policy": {
            "preserve_rolling_warmup": True,
            "backward_fill": False,
            "two_sided_interpolation": False,
            "training_imputation": "fit inside each training fold only",
        },
        "families": FACTOR_FAMILIES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_path = args.raw.resolve()
    factor_root = args.factor_root.resolve()
    matrix_path = factor_root / "matrix" / "corn_factors_daily_v1.csv"
    definition_path = factor_root / "library" / "daily_v1" / "factor_set.yaml"
    manifest_path = factor_root / "daily_v1_manifest.json"

    raw = pd.read_csv(raw_path).reset_index(drop=True)
    factors = build_factors(raw)
    matrix = build_matrix(raw, factors)
    validate(raw, factors, matrix)

    write_definition(definition_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_path, index=False, encoding="utf-8", float_format="%.8g")

    dates = pd.to_datetime(raw["date"], errors="raise")
    manifest = {
        "factor_set": "daily_v1",
        "status": "candidate",
        "frequency": "daily_dce_trading_rows",
        "prediction_contract": "Use factors available after DCE row t ends to predict the next actual DCE trading row.",
        "source": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(raw_path),
        "source_rows": int(len(raw)),
        "source_start_date": dates.min().strftime("%Y-%m-%d"),
        "source_end_date": dates.max().strftime("%Y-%m-%d"),
        "matrix": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": sha256(matrix_path),
        "definition": str(definition_path.relative_to(ROOT)).replace("\\", "/"),
        "definition_sha256": sha256(definition_path),
        "rows": int(len(matrix)),
        "columns": int(len(matrix.columns)),
        "candidate_factor_count": len(output_columns()),
        "factor_ids": output_columns(),
        "missing_value_count": {column: int(factors[column].isna().sum()) for column in factors},
        "strict_backtest_eligible_rows": int(matrix["strict_backtest_eligible"].sum()),
        "target_columns_included": False,
        "backward_fill_used": False,
        "two_sided_interpolation_used": False,
        "spot_basis_100ppi_lag_dce_rows": 1,
        "cbot_lag_dce_rows": 1,
        "weekly_monthly_yearly_files_modified": False,
        "training_policy": "Fit scaling, winsorization, imputation, and feature selection inside each training fold.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {matrix_path} ({matrix.shape[0]} rows x {matrix.shape[1]} columns)")
    print(f"wrote {definition_path} ({len(output_columns())} candidate factors)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
