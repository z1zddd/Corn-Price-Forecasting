#!/usr/bin/env python3
"""Build the leakage-aware monthly_v1 corn factor library and matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (
    ROOT
    / "corn_forecast"
    / "datasets"
    / "corn"
    / "processed"
    / "corn_monthly_core_v1.csv"
)
FACTOR_ROOT = ROOT / "corn_forecast" / "datasets" / "corn" / "factors"
LIBRARY_ROOT = FACTOR_ROOT / "library" / "monthly_v1"
MATRIX_PATH = FACTOR_ROOT / "matrix" / "corn_factors_monthly_v1.csv"
MANIFEST_PATH = FACTOR_ROOT / "monthly_v1_manifest.json"
SOURCE_VERSION = "corn_monthly_factors_v1"


FACTOR_SPECS = {
    "price_trend": {
        "group": "price",
        "inputs": ["dce_corn_close"],
        "publication_lag_months": 0,
        "availability_rule": "DCE month-t close is available after the month-t close.",
        "outputs": [
            {
                "factor_id": "price_momentum_1m",
                "expression": "close_t / close_t_minus_1 - 1",
                "minimum_history_months": 2,
            },
            {
                "factor_id": "price_momentum_3m",
                "expression": "close_t / close_t_minus_3 - 1",
                "minimum_history_months": 4,
            },
            {
                "factor_id": "price_ma_gap_6m",
                "expression": "close_t / mean(close_t_minus_5_to_t) - 1",
                "minimum_history_months": 6,
            },
        ],
    },
    "risk_volatility": {
        "group": "risk",
        "inputs": ["dce_corn_month_range_pct", "dce_corn_close"],
        "publication_lag_months": 0,
        "availability_rule": "DCE month-t OHLC is available after the month-t close.",
        "outputs": [
            {
                "factor_id": "price_range_1m",
                "expression": "(monthly_high - monthly_low) / monthly_close",
                "minimum_history_months": 1,
            },
            {
                "factor_id": "price_volatility_3m",
                "expression": "std(monthly_return_t_minus_2_to_t)",
                "minimum_history_months": 4,
            },
            {
                "factor_id": "volatility_ratio_3m_12m",
                "expression": "volatility_3m / volatility_12m",
                "minimum_history_months": 13,
            },
        ],
    },
    "market_activity": {
        "group": "market",
        "inputs": ["dce_corn_volume_sum", "dce_corn_open_interest_last"],
        "publication_lag_months": 0,
        "availability_rule": "DCE month-t volume and open interest are available after the month-t close.",
        "outputs": [
            {
                "factor_id": "volume_log_change_1m",
                "expression": "log1p(volume_t) - log1p(volume_t_minus_1)",
                "minimum_history_months": 2,
            },
            {
                "factor_id": "open_interest_log_change_1m",
                "expression": "log(open_interest_t) - log(open_interest_t_minus_1)",
                "minimum_history_months": 2,
            },
        ],
    },
    "basis_tightness": {
        "group": "basis",
        "inputs": ["corn_basis_rate_last"],
        "publication_lag_months": 1,
        "availability_rule": "Use month t-1 because the tracked spot/basis publication timestamp is undocumented.",
        "outputs": [
            {
                "factor_id": "basis_rate_level_lag1",
                "expression": "basis_rate_t_minus_1",
                "minimum_history_months": 2,
            },
            {
                "factor_id": "basis_rate_zscore_12m_lag1",
                "expression": "zscore(basis_rate_t_minus_1, trailing_12_months)",
                "minimum_history_months": 13,
            },
        ],
    },
    "term_structure": {
        "group": "term_structure",
        "inputs": [
            "corn_100ppi_nearby_futures_price_cny_t_last",
            "corn_100ppi_main_futures_price_cny_t_last",
        ],
        "publication_lag_months": 1,
        "availability_rule": "Use month t-1 because 100PPI publication timestamps are undocumented.",
        "outputs": [
            {
                "factor_id": "nearby_main_spread_ratio_lag1",
                "expression": "nearby_t_minus_1 / main_t_minus_1 - 1",
                "minimum_history_months": 2,
            },
            {
                "factor_id": "nearby_main_spread_change_1m_lag1",
                "expression": "spread_ratio_t_minus_1 - spread_ratio_t_minus_2",
                "minimum_history_months": 3,
            },
        ],
    },
    "processing_spread_proxy": {
        "group": "processing",
        "inputs": ["cs_c_spread_close_last", "dce_corn_close"],
        "publication_lag_months": 0,
        "availability_rule": "Both DCE settlement proxies are available after the month-t close.",
        "outputs": [
            {
                "factor_id": "starch_corn_spread_ratio",
                "expression": "(starch_close_t - corn_close_t) / corn_close_t",
                "minimum_history_months": 1,
            }
        ],
        "notes": [
            "This is a futures spread proxy, not a measured processing margin or processing demand series."
        ],
    },
    "cross_market_support": {
        "group": "external",
        "inputs": ["dce_corn_close", "cbot_corn_close_last"],
        "publication_lag_months": 0,
        "availability_rule": "Month-t exchange closes are available at the prediction cutoff.",
        "outputs": [
            {
                "factor_id": "cbot_corn_momentum_1m",
                "expression": "cbot_close_t / cbot_close_t_minus_1 - 1",
                "minimum_history_months": 2,
            },
            {
                "factor_id": "domestic_minus_cbot_momentum_1m",
                "expression": "dce_return_1m - cbot_return_1m",
                "minimum_history_months": 2,
            },
        ],
    },
    "realized_weather_anomaly": {
        "group": "weather",
        "inputs": ["ne_avg_precip_mm_mean", "ne_avg_t2m_c_mean"],
        "publication_lag_months": 0,
        "availability_rule": "Realized month-t weather is used only after month t ends.",
        "outputs": [
            {
                "factor_id": "precip_anomaly_same_month",
                "expression": "zscore(precip_t, prior_years_for_same_calendar_month)",
                "minimum_history_months": 37,
            },
            {
                "factor_id": "temperature_anomaly_same_month",
                "expression": "zscore(temperature_t, prior_years_for_same_calendar_month)",
                "minimum_history_months": 37,
            },
            {
                "factor_id": "hot_dry_weather_stress",
                "expression": "max(temperature_anomaly, 0) * max(-precip_anomaly, 0)",
                "minimum_history_months": 37,
            },
        ],
        "notes": [
            "Each baseline uses only prior observations from the same calendar month; full-sample climatology is forbidden."
        ],
    },
    "harvest_season_proxy": {
        "group": "harvest",
        "inputs": ["is_harvest_season_ne_cn_share"],
        "publication_lag_months": 0,
        "availability_rule": "Deterministic calendar share for month t.",
        "outputs": [
            {
                "factor_id": "harvest_season_share",
                "expression": "share_of_observations_marked_as_northeast_china_harvest_season",
                "minimum_history_months": 1,
            }
        ],
        "notes": ["This is a calendar proxy and is not observed harvest progress."],
    },
    "seasonal": {
        "group": "calendar",
        "inputs": ["month"],
        "publication_lag_months": 0,
        "availability_rule": "Calendar encoding is known before month t.",
        "outputs": [
            {
                "factor_id": "seasonal_sin",
                "expression": "sin(2 * pi * calendar_month / 12)",
                "minimum_history_months": 1,
            },
            {
                "factor_id": "seasonal_cos",
                "expression": "cos(2 * pi * calendar_month / 12)",
                "minimum_history_months": 1,
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=CORE_PATH)
    parser.add_argument("--factor-root", type=Path, default=FACTOR_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_same_month_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "calendar_month": dates.dt.month},
        index=values.index,
    )
    prior_mean = frame.groupby("calendar_month")["value"].transform(
        lambda series: series.shift(1).expanding(min_periods=3).mean()
    )
    prior_std = frame.groupby("calendar_month")["value"].transform(
        lambda series: series.shift(1).expanding(min_periods=3).std()
    )
    return (frame["value"] - prior_mean) / prior_std.replace(0.0, np.nan)


def build_factors(core: pd.DataFrame) -> pd.DataFrame:
    complete = core["is_complete_period"].astype(bool)
    dates = pd.to_datetime(core["month"], errors="raise")
    close = pd.to_numeric(core["dce_corn_close"], errors="coerce").where(complete)
    returns = close.pct_change(fill_method=None)

    factors = pd.DataFrame(index=core.index)
    factors["price_momentum_1m"] = returns
    factors["price_momentum_3m"] = close / close.shift(3) - 1.0
    factors["price_ma_gap_6m"] = close / close.rolling(6, min_periods=6).mean() - 1.0

    factors["price_range_1m"] = pd.to_numeric(
        core["dce_corn_month_range_pct"], errors="coerce"
    ).where(complete)
    volatility_3m = returns.rolling(3, min_periods=3).std()
    volatility_12m = returns.rolling(12, min_periods=12).std()
    factors["price_volatility_3m"] = volatility_3m
    factors["volatility_ratio_3m_12m"] = volatility_3m / volatility_12m.replace(0.0, np.nan)

    volume = pd.to_numeric(core["dce_corn_volume_sum"], errors="coerce").where(complete)
    open_interest = pd.to_numeric(
        core["dce_corn_open_interest_last"], errors="coerce"
    ).where(complete)
    factors["volume_log_change_1m"] = np.log1p(volume).diff()
    factors["open_interest_log_change_1m"] = np.log(open_interest.where(open_interest > 0)).diff()

    basis_lag1 = pd.to_numeric(core["corn_basis_rate_last"], errors="coerce").where(complete).shift(1)
    factors["basis_rate_level_lag1"] = basis_lag1
    basis_mean = basis_lag1.rolling(12, min_periods=12).mean()
    basis_std = basis_lag1.rolling(12, min_periods=12).std()
    factors["basis_rate_zscore_12m_lag1"] = (basis_lag1 - basis_mean) / basis_std.replace(0.0, np.nan)

    nearby = pd.to_numeric(
        core["corn_100ppi_nearby_futures_price_cny_t_last"], errors="coerce"
    ).where(complete)
    main = pd.to_numeric(
        core["corn_100ppi_main_futures_price_cny_t_last"], errors="coerce"
    ).where(complete)
    term_lag1 = (nearby / main.replace(0.0, np.nan) - 1.0).shift(1)
    factors["nearby_main_spread_ratio_lag1"] = term_lag1
    factors["nearby_main_spread_change_1m_lag1"] = term_lag1.diff()

    spread = pd.to_numeric(core["cs_c_spread_close_last"], errors="coerce").where(complete)
    factors["starch_corn_spread_ratio"] = spread / close.replace(0.0, np.nan)

    cbot_close = pd.to_numeric(core["cbot_corn_close_last"], errors="coerce").where(complete)
    cbot_return = cbot_close.pct_change(fill_method=None)
    factors["cbot_corn_momentum_1m"] = cbot_return
    factors["domestic_minus_cbot_momentum_1m"] = returns - cbot_return

    precipitation = pd.to_numeric(core["ne_avg_precip_mm_mean"], errors="coerce").where(complete)
    temperature = pd.to_numeric(core["ne_avg_t2m_c_mean"], errors="coerce").where(complete)
    precip_anomaly = prior_same_month_zscore(precipitation, dates)
    temperature_anomaly = prior_same_month_zscore(temperature, dates)
    factors["precip_anomaly_same_month"] = precip_anomaly
    factors["temperature_anomaly_same_month"] = temperature_anomaly
    factors["hot_dry_weather_stress"] = temperature_anomaly.clip(lower=0.0) * (
        -precip_anomaly
    ).clip(lower=0.0)

    factors["harvest_season_share"] = pd.to_numeric(
        core["is_harvest_season_ne_cn_share"], errors="coerce"
    ).where(complete)
    factors["seasonal_sin"] = np.sin(2.0 * np.pi * dates.dt.month / 12.0)
    factors["seasonal_cos"] = np.cos(2.0 * np.pi * dates.dt.month / 12.0)
    return factors


def output_columns() -> list[str]:
    return [
        output["factor_id"]
        for spec in FACTOR_SPECS.values()
        for output in spec["outputs"]
    ]


def write_definitions(library_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for family, spec in FACTOR_SPECS.items():
        directory = library_root / family
        directory.mkdir(parents=True, exist_ok=True)
        definition = {
            "schema_version": 1,
            "id": f"monthly_v1_{family}",
            "version": "monthly_v1",
            "status": "candidate",
            "instrument": "corn",
            "frequency": "monthly",
            "group": spec["group"],
            "source": {
                "dataset": "corn_forecast/datasets/corn/processed/corn_monthly_core_v1.csv",
                "source_version": "corn_monthly_v1",
                "inputs": spec["inputs"],
            },
            "prediction_contract": {
                "factor_period": "month_t",
                "target_period": "month_t_plus_1",
                "publication_lag_months": spec["publication_lag_months"],
                "availability_rule": spec["availability_rule"],
                "uses_future_target": False,
            },
            "missing_value_policy": {
                "preserve_insufficient_history": True,
                "backward_fill": False,
                "two_sided_interpolation": False,
            },
            "outputs": spec["outputs"],
        }
        if spec.get("notes"):
            definition["notes"] = spec["notes"]
        path = directory / "factor.yaml"
        path.write_text(
            json.dumps(definition, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths[family] = path
    return paths


def build_matrix(core: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(
        {
            "month": pd.to_datetime(core["month"]).dt.strftime("%Y-%m-%d"),
            "period_end": pd.to_datetime(core["last_trade_date"]).dt.strftime("%Y-%m-%d"),
            "is_complete_period": core["is_complete_period"].astype(bool),
        }
    )
    matrix["strict_backtest_eligible"] = matrix["is_complete_period"]
    matrix["available_factor_count"] = factors.notna().sum(axis=1).astype(int)
    matrix["candidate_factor_count"] = len(factors.columns)
    matrix["source_version"] = SOURCE_VERSION
    return pd.concat([matrix, factors], axis=1)


def write_values(
    core: pd.DataFrame,
    factors: pd.DataFrame,
    library_root: Path,
) -> dict[str, Path]:
    period_end = pd.to_datetime(core["last_trade_date"])
    complete = core["is_complete_period"].astype(bool)
    paths: dict[str, Path] = {}

    for family, spec in FACTOR_SPECS.items():
        lag = int(spec["publication_lag_months"])
        asof_date = period_end.shift(lag) if lag else period_end
        rows: list[pd.DataFrame] = []
        for output in spec["outputs"]:
            factor_id = output["factor_id"]
            value = factors[factor_id]
            quality = pd.Series("ok", index=core.index, dtype="object")
            quality.loc[value.isna()] = "insufficient_history"
            quality.loc[~complete] = "partial_period"
            rows.append(
                pd.DataFrame(
                    {
                        "period_end": period_end.dt.strftime("%Y-%m-%d"),
                        "period_key": pd.to_datetime(core["month"]).dt.strftime("%Y-%m"),
                        "frequency": "monthly",
                        "instrument": "corn",
                        "factor_id": factor_id,
                        "value": value,
                        "coverage": value.notna().astype(float),
                        "asof_date": asof_date.dt.strftime("%Y-%m-%d"),
                        "quality_flag": quality,
                        "source_version": SOURCE_VERSION,
                    }
                )
            )
        long_values = pd.concat(rows, ignore_index=True).sort_values(
            ["period_key", "factor_id"], kind="stable"
        )
        path = library_root / family / "values.csv"
        long_values.to_csv(path, index=False, encoding="utf-8", float_format="%.10g")
        paths[family] = path
    return paths


def validate(core: pd.DataFrame, factors: pd.DataFrame, matrix: pd.DataFrame) -> None:
    if core["month"].duplicated().any() or not pd.to_datetime(core["month"]).is_monotonic_increasing:
        raise AssertionError("Core months must be unique and sorted")
    expected = output_columns()
    if list(factors.columns) != expected:
        raise AssertionError("Generated factor order differs from the registered specification")
    forbidden_tokens = ("target", "next_month", "spike")
    if any(token in column.lower() for column in matrix.columns for token in forbidden_tokens):
        raise AssertionError("The factor matrix must not contain target-like columns")
    if len(matrix) != len(core) or not matrix["month"].equals(core["month"]):
        raise AssertionError("The factor matrix must preserve core_v1 month coverage")
    if matrix.loc[matrix["is_complete_period"], expected].isna().all(axis=1).any():
        raise AssertionError("A complete month cannot have every factor missing")
    constant = [
        column
        for column in expected
        if factors[column].dropna().nunique() <= 1
    ]
    if constant:
        raise AssertionError(f"Constant factors are forbidden: {constant}")
    for column in ("price_momentum_3m", "volatility_ratio_3m_12m", "precip_anomaly_same_month"):
        if factors[column].iloc[:3].notna().any():
            raise AssertionError(f"{column} must preserve early insufficient-history gaps")
    if "supply_pressure" in factors.columns:
        raise AssertionError("Supply pressure is disabled until observed supply data is added")


def main() -> None:
    args = parse_args()
    core_path = args.core.resolve()
    factor_root = args.factor_root.resolve()
    library_root = factor_root / "library" / "monthly_v1"
    matrix_path = factor_root / "matrix" / "corn_factors_monthly_v1.csv"
    manifest_path = factor_root / "monthly_v1_manifest.json"

    core = pd.read_csv(core_path)
    factors = build_factors(core)
    definitions = write_definitions(library_root)
    values = write_values(core, factors, library_root)
    matrix = build_matrix(core, factors)
    validate(core, factors, matrix)

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_path, index=False, encoding="utf-8", float_format="%.10g")
    manifest = {
        "factor_set": "monthly_v1",
        "status": "candidate",
        "frequency": "monthly",
        "prediction_contract": "Use factors from month t to predict month t+1.",
        "source": str(core_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(core_path),
        "matrix": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": sha256(matrix_path),
        "rows": int(len(matrix)),
        "candidate_factor_count": int(len(factors.columns)),
        "complete_periods": int(matrix["is_complete_period"].sum()),
        "incomplete_periods": matrix.loc[~matrix["is_complete_period"], "month"].tolist(),
        "factor_families": {
            family: [output["factor_id"] for output in spec["outputs"]]
            for family, spec in FACTOR_SPECS.items()
        },
        "definition_sha256": {family: sha256(path) for family, path in definitions.items()},
        "values_sha256": {family: sha256(path) for family, path in values.items()},
        "target_columns_included": False,
        "backward_fill_used": False,
        "two_sided_interpolation_used": False,
        "spot_basis_and_100ppi_lag_months": 1,
        "weather_baseline_policy": "Prior years from the same calendar month only.",
        "supply_pressure_enabled": False,
        "weekly_or_yearly_files_modified": False,
        "training_policy": "Fit scaling, winsorization, imputation, and feature selection inside each training fold.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {matrix_path} ({matrix.shape[0]} rows x {matrix.shape[1]} columns)")
    print(f"wrote {len(values)} factor-family value files under {library_root}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
