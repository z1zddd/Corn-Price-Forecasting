#!/usr/bin/env python3
"""Build daily market-chain factors from external normalized quote exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_ROOT = ROOT / "local_data" / "corn_market"
DEFAULT_RAW_QUOTES = DEFAULT_EXTERNAL_ROOT / "raw_quotes.csv"
DEFAULT_NORMALIZED = DEFAULT_EXTERNAL_ROOT / "normalized_prices.csv"
DCE_RAW = ROOT / "corn_forecast" / "datasets" / "corn" / "raw" / "玉米价格原始数据.csv"
FACTOR_ROOT = ROOT / "corn_forecast" / "datasets" / "corn" / "factors"
SOURCE_VERSION = "corn_daily_market_factors_v1"
RELEASE_CUTOFF = "15:00"

COL = {
    "record_id": "记录ID",
    "trainable": "是否可训练",
    "status": "状态",
    "product_code": "产品编码",
    "price": "价格",
    "unit": "单位",
    "region": "地区",
    "confidence": "置信度",
    "source": "来源名称",
    "release_time": "发布时间",
    "release_date": "发布日期",
    "authorization": "是否授权来源",
}

BYPRODUCT_CODES = ["corn_husk", "germ", "protein_powder"]
PROCESSING_CODES = [
    "corn_glucose",
    "corn_fructose",
    "glucose_syrup",
    "maltodextrin",
    "maltose_syrup",
    "crystalline_fructose",
]

FACTOR_FAMILIES = {
    "spot_price": {
        "group": "spot",
        "inputs": ["corn", "dce_corn_close"],
        "outputs": {
            "corn_spot_momentum_5d": "corn_spot_median_t / corn_spot_median_t_minus_5 - 1",
            "corn_spot_momentum_20d": "corn_spot_median_t / corn_spot_median_t_minus_20 - 1",
            "corn_spot_dce_basis": "corn_spot_median_t / dce_corn_close_t - 1",
            "corn_regional_dispersion": "IQR(corn_quotes_t) / median(corn_quotes_t), minimum_three_quotes",
        },
    },
    "quote_quality": {
        "group": "quality",
        "inputs": ["corn_quote_rows", "corn_quote_sources", "confidence"],
        "outputs": {
            "corn_quote_count_log": "log1p(valid_corn_quote_count_t)",
            "corn_source_count_log": "log1p(unique_corn_source_count_t)",
            "corn_confidence_mean": "mean(valid_corn_quote_confidence_t)",
        },
    },
    "processing_spread": {
        "group": "processing",
        "inputs": ["corn", "corn_starch"],
        "outputs": {
            "starch_corn_spread_ratio": "corn_starch_median_t / corn_spot_median_t - 1",
            "starch_corn_spread_change_5d": "spread_ratio_t - spread_ratio_t_minus_5",
        },
    },
    "byproduct_pressure": {
        "group": "processing",
        "inputs": BYPRODUCT_CODES,
        "outputs": {
            "byproduct_momentum_5d": "mean(five_row_return_by_product_t), minimum_two_products",
            "byproduct_momentum_20d": "mean(twenty_row_return_by_product_t), minimum_two_products",
        },
    },
    "processing_chain": {
        "group": "demand",
        "inputs": PROCESSING_CODES,
        "outputs": {
            "processing_chain_momentum_20d": "mean(twenty_row_return_by_product_t), minimum_three_products",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-quotes", type=Path, default=DEFAULT_RAW_QUOTES)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--dce-raw", type=Path, default=DCE_RAW)
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


def validate_source_pair(raw_quotes: pd.DataFrame, normalized: pd.DataFrame) -> None:
    record_id = COL["record_id"]
    if raw_quotes[record_id].duplicated().any() or normalized[record_id].duplicated().any():
        raise AssertionError("Source record IDs must be unique in both exports")
    if set(raw_quotes[record_id]) != set(normalized[record_id]):
        raise AssertionError("Raw and normalized record IDs must match exactly")


def map_available_dates(
    frame: pd.DataFrame,
    dce_calendar: pd.Series,
    cutoff: str = RELEASE_CUTOFF,
) -> pd.Series:
    release_date = pd.to_datetime(frame[COL["release_date"]], errors="raise").dt.normalize()
    release_clock = frame[COL["release_time"]].astype(str).str.extract(r"(\d{2}:\d{2})")[0]
    cutoff_minutes = int(cutoff[:2]) * 60 + int(cutoff[3:])
    release_minutes = (
        pd.to_numeric(release_clock.str[:2], errors="coerce") * 60
        + pd.to_numeric(release_clock.str[3:], errors="coerce")
    )
    after_cutoff_or_unknown = release_minutes.isna() | (release_minutes > cutoff_minutes)
    eligible_from = release_date + pd.to_timedelta(after_cutoff_or_unknown.astype(int), unit="D")

    calendar = pd.Series(pd.to_datetime(dce_calendar, errors="raise")).drop_duplicates().sort_values()
    positions = np.searchsorted(calendar.to_numpy(), eligible_from.to_numpy(), side="left")
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    result.loc[valid] = calendar.iloc[positions[valid]].to_numpy()
    return result


def clean_normalized(normalized: pd.DataFrame) -> pd.DataFrame:
    valid_status = {"normalized", "price_parsed", "accepted"}
    clean = normalized.loc[
        normalized[COL["trainable"]].eq("yes")
        & normalized[COL["status"]].isin(valid_status)
        & normalized[COL["unit"]].eq("元/吨")
    ].copy()
    clean[COL["price"]] = pd.to_numeric(clean[COL["price"]], errors="coerce")
    clean[COL["confidence"]] = pd.to_numeric(clean[COL["confidence"]], errors="coerce")
    clean = clean.loc[clean[COL["price"]] > 0].copy()
    return clean


def build_daily_panel(normalized: pd.DataFrame, dce: pd.DataFrame) -> pd.DataFrame:
    dce_dates = pd.to_datetime(dce["date"], errors="raise")
    if dce_dates.duplicated().any() or not dce_dates.is_monotonic_increasing:
        raise AssertionError("DCE dates must be unique and sorted")

    clean = clean_normalized(normalized)
    clean["available_date"] = map_available_dates(clean, dce_dates)
    clean = clean.dropna(subset=["available_date"])

    panel = pd.DataFrame(
        {
            "date": dce_dates.reset_index(drop=True),
            "dce_corn_close": pd.to_numeric(dce["dce_corn_close"], errors="coerce").reset_index(drop=True),
        }
    )
    panel = panel.set_index("date")

    product_prices = clean.pivot_table(
        index="available_date",
        columns=COL["product_code"],
        values=COL["price"],
        aggfunc="median",
    )
    for product_code in sorted(set(["corn", "corn_starch"] + BYPRODUCT_CODES + PROCESSING_CODES)):
        panel[f"product__{product_code}"] = product_prices.get(product_code)

    corn = clean.loc[clean[COL["product_code"]].eq("corn")]
    corn_group = corn.groupby("available_date")
    panel["corn_quote_q25"] = corn_group[COL["price"]].quantile(0.25)
    panel["corn_quote_q75"] = corn_group[COL["price"]].quantile(0.75)
    panel["corn_quote_count"] = corn_group[COL["record_id"]].size()
    panel["corn_source_count"] = corn_group[COL["source"]].nunique()
    panel["corn_confidence_mean"] = corn_group[COL["confidence"]].mean()
    panel["corn_release_count_all_products"] = clean.groupby("available_date")[COL["record_id"]].size()
    return panel.reset_index()


def mean_with_minimum(frame: pd.DataFrame, minimum: int) -> pd.Series:
    return frame.mean(axis=1, skipna=True).where(frame.notna().sum(axis=1) >= minimum)


def build_factors(panel: pd.DataFrame) -> pd.DataFrame:
    corn = panel["product__corn"]
    starch = panel["product__corn_starch"]
    factors = pd.DataFrame(index=panel.index)

    factors["corn_spot_momentum_5d"] = corn / corn.shift(5) - 1.0
    factors["corn_spot_momentum_20d"] = corn / corn.shift(20) - 1.0
    factors["corn_spot_dce_basis"] = corn / panel["dce_corn_close"].replace(0.0, np.nan) - 1.0
    quote_count = panel["corn_quote_count"]
    factors["corn_regional_dispersion"] = (
        (panel["corn_quote_q75"] - panel["corn_quote_q25"])
        / corn.replace(0.0, np.nan)
    ).where(quote_count >= 3)

    factors["corn_quote_count_log"] = np.log1p(quote_count).where(quote_count > 0)
    factors["corn_source_count_log"] = np.log1p(panel["corn_source_count"]).where(
        panel["corn_source_count"] > 0
    )
    factors["corn_confidence_mean"] = panel["corn_confidence_mean"]

    starch_spread = starch / corn.replace(0.0, np.nan) - 1.0
    factors["starch_corn_spread_ratio"] = starch_spread
    factors["starch_corn_spread_change_5d"] = starch_spread.diff(5)

    byproduct_prices = panel[[f"product__{code}" for code in BYPRODUCT_CODES]]
    byproduct_return_5d = byproduct_prices / byproduct_prices.shift(5) - 1.0
    factors["byproduct_momentum_5d"] = mean_with_minimum(byproduct_return_5d, 2)
    byproduct_return_20d = byproduct_prices / byproduct_prices.shift(20) - 1.0
    factors["byproduct_momentum_20d"] = mean_with_minimum(byproduct_return_20d, 2)

    processing_prices = panel[[f"product__{code}" for code in PROCESSING_CODES]]
    processing_return_20d = processing_prices / processing_prices.shift(20) - 1.0
    factors["processing_chain_momentum_20d"] = mean_with_minimum(processing_return_20d, 3)
    return factors.replace([np.inf, -np.inf], np.nan)


def build_matrix(dce: pd.DataFrame, panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(dce["date"], errors="raise")
    available_count = factors.notna().sum(axis=1).astype(int)
    matrix = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "factor_cutoff": "15:00_Asia_Shanghai",
            "point_in_time_verified": False,
            "strict_backtest_eligible": False,
            "shadow_backtest_eligible": available_count >= (len(factors.columns) // 2),
            "is_latest_observation": False,
            "available_factor_count": available_count,
            "candidate_factor_count": len(factors.columns),
            "source_version": SOURCE_VERSION,
        }
    )
    matrix.loc[matrix.index[-1], "is_latest_observation"] = True
    return pd.concat([matrix, factors], axis=1)


def validate_outputs(
    dce: pd.DataFrame,
    factors: pd.DataFrame,
    matrix: pd.DataFrame,
) -> None:
    dates = pd.to_datetime(dce["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise AssertionError("DCE dates must be unique and sorted")
    if list(factors.columns) != output_columns():
        raise AssertionError("Generated factor order differs from the registered definition")
    forbidden = ("target", "next_day", "spike")
    if any(token in column.lower() for column in matrix for token in forbidden):
        raise AssertionError("Market factor matrix must not contain target-like columns")
    constant = [column for column in factors if factors[column].dropna().nunique() <= 1]
    if constant:
        raise AssertionError(f"Constant factors are forbidden: {constant}")
    if factors.notna().sum().min() == 0:
        raise AssertionError("Every candidate factor must have at least one observed value")
    if matrix["point_in_time_verified"].any() or matrix["strict_backtest_eligible"].any():
        raise AssertionError("Historical source vintages are not point-in-time verified")
    if len(matrix) != len(dce):
        raise AssertionError("Market factor matrix must preserve the DCE calendar")


def write_definition(path: Path) -> None:
    definition = {
        "schema_version": 1,
        "id": "corn_daily_market_v1",
        "version": "daily_market_v1",
        "status": "shadow_candidate",
        "instrument": "corn",
        "frequency": "daily_dce_trading_rows",
        "source_files": {
            "raw_quotes": "external/raw_quotes.csv",
            "normalized_prices": "external/normalized_prices.csv",
            "committed_to_repository": False,
        },
        "prediction_contract": {
            "release_cutoff": "15:00 Asia/Shanghai",
            "after_cutoff_rule": "Map to the next DCE trading row.",
            "non_trading_day_rule": "Map to the next DCE trading row.",
            "target_row": "next actual DCE trading row",
            "uses_future_target": False,
            "point_in_time_vintage_verified": False,
        },
        "filtering": {
            "trainable": "yes",
            "allowed_status": ["normalized", "price_parsed", "accepted"],
            "unit": "元/吨",
            "positive_price_only": True,
        },
        "missing_value_policy": {
            "backward_fill": False,
            "forward_fill": False,
            "two_sided_interpolation": False,
            "training_imputation": "fit inside each training fold only",
        },
        "families": FACTOR_FAMILIES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_quotes_path = args.raw_quotes.resolve()
    normalized_path = args.normalized.resolve()
    dce_path = args.dce_raw.resolve()
    factor_root = args.factor_root.resolve()
    definition_path = factor_root / "library" / "daily_market_v1" / "factor_set.yaml"
    matrix_path = factor_root / "matrix" / "corn_market_daily_factors_v1.csv"
    manifest_path = factor_root / "daily_market_v1_manifest.json"

    raw_quotes = pd.read_csv(raw_quotes_path)
    normalized = pd.read_csv(normalized_path)
    dce = pd.read_csv(dce_path).reset_index(drop=True)
    validate_source_pair(raw_quotes, normalized)

    panel = build_daily_panel(normalized, dce)
    factors = build_factors(panel)
    matrix = build_matrix(dce, panel, factors)
    validate_outputs(dce, factors, matrix)

    write_definition(definition_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_path, index=False, encoding="utf-8", float_format="%.8g")

    clean = clean_normalized(normalized)
    source_dates = pd.to_datetime(normalized[COL["release_date"]], errors="raise")
    manifest = {
        "factor_set": "daily_market_v1",
        "status": "shadow_candidate",
        "frequency": "daily_dce_trading_rows",
        "external_sources_committed": False,
        "raw_quotes_file_name": raw_quotes_path.name,
        "raw_quotes_sha256": sha256(raw_quotes_path),
        "raw_quotes_rows": int(len(raw_quotes)),
        "normalized_file_name": normalized_path.name,
        "normalized_sha256": sha256(normalized_path),
        "normalized_rows": int(len(normalized)),
        "source_record_ids_match": True,
        "filtered_valid_rows": int(len(clean)),
        "excluded_rows": int(len(normalized) - len(clean)),
        "authorization_status_count": {
            str(key): int(value)
            for key, value in normalized[COL["authorization"]].value_counts(dropna=False).items()
        },
        "source_date_start": source_dates.min().strftime("%Y-%m-%d"),
        "source_date_end": source_dates.max().strftime("%Y-%m-%d"),
        "release_cutoff": "15:00 Asia/Shanghai",
        "point_in_time_vintage_verified": False,
        "matrix": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": sha256(matrix_path),
        "definition": str(definition_path.relative_to(ROOT)).replace("\\", "/"),
        "definition_sha256": sha256(definition_path),
        "rows": int(len(matrix)),
        "columns": int(len(matrix.columns)),
        "candidate_factor_count": len(output_columns()),
        "factor_ids": output_columns(),
        "missing_value_count": {column: int(factors[column].isna().sum()) for column in factors},
        "first_valid_date": {
            column: (
                matrix.loc[factors[column].first_valid_index(), "date"]
                if factors[column].first_valid_index() is not None
                else None
            )
            for column in factors
        },
        "shadow_backtest_eligible_rows": int(matrix["shadow_backtest_eligible"].sum()),
        "strict_backtest_eligible_rows": 0,
        "target_columns_included": False,
        "forward_fill_used": False,
        "backward_fill_used": False,
        "training_policy": "Use as a separate short-history feature block or residual overlay; fit all preprocessing inside each training fold.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {matrix_path} ({matrix.shape[0]} rows x {matrix.shape[1]} columns)")
    print(f"wrote {definition_path} ({len(output_columns())} candidate factors)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
