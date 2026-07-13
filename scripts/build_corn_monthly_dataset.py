#!/usr/bin/env python3
"""Build leakage-aware monthly corn feature tables from tracked data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "corn_forecast" / "datasets" / "corn" / "raw"
PROCESSED_DIR = ROOT / "corn_forecast" / "datasets" / "corn" / "processed"


MARKET_SPECS = {
    "dce_corn": {
        "settle": True,
        "volume": True,
        "open_interest": True,
    },
    "dce_corn_starch": {
        "settle": True,
        "volume": True,
        "open_interest": True,
    },
    "cbot_corn": {
        "settle": False,
        "volume": False,
        "open_interest": False,
    },
    "cbot_wheat": {
        "settle": False,
        "volume": False,
        "open_interest": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=None, help="Raw daily corn CSV.")
    parser.add_argument("--legacy", type=Path, default=None, help="Legacy monthly CSV containing PCA columns.")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def find_default_raw() -> Path:
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one tracked raw CSV in {RAW_DIR}, found {len(candidates)}")
    return candidates[0]


def find_default_legacy() -> Path:
    candidates = sorted(path for path in PROCESSED_DIR.glob("*.csv") if "LSTM" in path.name)
    if len(candidates) != 1:
        raise ValueError(f"Expected one legacy LSTM CSV in {PROCESSED_DIR}, found {len(candidates)}")
    return candidates[0]


def read_csv(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Unable to decode {path}: {'; '.join(errors)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compound_return(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def expected_business_month_end(month_start: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(month_start) + pd.offsets.BMonthEnd(0)


def build_core(raw: pd.DataFrame) -> pd.DataFrame:
    if "date" not in raw.columns:
        raise ValueError("Raw data must contain a date column")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].dt.date.astype(str).tolist()
        raise ValueError(f"Duplicate raw dates: {duplicates[:10]}")
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["period"] = frame["date"].dt.to_period("M")
    grouped = frame.groupby("period", sort=True)

    result = pd.DataFrame(index=grouped.size().index)
    result["month"] = result.index.to_timestamp()
    result["first_trade_date"] = grouped["date"].min().to_numpy()
    result["last_trade_date"] = grouped["date"].max().to_numpy()
    result["n_obs"] = grouped["dce_corn_close"].count().to_numpy(dtype=int)

    periods = list(result.index)
    complete = pd.Series(True, index=result.index, dtype=bool)
    if periods:
        complete.iloc[0] = False
        last_start = periods[-1].to_timestamp()
        complete.iloc[-1] = bool(result.iloc[-1]["last_trade_date"] >= expected_business_month_end(last_start))
    result["is_complete_period"] = complete.to_numpy()

    for prefix, spec in MARKET_SPECS.items():
        result[f"{prefix}_open_first"] = grouped[f"{prefix}_open"].first()
        result[f"{prefix}_high_max"] = grouped[f"{prefix}_high"].max()
        result[f"{prefix}_low_min"] = grouped[f"{prefix}_low"].min()
        result[f"{prefix}_close_last"] = grouped[f"{prefix}_close"].last()
        result[f"{prefix}_close_mean"] = grouped[f"{prefix}_close"].mean()
        result[f"{prefix}_close_std"] = grouped[f"{prefix}_close"].std()
        if spec["settle"]:
            result[f"{prefix}_settle_last"] = grouped[f"{prefix}_settle"].last()
        if spec["volume"]:
            result[f"{prefix}_volume_sum"] = grouped[f"{prefix}_volume"].sum(min_count=1)
        if spec["open_interest"]:
            result[f"{prefix}_open_interest_last"] = grouped[f"{prefix}_open_interest"].last()

        safe_close = result[f"{prefix}_close_last"].where(complete)
        result[f"{prefix}_ret_1m"] = safe_close.pct_change(fill_method=None)
        result[f"{prefix}_month_range_pct"] = (
            (result[f"{prefix}_high_max"] - result[f"{prefix}_low_min"])
            / result[f"{prefix}_close_last"].replace(0, np.nan)
        )
        daily_return_col = f"{prefix}_ret_1d"
        result[f"{prefix}_ret_1m_compound"] = grouped[daily_return_col].apply(compound_return).where(complete)

    aggregation_specs = {
        "cs_c_spread_close": ("mean", "last"),
        "domestic_corn_spot_price_cny_t": ("mean", "last"),
        "corn_basis_cny_t": ("mean", "last"),
        "corn_basis_rate": ("mean", "last"),
        "corn_100ppi_main_futures_price_cny_t": ("mean", "last"),
        "cbot_wheat_corn_ratio": ("mean", "last"),
        "hlj_precip_mm": ("sum", "mean"),
        "jilin_precip_mm": ("sum", "mean"),
        "inner_mongolia_precip_mm": ("sum", "mean"),
        "ne_avg_precip_mm": ("sum", "mean"),
        "hlj_t2m_c": ("mean", "last"),
        "jilin_t2m_c": ("mean", "last"),
        "inner_mongolia_t2m_c": ("mean", "last"),
        "ne_avg_t2m_c": ("mean", "last"),
        "corn_100ppi_nearby_futures_price_cny_t": ("mean", "last"),
        "corn_100ppi_nearby_basis_cny_t": ("mean", "last"),
        "corn_100ppi_nearby_basis_rate": ("mean", "last"),
    }
    for column, operations in aggregation_specs.items():
        for operation in operations:
            output = f"{column}_{operation}"
            if operation == "sum":
                result[output] = grouped[column].sum(min_count=1)
            elif operation == "mean":
                result[output] = grouped[column].mean()
            elif operation == "last":
                result[output] = grouped[column].last()
            else:
                raise ValueError(f"Unsupported aggregation: {operation}")

    harvest_share = grouped["is_harvest_season_ne_cn"].mean()
    result["is_harvest_season_ne_cn_share"] = harvest_share
    result["is_harvest_season_ne_cn_flag"] = (harvest_share > 0).astype(int)

    result["dce_corn_close"] = result["dce_corn_close_last"]
    safe_price = result["dce_corn_close"].where(complete)
    safe_return = safe_price.pct_change(fill_method=None)
    result["dce_corn_close_ret_1m"] = safe_return
    for window in (3, 6, 12):
        result[f"dce_corn_close_ma{window}"] = safe_price.rolling(window, min_periods=window).mean()
        result[f"dce_corn_close_ret_{window}m"] = safe_price / safe_price.shift(window) - 1.0
        result[f"dce_corn_close_vol_{window}m"] = safe_return.rolling(window, min_periods=window).std()

    result = result.reset_index(drop=True)
    result["month"] = pd.to_datetime(result["month"]).dt.strftime("%Y-%m-%d")
    result["first_trade_date"] = pd.to_datetime(result["first_trade_date"]).dt.strftime("%Y-%m-%d")
    result["last_trade_date"] = pd.to_datetime(result["last_trade_date"]).dt.strftime("%Y-%m-%d")
    return result


def build_news(legacy: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    if "month" not in legacy.columns:
        raise ValueError("Legacy monthly data must contain a month column")
    pca_columns = [column for column in legacy.columns if str(column).lower().startswith("pca_")]
    if not pca_columns:
        raise ValueError("Legacy monthly data contains no pca_* columns")

    news = legacy[["month", *pca_columns]].copy()
    news["month"] = pd.to_datetime(news["month"].astype(str), format="%y-%b").dt.strftime("%Y-%m-%d")
    completeness = core.set_index("month")["is_complete_period"]
    news.insert(1, "is_complete_period", news["month"].map(completeness).fillna(False).astype(bool))
    news.insert(2, "available_at", "")
    news.insert(3, "source_status", "legacy_unknown_provenance")
    news.insert(4, "strict_backtest_eligible", False)
    return news


def validate_outputs(core: pd.DataFrame, news: pd.DataFrame) -> None:
    if core["month"].duplicated().any() or news["month"].duplicated().any():
        raise AssertionError("Generated monthly keys must be unique")
    if not core["month"].is_monotonic_increasing:
        raise AssertionError("Core months must be sorted")
    if len(core) != len(news):
        raise AssertionError("Core and news tables must cover the same number of months")
    if set(core["month"]) != set(news["month"]):
        raise AssertionError("Core and news month keys differ")
    forbidden = {"target_price_fwd", "target_return_fwd", "target_direction_fwd", "spike"}
    if forbidden.intersection(core.columns) or forbidden.intersection(news.columns):
        raise AssertionError("Feature tables must not contain target columns")
    if core.filter(regex=r"^dce_corn_close_(ma|ret_[3612]|vol_)").iloc[0].notna().any():
        raise AssertionError("Rolling features must not be backward-filled")
    if bool(core.iloc[0]["is_complete_period"]):
        raise AssertionError("The first source month is partial and must be flagged")
    if bool(core.iloc[-1]["is_complete_period"]):
        raise AssertionError("The latest tracked month is incomplete and must be flagged")


def main() -> None:
    args = parse_args()
    raw_path = (args.raw or find_default_raw()).resolve()
    legacy_path = (args.legacy or find_default_legacy()).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv(raw_path)
    legacy = read_csv(legacy_path)
    core = build_core(raw)
    news = build_news(legacy, core)
    validate_outputs(core, news)

    core_path = output_dir / "corn_monthly_core_v1.csv"
    news_path = output_dir / "corn_monthly_news_legacy.csv"
    manifest_path = output_dir / "corn_monthly_v1_manifest.json"
    core.to_csv(core_path, index=False, encoding="utf-8")
    news.to_csv(news_path, index=False, encoding="utf-8")

    manifest = {
        "dataset_version": "corn_monthly_v1",
        "raw_source": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256(raw_path),
        "legacy_news_source": str(legacy_path.relative_to(ROOT)),
        "legacy_news_sha256": sha256(legacy_path),
        "data_cutoff": str(pd.to_datetime(raw["date"]).max().date()),
        "core_rows": int(len(core)),
        "core_columns": int(len(core.columns)),
        "news_rows": int(len(news)),
        "pca_columns": int(sum(column.startswith("pca_") for column in news.columns)),
        "incomplete_months": core.loc[~core["is_complete_period"], "month"].tolist(),
        "target_columns_included": False,
        "backward_fill_used": False,
        "two_sided_interpolation_used": False,
        "news_source_status": "legacy_unknown_provenance",
        "strict_backtest_policy": "Use core features only. Legacy PCA columns require provenance and fold-local generation.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {core_path} ({core.shape[0]} rows x {core.shape[1]} columns)")
    print(f"wrote {news_path} ({news.shape[0]} rows x {news.shape[1]} columns)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
