"""Collect a soybean monthly dataset similar to the corn mixed-feature table.

The dataset intentionally excludes news/PCA features. It combines:
- DCE/Sina main continuous futures for soybeans and related oilseed contracts.
- Yahoo Finance CBOT futures for soybeans, soybean meal, and soybean oil.
- World Bank Pink Sheet monthly commodity price series.
- Open-Meteo historical weather for China, US, and Brazil soybean regions.
- UN Comtrade public preview monthly China soybean imports, cached locally.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


DEFAULT_START_DATE = "2016-06-01"
DEFAULT_END_DATE = "2026-06-30"
RAW_OUTPUT = Path("玉米预测/datasets/raw/soybean_daily_enriched.csv")
MONTHLY_OUTPUT = Path("玉米预测/datasets/processed/soybean_monthly_no_news_pca.csv")
LATEST_OUTPUT = Path("玉米预测/datasets/processed/soybean_monthly_latest_unlabeled.csv")
CACHE_DIR = Path("玉米预测/datasets/raw/source_cache")

DCE_SYMBOLS = {
    "A0": "dce_soybean",
    "B0": "dce_soybean2",
    "M0": "dce_soybean_meal",
    "Y0": "dce_soybean_oil",
    "P0": "dce_palm_oil",
}

DCE_CONTRACT_PREFIXES = {
    "dce_soybean": "A",
    "dce_soybean2": "B",
    "dce_soybean_meal": "M",
    "dce_soybean_oil": "Y",
    "dce_palm_oil": "P",
}

YAHOO_SYMBOLS = {
    "ZS=F": "cbot_soybean",
    "ZM=F": "cbot_soybean_meal",
    "ZL=F": "cbot_soybean_oil",
    "ZC=F": "cbot_corn",
    "ZW=F": "cbot_wheat",
}

WORLD_BANK_MONTHLY_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)

WORLD_BANK_COMMODITIES = {
    "Soybeans": "global_soybean_price_usd_t",
    "Soybean meal": "global_soybean_meal_price_usd_t",
    "Soybean oil": "global_soybean_oil_price_usd_t",
    "Palm oil": "global_palm_oil_price_usd_t",
    "Maize": "global_corn_price_usd_t",
    "Wheat, US SRW": "global_wheat_price_usd_t",
}
GLOBAL_PRICE_COLUMNS = list(WORLD_BANK_COMMODITIES.values())

WEATHER_REGIONS = {
    "hlj": (45.8, 126.5),
    "jilin": (43.8, 125.3),
    "inner_mongolia": (43.6, 122.3),
    "iowa": (42.0, -93.5),
    "illinois": (40.0, -89.0),
    "mato_grosso": (-12.6, -55.7),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect soybean mixed-feature monthly dataset.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    parser.add_argument("--monthly-output", type=Path, default=MONTHLY_OUTPUT)
    parser.add_argument("--latest-output", type=Path, default=LATEST_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--skip-imports", action="store_true", help="Skip UN Comtrade import-volume collection.")
    parser.add_argument("--refresh-imports", action="store_true", help="Refresh UN Comtrade cache.")
    return parser.parse_args()


def request_text(url: str, params: dict | None = None, timeout: int = 30) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(1.5 + attempt)
                continue
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error


def fetch_sina_inner_daily(symbol: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    url = "https://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine"
    text = request_text(url, params={"symbol": symbol})
    data = json.loads(text)
    if not data:
        raise ValueError(f"Sina returned no data for {symbol}")
    df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df.rename(columns={col: f"{prefix}_{col}" for col in ["open", "high", "low", "close", "volume"]})
    return df.reset_index(drop=True)


def fetch_sina_contract_daily(symbol: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sina_contract_{symbol}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if not cached.empty and cached["date"].notna().any() and cached["close"].notna().any():
            return cached
        cache_path.unlink(missing_ok=True)
    url = (
        "https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_V21052021_4_12="
        "/InnerFuturesNewService.getDailyKLine"
    )
    params = {"symbol": symbol, "type": "2021_04_12"}
    text = request_text(url, params=params, timeout=20)
    if "=(" not in text:
        return pd.DataFrame()
    try:
        payload = json.loads(text.split("=(", 1)[1].rsplit(");", 1)[0])
    except json.JSONDecodeError:
        return pd.DataFrame()
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload).rename(
        columns={
            "d": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "p": "hold",
            "s": "settle",
        }
    )
    df = df[["date", "open", "high", "low", "close", "volume", "hold", "settle"]]
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "hold", "settle"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["contract"] = symbol
    df.to_csv(cache_path, index=False)
    return df


def contract_symbols(root: str, start: str, end: str) -> list[str]:
    start_year = pd.Timestamp(start).year - 1
    end_year = pd.Timestamp(end).year + 1
    months = [1, 3, 5, 7, 8, 9, 11, 12]
    if root in {"A"}:
        months = [1, 3, 5, 7, 9, 11]
    elif root in {"B", "P"}:
        months = list(range(1, 13))
    symbols = []
    for year in range(start_year, end_year + 1):
        yy = str(year)[-2:]
        for month in months:
            symbols.append(f"{root}{yy}{month:02d}")
    return symbols


def fetch_sina_synthetic_main(
    root: str,
    prefix: str,
    start: str,
    end: str,
    cache_dir: Path,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sina_synthetic_main_{prefix}_{start}_{end}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])

    frames = []
    symbols = contract_symbols(root, start, end)
    for idx, symbol in enumerate(symbols, 1):
        if idx == 1 or idx % 12 == 0 or idx == len(symbols):
            print(f"  {prefix} contract {idx}/{len(symbols)}: {symbol}", flush=True)
        df = fetch_sina_contract_daily(symbol, cache_dir)
        if df.empty:
            continue
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        if not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError(f"No contract data collected for {prefix}")

    all_contracts = pd.concat(frames, ignore_index=True)
    all_contracts = all_contracts.dropna(subset=["close"])
    all_contracts = all_contracts.sort_values(["date", "volume", "hold"], ascending=[True, False, False])
    selected = all_contracts.drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)
    selected = selected.rename(columns={col: f"{prefix}_{col}" for col in ["open", "high", "low", "close", "volume"]})
    selected = selected[["date", f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close", f"{prefix}_volume"]]
    selected.to_csv(cache_path, index=False)
    return selected


def splice_sina_main_with_contracts(symbol: str, prefix: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    base = fetch_sina_inner_daily(symbol, prefix, start, end)
    if base.empty:
        root = DCE_CONTRACT_PREFIXES[prefix]
        return fetch_sina_synthetic_main(root, prefix, start, end, cache_dir)
    last_base_date = base["date"].max()
    if last_base_date >= pd.Timestamp(end):
        return base
    root = DCE_CONTRACT_PREFIXES[prefix]
    extension_start = (last_base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    synthetic = fetch_sina_synthetic_main(root, prefix, extension_start, end, cache_dir)
    extension = synthetic[synthetic["date"] > last_base_date]
    out = pd.concat([base, extension], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out


def unix_timestamp(day: str, add_days: int = 0) -> int:
    stamp = pd.Timestamp(day) + pd.Timedelta(days=add_days)
    return int(stamp.to_pydatetime().replace(tzinfo=timezone.utc).timestamp())


def fetch_yahoo_daily(symbol: str, prefix: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": str(unix_timestamp(start)),
        "period2": str(unix_timestamp(end, add_days=1)),
        "interval": "1d",
        "events": "history",
    }
    payload = json.loads(request_text(url, params=params))
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).date,
            f"{prefix}_open": quote.get("open"),
            f"{prefix}_high": quote.get("high"),
            f"{prefix}_low": quote.get("low"),
            f"{prefix}_close": quote.get("close"),
            f"{prefix}_volume": quote.get("volume"),
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)


def fetch_world_bank_monthly_prices(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "worldbank_cmo_historical_data_monthly.xlsx"
    if not cache_path.exists():
        content = requests.get(WORLD_BANK_MONTHLY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).content
        cache_path.write_bytes(content)

    raw = pd.read_excel(cache_path, sheet_name="Monthly Prices", header=None)
    headers = raw.iloc[4].astype(str).str.strip().tolist()
    header_to_idx = {header: idx for idx, header in enumerate(headers)}
    data = raw.iloc[6:].copy()
    month_raw = data.iloc[:, 0].astype(str)
    out = pd.DataFrame({"month": pd.PeriodIndex(month_raw.str.replace("M", "-", regex=False), freq="M")})
    for header, col_name in WORLD_BANK_COMMODITIES.items():
        if header not in header_to_idx:
            raise ValueError(f"World Bank Pink Sheet column not found: {header}")
        values = data.iloc[:, header_to_idx[header]].replace("…", np.nan)
        out[col_name] = pd.to_numeric(values, errors="coerce")
    start_month = pd.Timestamp(start).to_period("M")
    end_month = pd.Timestamp(end).to_period("M")
    out = out[(out["month"] >= start_month) & (out["month"] <= end_month)]
    return out.sort_values("month").reset_index(drop=True)


def fetch_yahoo_fx(start: str, end: str) -> pd.DataFrame:
    df = fetch_yahoo_daily("CNY=X", "usd_cny", start, end)
    return df[["date", "usd_cny_close"]].rename(columns={"usd_cny_close": "usd_cny"})


def fetch_open_meteo(region: str, latitude: float, longitude: float, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"open_meteo_{region}_{start}_{end}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"])
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum,temperature_2m_mean",
        "timezone": "Asia/Shanghai",
    }
    payload = json.loads(request_text(url, params=params, timeout=45))
    daily = payload["daily"]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            f"{region}_precip_mm": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
            f"{region}_t2m_c": pd.to_numeric(daily["temperature_2m_mean"], errors="coerce"),
        }
    )
    out.to_csv(cache_path, index=False)
    return out


def month_periods(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"))


def fetch_comtrade_soybean_imports(
    start: str,
    end: str,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "comtrade_china_soybean_imports_hs1201_monthly.csv"
    periods = month_periods(start, end)
    needed = {period.strftime("%Y%m") for period in periods}
    cached = pd.DataFrame()
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, dtype={"period": str})

    cached_periods = set(cached["period"].astype(str).tolist()) if not cached.empty else set()
    rows = cached.to_dict("records") if not cached.empty else []
    missing_periods = sorted(needed - cached_periods)
    for idx, period in enumerate(missing_periods, 1):
        if idx == 1 or idx % 12 == 0 or idx == len(missing_periods):
            print(f"  Comtrade period {idx}/{len(missing_periods)}: {period}", flush=True)
        url = "https://comtradeapi.un.org/public/v1/preview/C/M/HS"
        params = {
            "reporterCode": "156",
            "period": period,
            "partnerCode": "0",
            "cmdCode": "1201",
            "flowCode": "M",
        }
        try:
            data = fetch_comtrade_period(url, params)
        except Exception as exc:
            print(f"  Comtrade period {period} failed: {exc}", flush=True)
            data = None
        rows.append(
            {
                "period": period,
                "month": pd.Period(period, freq="M").strftime("%Y-%m"),
                "soybean_import_volume_ton": data.get("netWgt", np.nan) / 1000.0 if data else np.nan,
                "soybean_import_value_usd": data.get("primaryValue", np.nan) if data else np.nan,
            }
        )
        pd.DataFrame(rows).drop_duplicates("period", keep="last").sort_values("period").to_csv(cache_path, index=False)
        if idx < len(missing_periods):
            time.sleep(1.15)

    out = pd.DataFrame(rows).drop_duplicates("period", keep="last").sort_values("period")
    if out.empty:
        raise ValueError("No Comtrade import rows were collected.")
    out.to_csv(cache_path, index=False)
    out["month"] = pd.PeriodIndex(out["month"], freq="M")
    out["soybean_import_unit_value_usd_t"] = out["soybean_import_value_usd"] / out["soybean_import_volume_ton"]
    return out[["month", "soybean_import_volume_ton", "soybean_import_value_usd", "soybean_import_unit_value_usd_t"]]


def fetch_comtrade_period(url: str, params: dict) -> dict | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(3):
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            time.sleep(1.0 + attempt)
            continue
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("data"):
            return None
        return payload["data"][0]
    raise RuntimeError(f"Comtrade rate limit did not clear for period={params.get('period')}")


def merge_on_date(frames: Iterable[pd.DataFrame], start: str, end: str) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    out = calendar
    for frame in frames:
        out = out.merge(frame, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def add_daily_features(daily: pd.DataFrame, global_prices_monthly: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["month"] = out["date"].dt.to_period("M")
    out = out.merge(global_prices_monthly, on="month", how="left")
    out["usd_cny"] = out["usd_cny"].ffill().bfill()
    price_cols = [col for col in GLOBAL_PRICE_COLUMNS if col in out.columns]
    for col in price_cols:
        out[col] = out[col].ffill().bfill()
        cny_col = col.replace("_usd_t", "_cny_t")
        out[cny_col] = out[col] * out["usd_cny"]

    close_cols = [col for col in out.columns if col.endswith("_close")]
    for col in close_cols:
        out[col.replace("_close", "_ret_1d")] = out[col].pct_change(fill_method=None)

    out["soybean_meal_soybean_spread_close"] = out["dce_soybean_meal_close"] - out["dce_soybean_close"]
    out["soybean_oil_soybean_spread_close"] = out["dce_soybean_oil_close"] - out["dce_soybean_close"]
    out["soybean2_soybean_spread_close"] = out["dce_soybean2_close"] - out["dce_soybean_close"]
    out["palm_soybean_oil_spread_close"] = out["dce_palm_oil_close"] - out["dce_soybean_oil_close"]
    out["global_soybean_basis_cny_t"] = out["global_soybean_price_cny_t"] - out["dce_soybean_close"]
    out["global_soybean_basis_rate"] = out["global_soybean_basis_cny_t"] / out["dce_soybean_close"]

    out["is_soybean_harvest_season_ne_cn"] = out["date"].dt.month.isin([9, 10]).astype(int)
    out["is_soybean_planting_season_ne_cn"] = out["date"].dt.month.isin([4, 5]).astype(int)
    out["is_us_soybean_harvest_season"] = out["date"].dt.month.isin([9, 10, 11]).astype(int)
    out["is_brazil_soybean_harvest_season"] = out["date"].dt.month.isin([2, 3, 4, 5]).astype(int)

    china_precip = ["hlj_precip_mm", "jilin_precip_mm", "inner_mongolia_precip_mm"]
    china_temp = ["hlj_t2m_c", "jilin_t2m_c", "inner_mongolia_t2m_c"]
    us_precip = ["iowa_precip_mm", "illinois_precip_mm"]
    us_temp = ["iowa_t2m_c", "illinois_t2m_c"]
    out["ne_avg_precip_mm"] = out[china_precip].mean(axis=1)
    out["ne_avg_t2m_c"] = out[china_temp].mean(axis=1)
    out["us_midwest_avg_precip_mm"] = out[us_precip].mean(axis=1)
    out["us_midwest_avg_t2m_c"] = out[us_temp].mean(axis=1)
    return out


def aggregate_ohlcv(daily: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close_col = f"{prefix}_close"
    frame = daily.dropna(subset=[close_col]).copy()
    grouped = frame.groupby("month", sort=True)
    out = pd.DataFrame(index=grouped.size().index)
    out[f"{prefix}_open_first"] = grouped[f"{prefix}_open"].first()
    out[f"{prefix}_high_max"] = grouped[f"{prefix}_high"].max()
    out[f"{prefix}_low_min"] = grouped[f"{prefix}_low"].min()
    out[f"{prefix}_close_last"] = grouped[close_col].last()
    out[f"{prefix}_close_mean"] = grouped[close_col].mean()
    out[f"{prefix}_close_std"] = grouped[close_col].std().fillna(0.0)
    if f"{prefix}_volume" in frame.columns:
        out[f"{prefix}_volume_sum"] = grouped[f"{prefix}_volume"].sum(min_count=1)
    out[f"{prefix}_ret_1m"] = out[f"{prefix}_close_last"].pct_change(fill_method=None)
    out[f"{prefix}_month_range_pct"] = (out[f"{prefix}_high_max"] - out[f"{prefix}_low_min"]) / out[f"{prefix}_close_last"]
    return out


def aggregate_monthly(daily: pd.DataFrame, imports: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade = daily.dropna(subset=["dce_soybean_close"]).copy()
    grouped_trade = trade.groupby("month", sort=True)
    monthly = pd.DataFrame(index=grouped_trade.size().index)
    monthly["first_trade_date"] = grouped_trade["date"].first().dt.strftime("%Y-%m-%d")
    monthly["last_trade_date"] = grouped_trade["date"].last().dt.strftime("%Y-%m-%d")
    monthly["n_obs"] = grouped_trade.size()

    for prefix in [*DCE_SYMBOLS.values(), *YAHOO_SYMBOLS.values()]:
        agg = aggregate_ohlcv(daily, prefix)
        monthly = monthly.join(agg, how="left")

    daily_grouped = daily.groupby("month", sort=True)
    mean_last_cols = [
        "usd_cny",
        "global_soybean_price_usd_t",
        "global_soybean_price_cny_t",
        "global_soybean_meal_price_usd_t",
        "global_soybean_meal_price_cny_t",
        "global_soybean_oil_price_usd_t",
        "global_soybean_oil_price_cny_t",
        "global_palm_oil_price_usd_t",
        "global_palm_oil_price_cny_t",
        "global_corn_price_usd_t",
        "global_corn_price_cny_t",
        "global_wheat_price_usd_t",
        "global_wheat_price_cny_t",
        "global_soybean_basis_cny_t",
        "global_soybean_basis_rate",
        "soybean_meal_soybean_spread_close",
        "soybean_oil_soybean_spread_close",
        "soybean2_soybean_spread_close",
        "palm_soybean_oil_spread_close",
    ]
    for col in mean_last_cols:
        monthly[f"{col}_mean"] = daily_grouped[col].mean()
        monthly[f"{col}_last"] = daily_grouped[col].last()

    weather_sum_cols = [
        col for col in daily.columns if col.endswith("_precip_mm") or col in {"ne_avg_precip_mm", "us_midwest_avg_precip_mm"}
    ]
    weather_mean_cols = [col for col in daily.columns if col.endswith("_t2m_c") or col in {"ne_avg_t2m_c", "us_midwest_avg_t2m_c"}]
    for col in weather_sum_cols:
        monthly[f"{col}_sum"] = daily_grouped[col].sum()
        monthly[f"{col}_mean"] = daily_grouped[col].mean()
    for col in weather_mean_cols:
        monthly[f"{col}_mean"] = daily_grouped[col].mean()
        monthly[f"{col}_last"] = daily_grouped[col].last()

    for col in [
        "is_soybean_harvest_season_ne_cn",
        "is_soybean_planting_season_ne_cn",
        "is_us_soybean_harvest_season",
        "is_brazil_soybean_harvest_season",
    ]:
        monthly[f"{col}_flag"] = daily_grouped[col].max()
        monthly[f"{col}_share"] = daily_grouped[col].mean()

    if imports is not None and not imports.empty:
        imports = imports.copy().set_index("month").sort_index()
        monthly = monthly.join(imports, how="left")
        for col in ["soybean_import_volume_ton", "soybean_import_value_usd", "soybean_import_unit_value_usd_t"]:
            monthly[f"{col}_ffill"] = monthly[col].ffill().bfill()
            monthly = monthly.drop(columns=[col])

    monthly["dce_soybean_close"] = monthly["dce_soybean_close_last"]
    close = monthly["dce_soybean_close"]
    monthly["dce_soybean_close_ret_1m"] = close.pct_change(fill_method=None)
    for window in (3, 6, 12):
        monthly[f"dce_soybean_close_ma{window}"] = close.rolling(window, min_periods=1).mean()
        monthly[f"dce_soybean_close_ret_{window}m"] = close / close.shift(window) - 1.0
        monthly[f"dce_soybean_close_vol_{window}m"] = monthly["dce_soybean_close_ret_1m"].rolling(window, min_periods=1).std()

    monthly["dce_soybean_close_next_month"] = close.shift(-1)
    monthly["dce_soybean_close_next_month_ret"] = monthly["dce_soybean_close_next_month"] / close - 1.0
    monthly["dce_soybean_close_next_month_direction"] = (monthly["dce_soybean_close_next_month_ret"] > 0).astype("float32")

    monthly = monthly.reset_index()
    monthly["month"] = monthly["month"].astype(str)
    latest = monthly.tail(1).copy()
    labeled = monthly.dropna(subset=["dce_soybean_close_next_month", "dce_soybean_close_next_month_ret"]).copy()
    abs_threshold = float(labeled["dce_soybean_close_next_month_ret"].abs().median())
    labeled["spike"] = (labeled["dce_soybean_close_next_month_ret"].abs() >= abs_threshold).astype("int64")

    labeled = finalize_monthly(labeled)
    latest = finalize_monthly(latest, keep_unlabeled=True)
    return labeled, latest


def finalize_monthly(df: pd.DataFrame, keep_unlabeled: bool = False) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col.endswith("_ret_1m") or "_ret_" in col or "_vol_" in col:
            out[col] = out[col].fillna(0.0)
        else:
            out[col] = out[col].ffill().bfill()
    if keep_unlabeled:
        for col in ["dce_soybean_close_next_month", "dce_soybean_close_next_month_ret", "dce_soybean_close_next_month_direction"]:
            if col in out.columns:
                out[col] = out[col].fillna(np.nan)
    missing_total = int(out.isna().sum().sum())
    if missing_total and not keep_unlabeled:
        missing_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Monthly output has {missing_total} missing values in columns: {missing_cols}")
    return out


def collect(args: argparse.Namespace) -> dict[str, object]:
    start, end = args.start_date, args.end_date
    print("Fetching DCE/Sina main continuous futures and contract-based extensions...", flush=True)
    futures_frames = [splice_sina_main_with_contracts(symbol, prefix, start, end, args.cache_dir) for symbol, prefix in DCE_SYMBOLS.items()]
    print("Fetching Yahoo/CBOT futures and USD/CNY...", flush=True)
    yahoo_frames = [fetch_yahoo_daily(symbol, prefix, start, end) for symbol, prefix in YAHOO_SYMBOLS.items()]
    fx = fetch_yahoo_fx(start, end)
    print("Fetching Open-Meteo weather archives...", flush=True)
    weather_frames = [fetch_open_meteo(region, lat, lon, start, end, args.cache_dir) for region, (lat, lon) in WEATHER_REGIONS.items()]
    print("Fetching World Bank Pink Sheet commodity prices...", flush=True)
    global_prices_monthly = fetch_world_bank_monthly_prices(start, end, args.cache_dir)
    print("Fetching UN Comtrade China soybean imports..." if not args.skip_imports else "Skipping UN Comtrade imports.", flush=True)
    imports = None if args.skip_imports else fetch_comtrade_soybean_imports(start, end, args.cache_dir, refresh=args.refresh_imports)

    print("Merging daily data and building monthly table...", flush=True)
    daily = merge_on_date([*futures_frames, *yahoo_frames, fx, *weather_frames], start, end)
    daily = add_daily_features(daily, global_prices_monthly)
    labeled, latest = aggregate_monthly(daily, imports)

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.monthly_output.parent.mkdir(parents=True, exist_ok=True)
    args.latest_output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.raw_output, index=False)
    labeled.to_csv(args.monthly_output, index=False)
    latest.to_csv(args.latest_output, index=False)

    return {
        "raw_output": str(args.raw_output),
        "monthly_output": str(args.monthly_output),
        "latest_output": str(args.latest_output),
        "daily_rows": int(daily.shape[0]),
        "daily_columns": int(daily.shape[1]),
        "monthly_rows": int(labeled.shape[0]),
        "monthly_columns": int(labeled.shape[1]),
        "latest_month": str(latest["month"].iloc[0]) if not latest.empty else None,
        "monthly_min": str(labeled["month"].iloc[0]),
        "monthly_max": str(labeled["month"].iloc[-1]),
        "monthly_missing_values": int(labeled.isna().sum().sum()),
        "imports_collected": imports is not None,
    }


def main() -> None:
    args = parse_args()
    summary = collect(args)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
