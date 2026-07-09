"""Commodity CSV loader adapted from Time-Series-Library data_provider/data_loader.py.

Only the project-specific path, feature columns, and future-price target creation
are changed. Time marker tensors are intentionally removed for this framework.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.windower import make_windows


DEFAULT_FEATURES = [
    "dce_corn_close",
    "dce_corn_volume",
    "dce_corn_open_interest",
    "dce_corn_starch_close",
    "domestic_corn_spot_price_cny_t",
    "corn_basis_cny_t",
    "cbot_corn_close",
    "cbot_wheat_close",
    "ne_avg_precip_mm",
    "ne_avg_t2m_c",
]


DEFAULT_EXCLUDE_FEATURES = {
    "target_price_fwd",
    "target_return_fwd",
    "target_direction_fwd",
}


def resolve_input_path(csv_path: str | Path) -> Path:
    candidates = [Path(csv_path), Path("玉米预测/datasets/processed") / Path(csv_path).name, Path("玉米预测/datasets/raw") / Path(csv_path).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"CSV not found. Tried: {', '.join(str(p) for p in candidates)}")


def load_real_data(
    csv_path: str | Path,
    feature_cols: list[str] | str | None = None,
    date_col: str = "date",
    exclude_feature_cols: list[str] | None = None,
    max_missing_ratio: float | None = None,
    date_format: str | None = None,
    feature_rank_path: str | Path | None = None,
    top_n: int | None = None,
    structured_top_n: int | None = None,
    news_top_n: int | None = None,
    news_feature_prefix: str = "pca_",
    endogenous_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    path = resolve_input_path(csv_path)
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Input CSV must contain date column: {date_col}.")
    df[date_col] = parse_dates(df[date_col], date_format)
    df = df.sort_values(date_col).reset_index(drop=True)

    selected = select_feature_columns(
        df,
        feature_cols,
        date_col,
        exclude_feature_cols,
        max_missing_ratio,
        feature_rank_path=feature_rank_path,
        top_n=top_n,
        structured_top_n=structured_top_n,
        news_top_n=news_top_n,
        news_feature_prefix=news_feature_prefix,
    )
    selected = move_endogenous_last(selected, endogenous_col)
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df, selected


def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: list[str] | str | None,
    date_col: str,
    exclude_feature_cols: list[str] | None = None,
    max_missing_ratio: float | None = None,
    feature_rank_path: str | Path | None = None,
    top_n: int | None = None,
    structured_top_n: int | None = None,
    news_top_n: int | None = None,
    news_feature_prefix: str = "pca_",
) -> list[str]:
    excluded = {date_col, *DEFAULT_EXCLUDE_FEATURES, *(exclude_feature_cols or [])}
    if feature_rank_path:
        ranks = pd.read_csv(resolve_input_path(feature_rank_path), encoding="utf-8-sig")
        if "feature" not in ranks.columns:
            raise ValueError("feature_rank_path must contain a 'feature' column.")
        selected = [c for c in ranks["feature"].astype(str).tolist() if c in df.columns and c not in excluded]
        if top_n:
            selected = selected[: int(top_n)]
    elif feature_cols == "dual_stream_default":
        numeric_cols = [
            col
            for col in df.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(df[col]) and (max_missing_ratio is None or float(df[col].isna().mean()) <= max_missing_ratio)
        ]
        structured = [col for col in numeric_cols if not col.startswith(news_feature_prefix)]
        news = [col for col in numeric_cols if col.startswith(news_feature_prefix)]
        if structured_top_n is not None:
            structured = structured[: int(structured_top_n)]
        if news_top_n is not None:
            news = news[: int(news_top_n)]
        selected = structured + news
    elif feature_cols == "auto_numeric":
        selected = []
        for col in df.columns:
            if col in excluded or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            missing_ratio = float(df[col].isna().mean())
            if max_missing_ratio is None or missing_ratio <= max_missing_ratio:
                selected.append(col)
    else:
        selected = list(feature_cols or DEFAULT_FEATURES)
    if not selected:
        raise ValueError("No feature columns selected.")
    return selected


def parse_dates(values: pd.Series, date_format: str | None = None) -> pd.Series:
    if date_format:
        parsed = pd.to_datetime(values.astype(str), format=date_format, errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise ValueError(f"Could not parse date values with format={date_format!r}: {bad}")
    return parsed


def move_endogenous_last(feature_cols: list[str], endogenous_col: str | None = None) -> list[str]:
    if not endogenous_col:
        return feature_cols
    if endogenous_col not in feature_cols:
        raise ValueError(f"endogenous_col={endogenous_col!r} is not in selected feature columns.")
    return [col for col in feature_cols if col != endogenous_col] + [endogenous_col]


def add_future_price_target(
    df: pd.DataFrame,
    price_col: str = "dce_corn_close",
    horizon: int = 30,
    target_col: str = "target_price_fwd",
) -> pd.DataFrame:
    out = df.copy()
    out[target_col] = out[price_col].shift(-horizon)
    out["target_return_fwd"] = out[target_col] / out[price_col] - 1.0
    out["target_direction_fwd"] = (out["target_return_fwd"] > 0).astype("float32")
    return out


def load_and_window(config: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    csv_path = config.get("csv_path", "玉米预测/datasets/raw/corn_daily_enriched.csv")
    feature_cols = config.get("feature_cols") or DEFAULT_FEATURES
    exclude_feature_cols = config.get("exclude_feature_cols") or []
    max_missing_ratio = config.get("max_missing_ratio")
    date_col = config.get("date_col", "date")
    date_format = config.get("date_format")
    seq_len = int(config.get("seq_len", config.get("window_len", 30)))
    horizon = int(config.get("horizon", 30))
    include_today = bool(config.get("include_today", True))
    target_mode = config.get("target_mode", "price")
    price_col = config.get("price_col", "dce_corn_close")
    series_id = config.get("series_id", "corn")

    df, feature_cols = load_real_data(
        csv_path,
        feature_cols,
        date_col=date_col,
        exclude_feature_cols=exclude_feature_cols,
        max_missing_ratio=None if max_missing_ratio is None else float(max_missing_ratio),
        date_format=date_format,
        feature_rank_path=config.get("feature_rank_path"),
        top_n=config.get("top_n"),
        structured_top_n=config.get("structured_top_n"),
        news_top_n=config.get("news_top_n"),
        news_feature_prefix=config.get("news_feature_prefix", "pca_"),
        endogenous_col=config.get("endogenous_col"),
    )
    default_target = {
        "classification": "target_direction_fwd",
        "return": "target_return_fwd",
        "price": "target_price_fwd",
    }.get(target_mode, "target_price_fwd")
    target_col = str(config.get("target_col") or default_target)
    if target_col not in df.columns:
        df = add_future_price_target(df, price_col=price_col, horizon=horizon)

    x, y, meta = make_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=seq_len,
        horizon=horizon,
        include_today=include_today,
        date_col=date_col,
        today_col=price_col,
        series_id=series_id,
        target_alignment=config.get("target_alignment", "anchor"),
    )
    meta["feature_cols"] = [feature_cols] * len(meta)
    meta.attrs["feature_cols"] = feature_cols
    meta.attrs["csv_path"] = str(resolve_input_path(csv_path))
    meta.attrs["target_col"] = target_col
    meta.attrs["target_mode"] = target_mode
    meta.attrs["horizon"] = horizon
    meta.attrs["series_id"] = series_id
    return x, y, meta
