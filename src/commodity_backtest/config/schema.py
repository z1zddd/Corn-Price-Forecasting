"""Config validation for commodity backtests."""

from __future__ import annotations

from collections.abc import Mapping


REQUIRED_TOP_LEVEL = (
    "commodity",
    "data",
    "target",
    "lookback",
    "train_window",
    "evaluation",
    "models",
)


def validate_config(config: Mapping) -> None:
    """Validate required fields and temporal window constraints."""

    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            raise ValueError(f"Missing required top-level config section: {key}")

    data = config["data"]
    for key in ("csv_path", "date_col", "price_col", "feature_cols"):
        if key not in data:
            raise ValueError(f"Missing required data config field: {key}")

    target = config["target"]
    if int(target.get("horizon", 0)) < 1:
        raise ValueError("target.horizon must be >= 1")
    if target.get("mode") not in {"classification", "return", "price"}:
        raise ValueError("target.mode must be classification, return, or price")

    lookback = config["lookback"]
    default_lookback = int(lookback.get("default", 0))
    if default_lookback < 1:
        raise ValueError("lookback.default must be >= 1")

    train_window = config["train_window"]
    mode = train_window.get("mode")
    if mode not in {"expanding", "rolling", "expanding_with_cap"}:
        raise ValueError("train_window.mode must be expanding, rolling, or expanding_with_cap")

    min_train = int(train_window.get("min_train_periods", 0))
    if min_train < 2:
        raise ValueError("train_window.min_train_periods must be >= 2")
    if default_lookback >= min_train:
        raise ValueError("lookback.default must be smaller than train_window.min_train_periods")
    candidates = lookback.get("candidates", [default_lookback])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("lookback.candidates must be a non-empty list")
    for candidate in candidates:
        candidate_lookback = int(candidate)
        if candidate_lookback < 1:
            raise ValueError("lookback.candidates values must be >= 1")
        if candidate_lookback >= min_train:
            raise ValueError("lookback.candidates values must be smaller than train_window.min_train_periods")

    stride = int(train_window.get("stride_periods", 1))
    if stride < 1:
        raise ValueError("train_window.stride_periods must be >= 1")

    if mode == "rolling":
        window_size = int(train_window.get("window_size_periods", 0))
        if window_size < min_train:
            raise ValueError("rolling window_size_periods must be >= min_train_periods")

    if mode == "expanding_with_cap":
        max_train = int(train_window.get("max_train_periods", 0))
        if max_train < min_train:
            raise ValueError("expanding_with_cap max_train_periods must be >= min_train_periods")

    split = config.get("split", {})
    val_ratio = float(split.get("val_ratio", 0.0))
    if not (0.0 <= val_ratio < 0.5):
        raise ValueError("split.val_ratio must be >= 0 and < 0.5")

    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")