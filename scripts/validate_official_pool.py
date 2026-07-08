#!/usr/bin/env python3
"""Validate the framework-integrated official 57-model pool."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import run_backtest
from backtest.splits import make_backtest_windows
from config.loader import load_config
from data.loader import load_commodity_csv, select_feature_columns
from data.targets import add_forward_targets
from data.windowing import make_windows
from models.official_pool import OFFICIAL_57_MODEL_NAMES
from models.registry import expand_model_configs


DEFAULT_CONFIGS = [
    "configs/corn_official_pool_57_h1_no_news.yaml",
    "configs/corn_official_pool_57_h2_no_news.yaml",
    "configs/corn_official_pool_57_h1_with_news.yaml",
    "configs/corn_official_pool_57_h2_with_news.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", default="experiments/official_pool_validation")
    parser.add_argument("--smoke-models", default="random_forest_shallow")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--ci-bootstrap-samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    smoke_rows = []
    for config_path in args.configs:
        cfg = load_config(config_path, validate=True)
        audit_rows.append(audit_config(config_path, cfg))
        if not args.skip_smoke:
            smoke_rows.append(run_smoke(config_path, cfg, output_dir, args.smoke_models, args.ci_bootstrap_samples))

    payload = {"configs": audit_rows, "smoke_runs": smoke_rows}
    (output_dir / "official_pool_validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def audit_config(config_path: str, cfg: dict) -> dict:
    expanded = expand_model_configs(cfg["models"])
    model_names = [model["name"] for model in expanded if model.get("enabled", True)]
    if model_names != OFFICIAL_57_MODEL_NAMES:
        raise AssertionError(f"{config_path}: official_57 expansion mismatch")

    data_cfg = cfg["data"]
    target_cfg = cfg["target"]
    train_cfg = cfg["train_window"]
    df, encoding = load_commodity_csv(
        data_cfg["csv_path"],
        date_col=data_cfg["date_col"],
        date_format=data_cfg.get("date_format"),
        encodings=data_cfg.get("encoding", ["utf-8", "gbk", "gb18030"]),
    )
    df = add_forward_targets(
        df,
        price_col=data_cfg["price_col"],
        horizon=int(target_cfg["horizon"]),
        spike_threshold=float(target_cfg.get("spike_threshold", 0.0)),
        date_col=data_cfg["date_col"],
    )
    feature_cols = select_feature_columns(
        df,
        data_cfg["feature_cols"],
        date_col=data_cfg["date_col"],
        exclude_feature_cols=data_cfg.get("exclude_feature_cols", []),
        exclude_feature_patterns=data_cfg.get("exclude_feature_patterns", []),
    )
    pca_count = sum(column.lower().startswith("pca") for column in feature_cols)
    config_name = Path(config_path).name
    if "no_news" in config_name and pca_count != 0:
        raise AssertionError(f"{config_path}: no_news config retained {pca_count} PCA columns")
    if "with_news" in config_name and pca_count <= 0:
        raise AssertionError(f"{config_path}: with_news config has no PCA columns")

    lookback_rows = []
    for lookback in cfg["lookback"].get("candidates", [cfg["lookback"]["default"]]):
        x, y, meta = make_windows(
            df,
            feature_cols=feature_cols,
            target_col="target_direction_fwd",
            date_col=data_cfg["date_col"],
            lookback=int(lookback),
        )
        _ = (x, y)
        dates = pd.to_datetime(meta["date"])
        target_dates = pd.to_datetime(meta["target_date"])
        windows = make_backtest_windows(
            dates,
            mode=train_cfg["mode"],
            min_train_periods=int(train_cfg["min_train_periods"]),
            stride_periods=int(train_cfg.get("stride_periods", 1)),
            window_size_periods=train_cfg.get("window_size_periods"),
            max_train_periods=train_cfg.get("max_train_periods"),
            target_dates=target_dates,
            target_known_only=bool(train_cfg.get("target_known_only", False)),
        )
        if not windows:
            raise AssertionError(f"{config_path}: lookback {lookback} produced no windows")
        leaks = 0
        for window in windows:
            train_target_max = target_dates.iloc[window.train_idx].max()
            test_anchor = dates.iloc[window.test_idx].min()
            if train_target_max > test_anchor:
                leaks += 1
            if len(window.test_idx) != 1:
                raise AssertionError(f"{config_path}: expected one test row per rolling window")
        if leaks:
            raise AssertionError(f"{config_path}: lookback {lookback} has {leaks} target-known leaks")
        lookback_rows.append({"lookback": int(lookback), "windows": len(windows)})

    return {
        "config": config_path,
        "encoding": encoding,
        "rows_after_target": int(len(df)),
        "models_enabled": len(model_names),
        "feature_count": len(feature_cols),
        "pca_feature_count": int(pca_count),
        "lookbacks": lookback_rows,
        "target_known_only": bool(train_cfg.get("target_known_only", False)),
    }


def run_smoke(config_path: str, cfg: dict, output_dir: Path, smoke_models: str, ci_bootstrap_samples: int) -> dict:
    run_cfg = copy.deepcopy(cfg)
    enabled = [item.strip() for item in smoke_models.split(",") if item.strip()]
    run_cfg["models"] = {"pool": "official_57", "enable_only": enabled}
    run_cfg.setdefault("evaluation", {})
    run_cfg["evaluation"]["ci_bootstrap_samples"] = int(ci_bootstrap_samples)
    run_name = Path(config_path).stem
    run_dir = output_dir / run_name
    comparison = run_backtest(run_cfg, output_dir=run_dir)
    comparison.to_csv(run_dir / "official_pool_smoke_comparison.csv", index=False)
    best = comparison.iloc[0].to_dict()
    return {
        "config": config_path,
        "output_dir": str(run_dir),
        "models": enabled,
        "rows": int(len(comparison)),
        "best_model": str(best["model"]),
        "best_DirAcc": float(best["DirAcc"]),
        "best_BalancedAcc": float(best["BalancedAcc"]),
        "best_AUC": float(best["AUC"]),
        "best_AP": float(best["AP"]),
        "best_R2_health": None if pd.isna(best.get("R2_health")) else best.get("R2_health"),
    }


if __name__ == "__main__":
    main()
