"""Unified experiment entry point.

The structure follows lightning-hydra-template's single train entry idea, but
uses a small YAML loader so this project can run without Hydra.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.config_loader import load_config, merge_dicts
from src.data.loader import resolve_input_path
from src.data.manifest import build_data_manifest
from src.data.pipeline import DataPipeline
from src.eval.comparator import write_comparison
from src.eval.metrics import best_classification_threshold, evaluate_classification, evaluate_model
from src.eval.reporter import build_classification_frame, build_prediction_frame, generate_classification_report, generate_report
from src.eval.summary import save_training_history, write_report

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified commodity trend experiments.")
    parser.add_argument("--experiment", default="experiment", help="Experiment YAML name under src/config.")
    parser.add_argument("--models", default="", help="Comma-separated model names to run.")
    parser.add_argument("--smoke", action="store_true", help="Run only the first enabled model with short DL epochs.")
    parser.add_argument("overrides", nargs="*", help="Dotted overrides, e.g. data.test_start=2024-06-01.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.overrides)
    exp_cfg = load_config(args.experiment)
    data_cfg = merge_dicts(load_config(exp_cfg.get("data_config", "data")), overrides.get("data", {}))
    if args.smoke:
        data_cfg["smoke_train_tail_n"] = 200
        data_cfg["smoke_val_n"] = 80
        data_cfg["smoke_test_n"] = 80
    seed_everything(int(exp_cfg.get("seed", 42)))
    bundle = DataPipeline(data_cfg).run()

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    out_root = Path(exp_cfg.get("output_dir", "experiments")) / f"{run_id}_{exp_cfg.get('experiment_name', 'experiment')}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "experiment_config.json").write_text(json.dumps(exp_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "data_config.json").write_text(json.dumps(data_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = resolve_input_path(data_cfg.get("csv_path", "玉米预测/datasets/raw/corn_daily_enriched.csv"))
    data_manifest = build_data_manifest(csv_path, bundle.feature_cols, bundle, data_cfg)
    (out_root / "data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "preprocess.json").write_text(json.dumps(bundle.scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    requested = {m.strip() for m in args.models.split(",") if m.strip()}
    model_entries = [m for m in exp_cfg.get("models", []) if m.get("enabled", True)]
    if requested:
        model_entries = [m for m in model_entries if m["name"] in requested]
    if args.smoke and not requested and model_entries:
        model_entries = model_entries[:1]

    results: dict[str, dict[str, float]] = {}
    for entry in model_entries:
        model_cfg = load_config(entry["config"])
        model_name = model_cfg["name"]
        if args.smoke and model_cfg["class_path"].startswith("src.models.deep"):
            model_cfg.setdefault("params", {})["epochs"] = 2
        if args.smoke and model_name in {"random_forest", "xgboost", "lightgbm"}:
            model_cfg.setdefault("params", {})["n_estimators"] = min(int(model_cfg["params"].get("n_estimators", 20)), 20)
            if model_name in {"xgboost", "lightgbm"}:
                model_cfg["params"]["n_estimators"] = 5
                model_cfg["params"]["n_jobs"] = 1
                model_cfg["params"]["max_depth"] = min(int(model_cfg["params"].get("max_depth", 2)), 2)
            if model_name == "xgboost":
                model_cfg["params"]["n_estimators"] = 1
                model_cfg["params"]["tree_method"] = "hist"
                model_cfg["params"]["verbosity"] = 0
        print(f"\n=== {model_name} ===", flush=True)
        model = instantiate_model(model_cfg, bundle, data_cfg)
        if hasattr(model, "set_y_inverse"):
            inverse_fn = bundle.y_scaler.inverse_y if data_cfg.get("scale_y", True) else lambda y: np.asarray(y)
            model.set_y_inverse(inverse_fn)
        model.fit(bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val)
        model_dir = out_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_dir / model_artifact_name(model_cfg))
        save_training_history(model, model_dir)
        if is_classification_task(data_cfg, model_cfg):
            class_outputs = write_classification_split_outputs(model, model_name, bundle, data_cfg, model_dir)
            test_payload = class_outputs["payloads"]["test"]
            metrics = generate_classification_report(
                y_true=test_payload["y_true"],
                prob=test_payload["prob"],
                model_name=model_name,
                output_dir=model_dir,
                meta=bundle.meta_test,
                threshold=float(class_outputs["threshold"]),
                threshold_rule=class_outputs["threshold_rule"],
            )
            results[model_name] = metrics
            (model_dir / "model_config.json").write_text(json.dumps(model_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        split_outputs = write_split_outputs(model, model_name, model_cfg, bundle, data_cfg, model_dir, csv_path)
        test_payload = split_outputs["payloads"]["test"]
        metrics = generate_report(
            y_true=test_payload["y_true"],
            y_pred=test_payload["y_pred"],
            today_close=test_payload["today_close"],
            model_name=model_name,
            output_dir=model_dir,
            meta=bundle.meta_test,
            y_true_return=test_payload["y_true_return"],
            y_pred_return=test_payload["y_pred_return"],
            periods_per_year=int(data_cfg.get("periods_per_year", 252)),
        )
        results[model_name] = metrics
        (model_dir / "model_config.json").write_text(json.dumps(model_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = write_comparison(results, out_root)
    extra_lines = ["- Model dirs include `all_predictions.csv` and `split_metrics.csv`."]
    if data_cfg.get("github_compatible_outputs", False):
        extra_lines.append("- GitHub-compatible monthly outputs: `monthly_lstm_predictions.csv`, `monthly_lstm_features.csv`, `monthly_lstm_summary.json`.")
    write_report(out_root, exp_cfg.get("experiment_name", "experiment"), "holdout_smoke" if args.smoke else "holdout", data_cfg, comparison, extra_lines=extra_lines)
    print("\n=== Comparison ===")
    print(comparison.to_string(index=False))


def instantiate_model(model_cfg: dict[str, Any], bundle, data_cfg: dict[str, Any] | None = None):
    cls = import_from_path(model_cfg["class_path"])
    params = dict(model_cfg.get("params", {}))
    if model_cfg["class_path"].startswith("src.models.deep"):
        params.setdefault("input_size", bundle.X_train.shape[1])
        params.setdefault("seq_len", bundle.X_train.shape[2])
        if "dual_stream_lstm" in model_cfg["class_path"]:
            params.setdefault("feature_cols", bundle.feature_cols)
            params.setdefault("news_feature_prefix", (data_cfg or {}).get("news_feature_prefix", "pca_"))
    else:
        params.setdefault("task", model_cfg.get("task", "regression"))
    if model_cfg["class_path"].endswith("baseline.ZeroReturnBaseline") and (data_cfg or {}).get("target_mode") == "return":
        params["scaled_value"] = float(bundle.y_scaler.transform_y(np.asarray([0.0], dtype=np.float32))[0])
    if "baseline.LastReturnBaseline" in model_cfg["class_path"] or "baseline.MovingAverageReturnBaseline" in model_cfg["class_path"]:
        close_idx = int(params.get("close_feature_idx", 0))
        params["close_mean"] = float(bundle.scaler.x_mean[close_idx])
        params["close_std"] = float(bundle.scaler.x_std[close_idx])
        params["y_mean"] = float(bundle.scaler.y_mean)
        params["y_std"] = float(bundle.scaler.y_std)
    return cls(**params)


def is_classification_task(data_cfg: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> bool:
    return data_cfg.get("target_mode") == "classification" or (model_cfg or {}).get("task") == "classification"


def write_classification_split_outputs(model, model_name: str, bundle, data_cfg: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    val_prob = predict_classification_prob(model, bundle.X_val)
    threshold_rule = "fixed_0p5"
    threshold = float(data_cfg.get("classification_threshold", 0.5))
    if data_cfg.get("classification_threshold_strategy", "fixed") == "validation_f1_weighted":
        threshold_info = best_classification_threshold(bundle.y_val.astype(int), val_prob)
        threshold = float(threshold_info["threshold"])
        threshold_rule = "validation_f1_weighted"

    payloads: dict[str, dict[str, np.ndarray]] = {}
    frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        X = getattr(bundle, f"X_{split}")
        y = getattr(bundle, f"y_{split}").astype(int)
        meta = getattr(bundle, f"meta_{split}")
        prob = predict_classification_prob(model, X)
        payloads[split] = {"y_true": y, "prob": prob}
        frame = build_classification_frame(y, prob, meta=meta, threshold=threshold)
        frame.insert(0, "split", split)
        frames.append(frame)
        metric_rows.append({"split": split, "model": model_name, **evaluate_classification(y, prob, threshold=threshold, threshold_rule=threshold_rule)})

    pd.concat(frames, ignore_index=True).to_csv(model_dir / "all_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(model_dir / "split_metrics.csv", index=False)
    return {"payloads": payloads, "threshold": threshold, "threshold_rule": threshold_rule}


def predict_classification_prob(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def write_split_outputs(model, model_name: str, model_cfg: dict[str, Any], bundle, data_cfg: dict[str, Any], model_dir: Path, csv_path: Path) -> dict[str, Any]:
    periods_per_year = int(data_cfg.get("periods_per_year", 252))
    payloads: dict[str, dict[str, np.ndarray]] = {}
    prediction_frames: list[pd.DataFrame] = []
    github_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        X = getattr(bundle, f"X_{split}")
        y = getattr(bundle, f"y_{split}")
        meta = getattr(bundle, f"meta_{split}")
        payload = prediction_payload(model, X, y, meta, bundle, data_cfg)
        payloads[split] = payload

        frame = build_prediction_frame(
            y_true=payload["y_true"],
            y_pred=payload["y_pred"],
            today_close=payload["today_close"],
            meta=meta,
            y_true_return=payload["y_true_return"],
            y_pred_return=payload["y_pred_return"],
        )
        frame.insert(0, "split", split)
        prediction_frames.append(frame)

        metrics = evaluate_model(payload["y_true"], payload["y_pred"], payload["today_close"], periods_per_year=periods_per_year)
        metrics["pred_return_constant_flag"] = bool(np.nanstd(payload["y_pred_return"]) < 1e-12)
        metric_rows.append({"split": split, "model": model_name, **metrics})

        if data_cfg.get("github_compatible_outputs", False):
            github_frames.append(build_github_monthly_prediction_frame(split, meta, payload))

    pd.concat(prediction_frames, ignore_index=True).to_csv(model_dir / "all_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(model_dir / "split_metrics.csv", index=False)

    if github_frames:
        pd.concat(github_frames, ignore_index=True).to_csv(model_dir / "monthly_lstm_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"feature": bundle.feature_cols}).to_csv(model_dir / "monthly_lstm_features.csv", index=False, encoding="utf-8-sig")
        monthly_summary = build_github_monthly_summary(model, model_cfg, bundle, data_cfg, csv_path, metric_rows)
        (model_dir / "monthly_lstm_summary.json").write_text(json.dumps(monthly_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"payloads": payloads}


def prediction_payload(model, X, y_scaled, meta, bundle, data_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    pred_scaled = model.predict(X)
    y_pred_target = inverse_target(pred_scaled.reshape(-1), bundle, data_cfg)
    y_true_target = inverse_target(np.asarray(y_scaled).reshape(-1), bundle, data_cfg)
    today_close = meta["today_close"].to_numpy(dtype=float)
    if data_cfg.get("target_mode") == "return":
        y_pred = today_close * (1.0 + y_pred_target)
        y_true = today_close * (1.0 + y_true_target)
        y_pred_return = y_pred_target
        y_true_return = y_true_target
    else:
        y_pred = y_pred_target
        y_true = y_true_target
        y_pred_return = y_pred / today_close - 1.0
        y_true_return = y_true / today_close - 1.0
    return {
        "y_true": np.asarray(y_true, dtype=float).reshape(-1),
        "y_pred": np.asarray(y_pred, dtype=float).reshape(-1),
        "today_close": np.asarray(today_close, dtype=float).reshape(-1),
        "y_true_return": np.asarray(y_true_return, dtype=float).reshape(-1),
        "y_pred_return": np.asarray(y_pred_return, dtype=float).reshape(-1),
    }


def inverse_target(values, bundle, data_cfg: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if data_cfg.get("scale_y", True):
        return bundle.y_scaler.inverse_y(arr).reshape(-1)
    return arr


def build_github_monthly_prediction_frame(split: str, meta: pd.DataFrame, payload: dict[str, np.ndarray]) -> pd.DataFrame:
    today_close = payload["today_close"]
    y_true = payload["y_true"]
    y_pred = payload["y_pred"]
    out = pd.DataFrame(
        {
            "split": split,
            "input_end_month": pd.to_datetime(meta["date"]).dt.strftime("%Y-%m"),
            "target_month": pd.to_datetime(meta["target_date"]).dt.strftime("%Y-%m"),
            "today_close": today_close,
            "actual_close": y_true,
            "actual_change": y_true - today_close,
            "actual_return": y_true / today_close - 1.0,
            "pred_close": y_pred,
            "pred_change": y_pred - today_close,
            "actual_direction": (y_true > today_close).astype(int),
            "pred_direction": (y_pred > today_close).astype(int),
        }
    )
    return out


def build_github_monthly_summary(model, model_cfg: dict[str, Any], bundle, data_cfg: dict[str, Any], csv_path: Path, metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metas = pd.concat([bundle.meta_train, bundle.meta_val, bundle.meta_test], ignore_index=True)
    metrics_by_split = {}
    for row in metric_rows:
        split = row["split"]
        metrics_by_split[split] = {k: v for k, v in row.items() if k not in {"split", "model"}}
    history = getattr(model, "history", None) or {}
    best = best_training_point(history)
    return {
        "data_path": str(csv_path.resolve()),
        "target": data_cfg.get("price_col", "dce_corn_close"),
        "task": data_cfg.get("task", "past seq_len months -> next month dce_corn_close regression"),
        "seq_len": int(data_cfg.get("seq_len", 12)),
        "horizon": int(data_cfg.get("horizon", 1)),
        "features": len(bundle.feature_cols),
        "windows": int(len(metas)),
        "train_windows": int(len(bundle.X_train)),
        "val_windows": int(len(bundle.X_val)),
        "test_windows": int(len(bundle.X_test)),
        "first_target_month": pd.to_datetime(metas["target_date"].iloc[0]).strftime("%Y-%m"),
        "last_target_month": pd.to_datetime(metas["target_date"].iloc[-1]).strftime("%Y-%m"),
        **best,
        "params": model_cfg.get("params", {}),
        "train_metrics": metrics_by_split.get("train", {}),
        "val_metrics": metrics_by_split.get("val", {}),
        "test_metrics": metrics_by_split.get("test", {}),
    }


def best_training_point(history: dict[str, Any]) -> dict[str, Any]:
    if history.get("val_mae"):
        values = np.asarray(history["val_mae"], dtype=float)
        idx = int(np.nanargmin(values))
        return {"best_epoch": int(history["epoch"][idx]), "best_val_mae": float(values[idx])}
    if history.get("val_loss"):
        values = np.asarray(history["val_loss"], dtype=float)
        idx = int(np.nanargmin(values))
        return {"best_epoch": int(history["epoch"][idx]), "best_val_loss": float(values[idx])}
    return {}


def import_from_path(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def model_artifact_name(model_cfg: dict[str, Any]) -> str:
    return "model.pt" if model_cfg["class_path"].startswith("src.models.deep") else "model.pkl"


def parse_overrides(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        value = coerce_value(raw)
        current = parsed
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return parsed


def coerce_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
