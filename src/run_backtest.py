"""Rolling-origin backtest entry point."""

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
from sklearn.metrics import brier_score_loss

from src.config.config_loader import load_config, merge_dicts
from src.data.cv import rolling_origin_splits
from src.data.loader import load_and_window, resolve_input_path
from src.data.manifest import build_data_manifest
from src.data.pipeline import build_bundle
from src.eval.metrics import (
    apply_platt,
    best_classification_threshold,
    evaluate_classification,
    evaluate_classification_predictions,
    fit_positive_platt,
    logit_np,
    probability_diagnostics,
    sigmoid_np,
)
from src.eval.reporter import build_classification_frame, generate_classification_report, generate_report
from src.eval.summary import save_training_history, write_report

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin commodity trend backtests.")
    parser.add_argument("--experiment", default="experiment")
    parser.add_argument("--validation", default="validation")
    parser.add_argument("--models", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.overrides)
    exp_cfg = load_config(args.experiment)
    data_cfg = merge_dicts(load_config(exp_cfg.get("data_config", "data")), overrides.get("data", {}))
    val_cfg = merge_dicts(load_config(args.validation), overrides.get("validation", {}))
    if args.smoke:
        val_cfg["n_windows"] = min(int(val_cfg.get("n_windows", 2)), 2)
        val_cfg["h"] = min(int(val_cfg.get("h", 30)), 30)
        val_cfg["min_train_size"] = min(int(val_cfg.get("min_train_size", 200)), 200)
    seed_everything(int(exp_cfg.get("seed", 42)))

    x, y, meta = load_and_window(data_cfg)
    windows = rolling_origin_splits(
        meta["target_date"],
        h=int(val_cfg.get("h", 30)),
        n_windows=int(val_cfg.get("n_windows", 6)),
        step_size=val_cfg.get("step_size"),
        min_train_size=val_cfg.get("min_train_size"),
        max_train_size=val_cfg.get("max_train_size"),
    )
    if not windows:
        raise ValueError("No rolling-origin windows were produced; loosen validation config.")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    out_root = Path(exp_cfg.get("output_dir", "experiments")) / f"{run_id}_{exp_cfg.get('experiment_name', 'experiment')}_backtest"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "experiment_config.json").write_text(json.dumps(exp_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "data_config.json").write_text(json.dumps(data_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "validation_config.json").write_text(json.dumps(val_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    requested = {m.strip() for m in args.models.split(",") if m.strip()}
    model_entries = [m for m in exp_cfg.get("models", []) if m.get("enabled", True)]
    if requested:
        model_entries = [m for m in model_entries if m["name"] in requested]

    all_rows: list[dict[str, Any]] = []
    classification_prediction_rows: list[dict[str, Any]] = []
    platt_prediction_rows: list[dict[str, Any]] = []
    first_bundle = None
    for window in windows:
        train_idx, val_idx = split_train_val(window.train_idx, float(data_cfg.get("val_ratio", 0.1)))
        test_idx = window.test_idx
        bundle = build_bundle(x, y, meta, train_idx, val_idx, test_idx, data_cfg)
        first_bundle = first_bundle or bundle
        cutoff_dir = out_root / f"cutoff_{window.window_id:03d}_{window.cutoff_date.date()}"
        cutoff_dir.mkdir(parents=True, exist_ok=True)
        (cutoff_dir / "split.json").write_text(
            json.dumps(
                {
                    "window_id": window.window_id,
                    "cutoff_date": str(window.cutoff_date.date()),
                    "train_rows": int(len(train_idx)),
                    "val_rows": int(len(val_idx)),
                    "test_rows": int(len(test_idx)),
                    "test_start": str(pd.to_datetime(bundle.meta_test["target_date"]).min().date()),
                    "test_end": str(pd.to_datetime(bundle.meta_test["target_date"]).max().date()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        for entry in model_entries:
            model_cfg = load_config(entry["config"])
            model_name = model_cfg["name"]
            apply_smoke_model_params(model_cfg, model_name, args.smoke)
            print(f"\n=== cutoff={window.window_id} model={model_name} ===", flush=True)
            seed_everything(int(exp_cfg.get("seed", 42)) + window.window_id + 1)
            model = instantiate_model(model_cfg, bundle, data_cfg)
            if hasattr(model, "set_y_inverse"):
                inverse_fn = bundle.y_scaler.inverse_y if data_cfg.get("scale_y", True) else lambda value: np.asarray(value)
                model.set_y_inverse(inverse_fn)
            model.fit(bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val)
            model_dir = cutoff_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / model_artifact_name(model_cfg))
            save_training_history(model, model_dir)
            if is_classification_task(data_cfg, model_cfg):
                if data_cfg.get("classification_evaluation") == "validation_platt":
                    frame, metrics = evaluate_platt_classification_window(model_name, model, bundle, data_cfg)
                    frame.insert(0, "model", model_name)
                    frame.insert(0, "cutoff_date", str(window.cutoff_date.date()))
                    frame.insert(0, "window_id", window.window_id)
                    frame.to_csv(model_dir / "platt_predictions.csv", index=False)
                    (model_dir / "platt_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    (model_dir / "model_config.json").write_text(json.dumps(model_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                    all_rows.append({"window_id": window.window_id, "cutoff_date": str(window.cutoff_date.date()), "model": model_name, **metrics["calibrated_threshold_0p5"]})
                    platt_prediction_rows.extend(frame.to_dict("records"))
                    continue

                threshold, threshold_rule = choose_classification_threshold(model, bundle, data_cfg)
                prob_test = predict_classification_prob(model, bundle.X_test)
                y_test = bundle.y_test.astype(int)
                metrics = generate_classification_report(
                    y_true=y_test,
                    prob=prob_test,
                    model_name=model_name,
                    output_dir=model_dir,
                    meta=bundle.meta_test,
                    threshold=threshold,
                    threshold_rule=threshold_rule,
                )
                metrics["val_threshold"] = threshold
                (model_dir / "model_config.json").write_text(json.dumps(model_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                all_rows.append({"window_id": window.window_id, "cutoff_date": str(window.cutoff_date.date()), "model": model_name, **metrics})
                frame = build_classification_frame(y_test, prob_test, meta=bundle.meta_test, threshold=threshold)
                frame.insert(0, "model", model_name)
                frame.insert(0, "cutoff_date", str(window.cutoff_date.date()))
                frame.insert(0, "window_id", window.window_id)
                frame["threshold_rule"] = threshold_rule
                classification_prediction_rows.extend(frame.to_dict("records"))
                continue

            y_true, y_pred, y_true_return, y_pred_return, today_close = predict_prices(model, bundle, data_cfg)
            metrics = generate_report(
                y_true=y_true,
                y_pred=y_pred,
                today_close=today_close,
                model_name=model_name,
                output_dir=model_dir,
                meta=bundle.meta_test,
                y_true_return=y_true_return,
                y_pred_return=y_pred_return,
            )
            (model_dir / "model_config.json").write_text(json.dumps(model_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            all_rows.append({"window_id": window.window_id, "cutoff_date": str(window.cutoff_date.date()), "model": model_name, **metrics})

    assert first_bundle is not None
    csv_path = resolve_input_path(data_cfg.get("csv_path", "玉米预测/datasets/raw/corn_daily_enriched.csv"))
    manifest = build_data_manifest(csv_path, first_bundle.feature_cols, first_bundle, data_cfg)
    manifest["backtest_windows"] = len(windows)
    (out_root / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_root / "backtest_metrics.csv", index=False)
    if platt_prediction_rows:
        prediction_df = pd.DataFrame(platt_prediction_rows)
        prediction_df.to_csv(out_root / "rolling_platt_predictions.csv", index=False, encoding="utf-8-sig")
        prediction_df.to_csv(out_root / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
        comparison = aggregate_platt_predictions(prediction_df)
        comparison.to_csv(out_root / "comparison.csv", index=False)
        write_platt_classification_summary(prediction_df, out_root)
    elif classification_prediction_rows:
        prediction_df = pd.DataFrame(classification_prediction_rows)
        prediction_df.to_csv(out_root / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
        comparison = aggregate_classification_predictions(prediction_df)
        comparison.to_csv(out_root / "comparison.csv", index=False)
        write_classification_summary(prediction_df, out_root)
    else:
        comparison = aggregate_comparison(metrics_df)
        comparison.to_csv(out_root / "comparison.csv", index=False)
    write_report(
        out_root,
        exp_cfg.get("experiment_name", "experiment"),
        "backtest_smoke" if args.smoke else "backtest",
        data_cfg,
        comparison,
        extra_lines=[f"Backtest windows: `{len(windows)}`", f"Validation config: `validation_config.json`"],
    )
    print("\n=== Backtest Comparison ===")
    print(comparison.to_string(index=False))


def split_train_val(train_idx: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    val_size = max(1, int(np.ceil(len(train_idx) * val_ratio)))
    if len(train_idx) <= val_size:
        raise ValueError("Not enough rows to split train/val.")
    return train_idx[:-val_size], train_idx[-val_size:]


def apply_smoke_model_params(model_cfg: dict[str, Any], model_name: str, smoke: bool) -> None:
    if not smoke:
        return
    if model_cfg["class_path"].startswith("src.models.deep"):
        model_cfg.setdefault("params", {})["epochs"] = 2
    if model_name in {"random_forest", "extra_trees", "xgboost", "lightgbm", "gradient_boosting", "adaboost"}:
        model_cfg.setdefault("params", {})["n_estimators"] = min(int(model_cfg["params"].get("n_estimators", 20)), 20)
        if model_name in {"xgboost", "lightgbm"}:
            model_cfg["params"]["n_estimators"] = 5
            model_cfg["params"]["n_jobs"] = 1
            model_cfg["params"]["max_depth"] = min(int(model_cfg["params"].get("max_depth", 2)), 2)
        if model_name == "xgboost":
            model_cfg["params"]["n_estimators"] = 1
            model_cfg["params"]["tree_method"] = "hist"
            model_cfg["params"]["verbosity"] = 0
    if model_name in {"hist_gradient_boosting", "mlp"}:
        model_cfg.setdefault("params", {})["max_iter"] = min(int(model_cfg["params"].get("max_iter", 20)), 20)
    if model_name == "catboost":
        model_cfg.setdefault("params", {})["iterations"] = min(int(model_cfg["params"].get("iterations", 20)), 20)
        model_cfg["params"]["verbose"] = False
    if model_name in {"rocket", "minirocket", "multirocket"}:
        model_cfg.setdefault("params", {})["num_kernels"] = min(int(model_cfg["params"].get("num_kernels", 500)), 500)


def instantiate_model(model_cfg: dict[str, Any], bundle, data_cfg: dict[str, Any]):
    cls = import_from_path(model_cfg["class_path"])
    params = dict(model_cfg.get("params", {}))
    is_torch_like = model_cfg["class_path"].startswith(("src.models.deep", "src.models.official"))
    if is_torch_like:
        params.setdefault("input_size", bundle.X_train.shape[1])
        params.setdefault("seq_len", bundle.X_train.shape[2])
        if "dual_stream_lstm" in model_cfg["class_path"]:
            params.setdefault("feature_cols", bundle.feature_cols)
            params.setdefault("news_feature_prefix", data_cfg.get("news_feature_prefix", "pca_"))
    else:
        params.setdefault("task", model_cfg.get("task", "regression"))
    if model_cfg["class_path"].endswith("baseline.ZeroReturnBaseline") and data_cfg.get("target_mode") == "return":
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


def choose_classification_threshold(model, bundle, data_cfg: dict[str, Any]) -> tuple[float, str]:
    threshold = float(data_cfg.get("classification_threshold", 0.5))
    threshold_rule = "fixed_0p5"
    if data_cfg.get("classification_threshold_strategy", "fixed") == "validation_f1_weighted":
        val_prob = predict_classification_prob(model, bundle.X_val)
        threshold_info = best_classification_threshold(bundle.y_val.astype(int), val_prob)
        threshold = float(threshold_info["threshold"])
        threshold_rule = "validation_f1_weighted"
    return threshold, threshold_rule


def predict_classification_prob(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def predict_classification_logits(model, X) -> np.ndarray:
    if hasattr(model, "predict_logits"):
        return np.asarray(model.predict_logits(X), dtype=float).reshape(-1)
    return logit_np(predict_classification_prob(model, X)).reshape(-1)


def evaluate_platt_classification_window(model_name: str, model, bundle, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_val = bundle.y_val.astype(int)
    y_test = bundle.y_test.astype(int)
    val_logits = predict_classification_logits(model, bundle.X_val)
    val_raw_prob = sigmoid_np(val_logits)
    raw_threshold_info = best_classification_threshold(y_val, val_raw_prob)
    raw_val_threshold = float(raw_threshold_info["threshold"])

    calibrator = fit_positive_platt(
        val_logits,
        y_val,
        l2=float(data_cfg.get("calibrator_l2", 1e-3)),
        max_iter=int(data_cfg.get("calibrator_max_iter", 100)),
    )
    val_calibrated_prob = apply_platt(val_logits, calibrator)
    calibrated_threshold_info = best_classification_threshold(y_val, val_calibrated_prob)
    calibrated_val_threshold = float(calibrated_threshold_info["threshold"])

    test_logits = predict_classification_logits(model, bundle.X_test)
    raw_probability = sigmoid_np(test_logits)
    calibrated_probability = apply_platt(test_logits, calibrator)
    frame = build_platt_prediction_frame(
        y_test=y_test,
        logits=test_logits,
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        meta=bundle.meta_test,
        raw_val_threshold=raw_val_threshold,
        calibrated_val_threshold=calibrated_val_threshold,
        calibrator=calibrator,
        val_summary={
            "raw_val_threshold_accuracy": raw_threshold_info["accuracy"],
            "raw_val_threshold_balanced_accuracy": raw_threshold_info["balanced_accuracy"],
            "raw_val_threshold_f1_weighted": raw_threshold_info["f1_weighted"],
            "raw_val_threshold_f1_positive": raw_threshold_info["f1_positive"],
            "raw_val_brier": float(brier_score_loss(y_val, val_raw_prob)),
            "calibrated_val_brier": float(brier_score_loss(y_val, val_calibrated_prob)),
            "train_samples": int(len(bundle.y_train)),
            "val_samples": int(len(y_val)),
        },
    )
    metrics = platt_metrics_from_frame(frame)
    return frame, metrics


def build_platt_prediction_frame(
    y_test,
    logits,
    raw_probability,
    calibrated_probability,
    meta,
    raw_val_threshold: float,
    calibrated_val_threshold: float,
    calibrator: dict[str, float | str],
    val_summary: dict[str, Any],
) -> pd.DataFrame:
    y_test = np.asarray(y_test, dtype=int).reshape(-1)
    logits = np.asarray(logits, dtype=float).reshape(-1)
    raw_probability = np.asarray(raw_probability, dtype=float).reshape(-1)
    calibrated_probability = np.asarray(calibrated_probability, dtype=float).reshape(-1)
    frame = pd.DataFrame(
        {
            "target": y_test,
            "logit": logits,
            "raw_probability": raw_probability,
            "calibrated_probability": calibrated_probability,
            "raw_pred_0p5": (raw_probability >= 0.5).astype(int),
            "calibrated_pred_0p5": (calibrated_probability >= 0.5).astype(int),
            "calibrated_pred_0p6": (calibrated_probability >= 0.6).astype(int),
            "calibrated_pred_0p7": (calibrated_probability >= 0.7).astype(int),
            "raw_val_threshold": raw_val_threshold,
            "raw_pred_val_threshold": (raw_probability >= raw_val_threshold).astype(int),
            "calibrated_val_threshold": calibrated_val_threshold,
            "calibrated_pred_val_threshold": (calibrated_probability >= calibrated_val_threshold).astype(int),
            "calibrator_method": str(calibrator["method"]),
            "calibrator_a": float(calibrator["a"]),
            "calibrator_b": float(calibrator["b"]),
            **val_summary,
        }
    )
    meta_frame = pd.DataFrame(meta).reset_index(drop=True)
    for col in ["series_id", "date", "target_date", "horizon"]:
        if col in meta_frame:
            if col in {"date", "target_date"}:
                frame[col] = pd.to_datetime(meta_frame[col]).dt.strftime("%Y-%m-%d")
            else:
                frame[col] = meta_frame[col].to_numpy()
    if "date" in frame:
        frame = frame.rename(columns={"date": "input_end_date"})
    return frame


def platt_metrics_from_frame(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    y_true = frame["target"].to_numpy(dtype=int)
    raw_prob = frame["raw_probability"].to_numpy(dtype=float)
    calibrated_prob = frame["calibrated_probability"].to_numpy(dtype=float)
    return {
        "raw_threshold_0p5": evaluate_classification(y_true, raw_prob, threshold=0.5),
        "raw_validation_selected_threshold_reference": evaluate_classification_predictions(
            y_true,
            frame["raw_pred_val_threshold"].to_numpy(dtype=int),
            raw_prob,
            threshold_rule="Reference only: each rolling test month uses that run's past validation set to choose a raw-probability threshold.",
        ),
        "calibrated_threshold_0p5": evaluate_classification(y_true, calibrated_prob, threshold=0.5),
        "calibrated_threshold_0p6": evaluate_classification(y_true, calibrated_prob, threshold=0.6),
        "calibrated_threshold_0p7": evaluate_classification(y_true, calibrated_prob, threshold=0.7),
        "calibrated_validation_selected_threshold_reference": evaluate_classification_predictions(
            y_true,
            frame["calibrated_pred_val_threshold"].to_numpy(dtype=int),
            calibrated_prob,
            threshold_rule="Reference only: threshold is chosen on calibrated validation probabilities, not used as the deployable main result.",
        ),
    }


def predict_prices(model, bundle, data_cfg: dict[str, Any]):
    pred_scaled = model.predict(bundle.X_test)
    y_pred_target = bundle.y_scaler.inverse_y(pred_scaled.reshape(-1)).reshape(-1)
    y_true_target = bundle.y_scaler.inverse_y(bundle.y_test.reshape(-1)).reshape(-1)
    today_close = bundle.meta_test["today_close"].to_numpy(dtype=float)
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
    return y_true, y_pred, y_true_return, y_pred_return, today_close


def aggregate_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "window_id"]
    out = metrics_df.groupby("model", as_index=False)[numeric_cols].mean()
    if "balanced_accuracy" in out.columns and "f1_weighted" in out.columns:
        return out.sort_values(["balanced_accuracy", "f1_weighted"], ascending=[False, False])
    return out.sort_values(["direction_accuracy", "profit_factor"], ascending=[False, False])


def aggregate_classification_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in prediction_df.groupby("model", sort=False):
        y_true = group["target"].to_numpy(dtype=int)
        prob = group["probability"].to_numpy(dtype=float)
        pred = group["pred_label"].to_numpy(dtype=int)
        metrics = evaluate_classification_predictions(
            y_true,
            pred,
            prob,
            threshold_rule="Each rolling test month uses its own past-validation selected threshold.",
        )
        metrics["threshold_mean"] = float(group["threshold"].mean())
        metrics["threshold_median"] = float(group["threshold"].median())
        metrics["threshold_min"] = float(group["threshold"].min())
        metrics["threshold_max"] = float(group["threshold"].max())
        rows.append({"model": model, **metrics})
    return pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1_weighted"], ascending=[False, False])


def write_classification_summary(prediction_df: pd.DataFrame, output_dir: Path) -> None:
    summary: dict[str, Any] = {}
    for model, group in prediction_df.groupby("model", sort=False):
        y_true = group["target"].to_numpy(dtype=int)
        prob = group["probability"].to_numpy(dtype=float)
        pred = group["pred_label"].to_numpy(dtype=int)
        summary[model] = {
            "threshold_0p5": evaluate_classification(y_true, prob, threshold=0.5),
            "validation_selected_threshold": evaluate_classification_predictions(
                y_true,
                pred,
                prob,
                threshold_rule="Each rolling test month uses its own past-validation selected threshold.",
            ),
            "validation_threshold_summary": {
                "mean": float(group["threshold"].mean()),
                "median": float(group["threshold"].median()),
                "min": float(group["threshold"].min()),
                "max": float(group["threshold"].max()),
            },
            "best_global_threshold_leaky_reference": best_classification_threshold(y_true, prob),
        }
    (output_dir / "rolling_classification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_platt_predictions(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in prediction_df.groupby("model", sort=False):
        metrics = platt_metrics_from_frame(group)
        for variant, values in metrics.items():
            row = {"model": model, "variant": variant, **values}
            if variant == "raw_validation_selected_threshold_reference":
                row["threshold_mean"] = float(group["raw_val_threshold"].mean())
                row["threshold_median"] = float(group["raw_val_threshold"].median())
                row["threshold_min"] = float(group["raw_val_threshold"].min())
                row["threshold_max"] = float(group["raw_val_threshold"].max())
            elif variant == "calibrated_validation_selected_threshold_reference":
                row["threshold_mean"] = float(group["calibrated_val_threshold"].mean())
                row["threshold_median"] = float(group["calibrated_val_threshold"].median())
                row["threshold_min"] = float(group["calibrated_val_threshold"].min())
                row["threshold_max"] = float(group["calibrated_val_threshold"].max())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1_weighted"], ascending=[False, False])


def write_platt_classification_summary(prediction_df: pd.DataFrame, output_dir: Path) -> None:
    summary: dict[str, Any] = {}
    for model, group in prediction_df.groupby("model", sort=False):
        y_true = group["target"].to_numpy(dtype=int)
        raw_prob = group["raw_probability"].to_numpy(dtype=float)
        calibrated_prob = group["calibrated_probability"].to_numpy(dtype=float)
        summary[model] = {
            **platt_metrics_from_frame(group),
            "probability_diagnostics": {
                **probability_diagnostics(y_true, raw_prob, "raw"),
                **probability_diagnostics(y_true, calibrated_prob, "calibrated"),
            },
            "raw_validation_threshold_summary": {
                "mean": float(group["raw_val_threshold"].mean()),
                "median": float(group["raw_val_threshold"].median()),
                "min": float(group["raw_val_threshold"].min()),
                "max": float(group["raw_val_threshold"].max()),
            },
            "calibrated_validation_threshold_summary_reference": {
                "mean": float(group["calibrated_val_threshold"].mean()),
                "median": float(group["calibrated_val_threshold"].median()),
                "min": float(group["calibrated_val_threshold"].min()),
                "max": float(group["calibrated_val_threshold"].max()),
            },
            "calibrator_summary": {
                "positive_platt_count": int((group["calibrator_method"] == "positive_platt").sum()),
                "fallback_count": int((group["calibrator_method"] != "positive_platt").sum()),
                "a_mean": float(group["calibrator_a"].mean()),
                "a_median": float(group["calibrator_a"].median()),
                "b_mean": float(group["calibrator_b"].mean()),
                "b_median": float(group["calibrator_b"].median()),
            },
            "validation_brier_summary": {
                "raw_val_brier_mean": float(group["raw_val_brier"].mean()),
                "calibrated_val_brier_mean": float(group["calibrated_val_brier"].mean()),
            },
        }
    (output_dir / "rolling_classification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def import_from_path(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def model_artifact_name(model_cfg: dict[str, Any]) -> str:
    return "model.pt" if model_cfg["class_path"].startswith(("src.models.deep", "src.models.official")) else "model.pkl"


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
