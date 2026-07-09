"""Backtest execution engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from corn_forecast.pipeline.backtest.splits import make_backtest_windows
from corn_forecast.config.schema import validate_config
from corn_forecast.data.loader import load_commodity_csv, select_feature_columns
from corn_forecast.data.scaler import SequenceStandardizer
from corn_forecast.data.targets import add_forward_targets
from corn_forecast.data.windowing import make_windows
from corn_forecast.pipeline.eval.metrics import compute_all_metrics
from corn_forecast.operator.model.registry import create_model, expand_model_configs, normalize_model_config
from corn_forecast.pipeline.report.verdict import build_agent_verdict
from corn_forecast.pipeline.report.writer import write_experiment_report, write_model_outputs


def run_backtest(config: dict, *, output_dir: str | Path) -> pd.DataFrame:
    """Run a complete config-driven backtest and write outputs."""

    validate_config(config)
    data_cfg = config["data"]
    target_cfg = config["target"]
    lookback = int(config["lookback"]["default"])
    train_window = config["train_window"]
    eval_cfg = config["evaluation"]
    split_cfg = config.get("split", {})
    val_ratio = float(split_cfg.get("val_ratio", 0.0))

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
    x, y, meta = make_windows(
        df,
        feature_cols=feature_cols,
        target_col="target_direction_fwd",
        date_col=data_cfg["date_col"],
        lookback=lookback,
    )
    y_returns = df["target_return_fwd"].iloc[meta["row_idx"].to_numpy(dtype=int)].to_numpy(dtype=float)
    dates = pd.to_datetime(meta["date"])
    target_dates = pd.to_datetime(meta["target_date"]) if "target_date" in meta.columns else None
    windows = make_backtest_windows(
        dates,
        mode=train_window["mode"],
        min_train_periods=int(train_window["min_train_periods"]),
        stride_periods=int(train_window.get("stride_periods", 1)),
        window_size_periods=train_window.get("window_size_periods"),
        max_train_periods=train_window.get("max_train_periods"),
        target_dates=target_dates,
        target_known_only=bool(train_window.get("target_known_only", False)),
    )
    if not windows:
        raise ValueError("No backtest windows produced")

    comparison_rows: list[dict] = []
    model_outputs: dict[str, tuple[pd.DataFrame, dict]] = {}
    first_model_metrics: dict | None = None
    baseline_metrics: dict | None = None

    enabled_models = [normalize_model_config(model) for model in expand_model_configs(config["models"])]
    for model_cfg in [model for model in enabled_models if model.get("enabled", True)]:
        model_name = model_cfg["name"]
        pred_probs: list[float] = []
        true_labels: list[int] = []
        actual_returns: list[float] = []
        raw_reg_predictions: list[float] = []
        naive_return_predictions: list[float] = []
        rows: list[dict] = []

        for window in windows:
            train_idx = np.asarray(window.train_idx, dtype=int)
            val_idx = np.asarray([], dtype=int)
            if val_ratio > 0:
                val_size = int(len(train_idx) * val_ratio)
                if val_size >= 1 and len(train_idx) - val_size >= 2:
                    val_idx = train_idx[-val_size:]
                    train_idx = train_idx[:-val_size]

            scaler = SequenceStandardizer().fit(x[train_idx], y[train_idx])
            x_train = scaler.transform_x(x[train_idx])
            x_val = scaler.transform_x(x[val_idx]) if len(val_idx) else None
            y_val = y[val_idx] if len(val_idx) else None
            x_test = scaler.transform_x(x[window.test_idx])
            meta_target_dates = pd.to_datetime(meta["target_date"]) if "target_date" in meta.columns else None

            run_model_cfg = dict(model_cfg)
            if model_cfg.get("name") == "dual_stream_lstm" or model_cfg.get("type") == "dual_stream_lstm":
                params = dict(run_model_cfg.get("params") or {})
                params.setdefault("feature_cols", feature_cols)
                run_model_cfg["params"] = params
            model = create_model(run_model_cfg)
            if hasattr(model, "fit_with_targets"):
                y_return_val = y_returns[val_idx] if len(val_idx) else None
                model.fit_with_targets(x_train, y[train_idx], y_returns[train_idx], x_val, y_val, y_return_val)
            else:
                model.fit(x_train, y[train_idx], x_val, y_val)

            raw_reg = model.predict_regression(x_test) if hasattr(model, "predict_regression") else None
            if raw_reg is not None and not hasattr(model, "predict_proba"):
                raw_reg = np.asarray(raw_reg, dtype=float).reshape(-1)
                pred = (raw_reg > 0.0).astype(int)
                prob = pred.astype(float)
            else:
                prob = model.predict_proba(x_test)
                if prob is None:
                    pred = model.predict(x_test).astype(int)
                    prob = pred.astype(float)
                else:
                    prob = np.asarray(prob, dtype=float).reshape(-1)
                    pred = (prob > 0.5).astype(int)
                raw_reg = np.asarray(raw_reg, dtype=float).reshape(-1) if raw_reg is not None else None

            naive_return = float(np.mean(y_returns[train_idx]))

            for local_idx, sample_idx in enumerate(window.test_idx):
                true_label = int(y[sample_idx])
                source_row_idx = int(meta["row_idx"].iloc[sample_idx])
                actual_return = float(df["target_return_fwd"].iloc[source_row_idx])
                pred_label = int(pred[local_idx])
                pred_prob = float(prob[local_idx])
                target_date = (
                    str(pd.to_datetime(meta_target_dates.iloc[sample_idx]).date())
                    if meta_target_dates is not None
                    else ""
                )
                pred_probs.append(pred_prob)
                true_labels.append(true_label)
                actual_returns.append(actual_return)
                if raw_reg is not None:
                    raw_reg_predictions.append(float(raw_reg[local_idx]))
                    naive_return_predictions.append(naive_return)
                rows.append(
                    {
                        "date": str(pd.to_datetime(meta["date"].iloc[sample_idx]).date()),
                        "actual_label": true_label,
                        "predicted_label": pred_label,
                        "predicted_probability": pred_prob,
                        "predicted_return": float(raw_reg[local_idx]) if raw_reg is not None else "",
                        "direction_correct": int(true_label == pred_label),
                        "actual_return": actual_return,
                        "strategy_return": actual_return if pred_label == 1 else -actual_return,
                        "model": model_name,
                        "window_id": window.window_id,
                        "train_start_date": str(pd.to_datetime(meta["date"].iloc[train_idx[0]]).date()),
                        "train_end_date": str(pd.to_datetime(meta["date"].iloc[train_idx[-1]]).date()),
                        "train_max_target_date": (
                            str(pd.to_datetime(meta_target_dates.iloc[train_idx].max()).date())
                            if meta_target_dates is not None
                            else ""
                        ),
                        "val_start_date": str(pd.to_datetime(meta["date"].iloc[val_idx[0]]).date()) if len(val_idx) else "",
                        "val_end_date": str(pd.to_datetime(meta["date"].iloc[val_idx[-1]]).date()) if len(val_idx) else "",
                        "val_max_target_date": (
                            str(pd.to_datetime(meta_target_dates.iloc[val_idx].max()).date())
                            if meta_target_dates is not None and len(val_idx)
                            else ""
                        ),
                        "test_date": str(pd.to_datetime(meta["date"].iloc[sample_idx]).date()),
                        "test_target_date": target_date,
                    }
                )

        predictions = pd.DataFrame(rows)
        predictions["equity"] = (1.0 + predictions["strategy_return"]).cumprod()
        metrics = compute_all_metrics(
            np.asarray(true_labels),
            np.asarray(pred_probs),
            np.asarray(actual_returns),
            n_bootstrap=int(eval_cfg.get("ci_bootstrap_samples", 200)),
            ci_level=float(eval_cfg.get("ci_level", 0.95)),
            annualize=12,
            raw_reg_pred=np.asarray(raw_reg_predictions) if len(raw_reg_predictions) == len(true_labels) else None,
            naive_mean_ret_pred=np.asarray(naive_return_predictions) if len(naive_return_predictions) == len(true_labels) else None,
        )
        if first_model_metrics is None:
            first_model_metrics = metrics
        if baseline_metrics is None and model_cfg.get("type") == "baseline":
            baseline_metrics = metrics
        comparison_rows.append({"model": model_name, **metrics})
        model_outputs[model_name] = (predictions, metrics)

    comparison = pd.DataFrame(comparison_rows).sort_values(["DirAcc", "ProfitFactor", "Sharpe"], ascending=[False, False, False])
    best_row = comparison.iloc[0].to_dict()
    verdict = build_agent_verdict(best_row, baseline_metrics=baseline_metrics or first_model_metrics, primary_metric="DirAcc")
    best_model = str(best_row["model"])
    best_predictions = model_outputs[best_model][0]
    for output_model_name, (model_predictions, model_metrics) in model_outputs.items():
        write_model_outputs(output_dir, output_model_name, model_predictions, model_metrics)
    write_experiment_report(
        output_dir=output_dir,
        model_name=best_model,
        predictions=best_predictions,
        comparison=comparison,
        metrics=best_row,
        verdict=verdict,
        config=config,
        write_model_output=False,
    )
    data_manifest = {
        "csv_path": data_cfg["csv_path"],
        "encoding": encoding,
        "rows_after_target": int(len(df)),
        "feature_cols": feature_cols,
        "exclude_feature_cols": data_cfg.get("exclude_feature_cols", []),
        "exclude_feature_patterns": data_cfg.get("exclude_feature_patterns", []),
        "windows": len(windows),
        "scaling": "train_only_sequence_standardizer",
        "target_known_only": bool(train_window.get("target_known_only", False)),
        "val_ratio": val_ratio,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return comparison
