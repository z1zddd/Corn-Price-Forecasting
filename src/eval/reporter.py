"""Report writer for predictions, metrics, comparison CSVs, and equity curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.metrics import evaluate_classification, evaluate_model


def generate_report(
    y_true,
    y_pred,
    today_close,
    model_name: str,
    output_dir: str | Path,
    meta=None,
    y_true_return=None,
    y_pred_return=None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = build_prediction_frame(
        y_true=y_true,
        y_pred=y_pred,
        today_close=today_close,
        meta=meta,
        y_true_return=y_true_return,
        y_pred_return=y_pred_return,
    )
    frame.to_csv(output / "predictions.csv", index=False)
    metrics = evaluate_model(y_true, y_pred, today_close, periods_per_year=periods_per_year)
    pred_return = frame["y_pred_return"].to_numpy(dtype=float) if "y_pred_return" in frame else np.asarray(y_pred, dtype=float)
    metrics["pred_return_constant_flag"] = bool(np.nanstd(pred_return) < 1e-12)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(output / "metrics.csv", index=False)
    plot_equity(frame, output / "equity_curve.png", model_name)
    return metrics


def build_prediction_frame(
    y_true,
    y_pred,
    today_close,
    meta=None,
    y_true_return=None,
    y_pred_return=None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    today_close = np.asarray(today_close, dtype=float).reshape(-1)

    actual_return = y_true / today_close - 1.0
    pred_return = y_pred / today_close - 1.0
    if y_true_return is not None:
        actual_return = np.asarray(y_true_return, dtype=float).reshape(-1)
    if y_pred_return is not None:
        pred_return = np.asarray(y_pred_return, dtype=float).reshape(-1)
    pred_dir = y_pred > today_close
    actual_dir = y_true > today_close
    strategy_return = np.where(pred_dir, actual_return, -actual_return)
    frame = pd.DataFrame(
        {
            "today_close": today_close,
            "y_true_return": actual_return,
            "y_pred_return": pred_return,
            "actual_price": y_true,
            "pred_price": y_pred,
            "predicted_change": y_pred - today_close,
            "actual_direction": np.where(actual_dir, "UP", "DOWN"),
            "pred_direction": np.where(pred_dir, "UP", "DOWN"),
            "actual_label": actual_dir.astype(int),
            "predicted_label": pred_dir.astype(int),
            "direction_correct": (actual_dir == pred_dir).astype(int),
            "actual_return": actual_return,
            "strategy_return": strategy_return,
        }
    )
    if meta is not None:
        meta_frame = pd.DataFrame(meta).reset_index(drop=True)
        for col in ["series_id", "date", "target_date", "horizon"]:
            if col in meta_frame:
                if col in {"date", "target_date"}:
                    frame[col] = pd.to_datetime(meta_frame[col]).dt.strftime("%Y-%m-%d")
                else:
                    frame[col] = meta_frame[col].to_numpy()
        if "date" in frame:
            frame = frame.rename(columns={"date": "today_date"})
        ordered = [
            "series_id",
            "today_date",
            "target_date",
            "horizon",
            "today_close",
            "y_true_return",
            "y_pred_return",
            "actual_price",
            "pred_price",
            "predicted_change",
            "actual_direction",
            "pred_direction",
            "actual_label",
            "predicted_label",
            "direction_correct",
            "actual_return",
            "strategy_return",
        ]
        frame = frame[[c for c in ordered if c in frame.columns]]

    frame["equity"] = (1 + frame["strategy_return"]).cumprod()
    return frame


def plot_equity(frame: pd.DataFrame, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(frame["today_date"]) if "today_date" in frame else np.arange(len(frame))
    ax.plot(x, frame["equity"], label="strategy")
    ax.set_title(f"{title} equity curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_classification_report(
    y_true,
    prob,
    model_name: str,
    output_dir: str | Path,
    meta=None,
    threshold: float = 0.5,
    threshold_rule: str | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = build_classification_frame(y_true, prob, meta=meta, threshold=threshold)
    frame.to_csv(output / "predictions.csv", index=False)
    metrics = evaluate_classification(y_true, prob, threshold=threshold, threshold_rule=threshold_rule)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(output / "metrics.csv", index=False)
    return metrics


def build_classification_frame(y_true, prob, meta=None, threshold: float = 0.5) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    pred = (prob >= threshold).astype(int)
    frame = pd.DataFrame(
        {
            "target": y_true,
            "probability": prob,
            "threshold": float(threshold),
            "pred_label": pred,
            "correct": (pred == y_true).astype(int),
        }
    )
    if meta is not None:
        meta_frame = pd.DataFrame(meta).reset_index(drop=True)
        for col in ["series_id", "date", "target_date", "horizon"]:
            if col in meta_frame:
                if col in {"date", "target_date"}:
                    frame[col] = pd.to_datetime(meta_frame[col]).dt.strftime("%Y-%m-%d")
                else:
                    frame[col] = meta_frame[col].to_numpy()
        if "date" in frame:
            frame = frame.rename(columns={"date": "input_end_date"})
        ordered = [
            "series_id",
            "input_end_date",
            "target_date",
            "horizon",
            "target",
            "probability",
            "threshold",
            "pred_label",
            "correct",
        ]
        frame = frame[[c for c in ordered if c in frame.columns]]
    return frame
