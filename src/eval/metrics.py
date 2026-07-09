"""Metrics copied from Time-Series-Library utils/metrics.py and extended for trading."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import nn


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / np.where(true == 0, np.nan, true)))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / np.where(true == 0, np.nan, true)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.where(peak == 0, 1.0, peak) - 1.0
    return float(np.nanmin(drawdown))


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    std = np.nanstd(returns)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.nanmean(returns) / std * np.sqrt(periods_per_year))


def evaluate_model(y_true, y_pred, today_close, periods_per_year: int = 252) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    today_close = np.asarray(today_close, dtype=float).reshape(-1)
    true_dir = (y_true > today_close).astype(int)
    pred_dir = (y_pred > today_close).astype(int)
    pred_score = y_pred - today_close
    actual_return = y_true / today_close - 1.0
    position = np.where(pred_dir == 1, 1.0, -1.0)
    strategy_return = position * actual_return
    gross_profit = strategy_return[strategy_return > 0].sum()
    gross_loss = np.abs(strategy_return[strategy_return < 0].sum())
    equity = np.cumprod(1.0 + np.nan_to_num(strategy_return, nan=0.0))
    nonzero = y_true != 0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    out = {
        "samples": float(len(y_true)),
        "mae": float(MAE(y_pred, y_true)),
        "rmse": float(RMSE(y_pred, y_true)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else 0.0,
        "mape": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) if np.any(nonzero) else float("nan"),
        "direction_accuracy": float(np.mean(true_dir == pred_dir)),
        "balanced_accuracy": float(balanced_accuracy_score(true_dir, pred_dir)),
        "precision_up": float(precision_score(true_dir, pred_dir, zero_division=0)),
        "recall_up": float(recall_score(true_dir, pred_dir, zero_division=0)),
        "f1_up": float(f1_score(true_dir, pred_dir, zero_division=0)),
        "actual_up_count": float((true_dir == 1).sum()),
        "predicted_up_count": float((pred_dir == 1).sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "sharpe_ratio": sharpe_ratio(strategy_return, periods_per_year=periods_per_year),
        "win_rate": float(np.mean(strategy_return > 0)),
        "max_drawdown": max_drawdown(equity),
        "pred_up_rate": float(np.mean(pred_dir == 1)),
        "actual_up_rate": float(np.mean(true_dir == 1)),
        "pred_constant_flag": bool(np.nanstd(y_pred) < 1e-12),
    }
    try:
        out["auc_from_predicted_change"] = float(roc_auc_score(true_dir, pred_score))
    except ValueError:
        out["auc_from_predicted_change"] = float("nan")
    return out


def evaluate_classification(y_true, prob, threshold: float = 0.5, threshold_rule: str | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    pred = (prob >= threshold).astype(int)
    return evaluate_classification_predictions(y_true, pred, prob, threshold=threshold, threshold_rule=threshold_rule)


def evaluate_classification_predictions(
    y_true,
    pred,
    prob,
    threshold: float | None = None,
    threshold_rule: str | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    pred = np.asarray(pred, dtype=int).reshape(-1)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    out = {
        "threshold": None if threshold is None else float(threshold),
        "threshold_rule": threshold_rule,
        "n": int(len(y_true)),
        "class_0_count": int((y_true == 0).sum()),
        "class_1_count": int((y_true == 1).sum()),
        "accuracy": float(np.mean(y_true == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) == 2 else float("nan"),
        "precision_weighted": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "precision_positive": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    else:
        out["auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def best_classification_threshold(y_true, prob) -> dict:
    rows = [evaluate_classification(y_true, prob, threshold=float(threshold)) for threshold in np.linspace(0.05, 0.95, 91)]
    return max(rows, key=lambda row: (row["f1_weighted"], safe_metric(row["balanced_accuracy"]), row["f1_positive"]))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def logit_np(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(prob / (1.0 - prob))


def fit_positive_platt(logits: np.ndarray, y: np.ndarray, l2: float = 1e-3, max_iter: int = 100) -> dict[str, float | str]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(np.unique(y.astype(int))) < 2:
        return {"method": "raw_fallback_single_class_validation", "a": 1.0, "b": 0.0}

    z = torch.tensor(logits, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    log_a = torch.zeros((), dtype=torch.float32, requires_grad=True)
    prior = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    bias_init = np.log(prior / (1.0 - prior))
    b = torch.tensor(bias_init, dtype=torch.float32, requires_grad=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([log_a, b], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        a = torch.exp(log_a)
        calibrated_logits = a * z + b
        loss = criterion(calibrated_logits, target) + l2 * (log_a.square() + b.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"method": "positive_platt", "a": float(torch.exp(log_a).detach().cpu()), "b": float(b.detach().cpu())}


def apply_platt(logits: np.ndarray, calibrator: dict[str, float | str]) -> np.ndarray:
    a = float(calibrator["a"])
    b = float(calibrator["b"])
    return sigmoid_np(a * np.asarray(logits, dtype=float) + b)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        mask = (prob >= lo) & (prob <= hi) if idx == n_bins - 1 else (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def probability_diagnostics(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_brier": float(brier_score_loss(y_true, prob)),
        f"{prefix}_ece_10bins": expected_calibration_error(y_true, prob, n_bins=10),
        f"{prefix}_prob_mean": float(np.mean(prob)),
        f"{prefix}_prob_median": float(np.median(prob)),
    }


def safe_metric(value) -> float:
    value = float(value)
    return value if np.isfinite(value) else -1.0


def compare_models(results_dict: dict[str, dict[str, float]]):
    import pandas as pd

    if results_dict and any("accuracy" in metrics for metrics in results_dict.values()):
        order = [
            "accuracy",
            "balanced_accuracy",
            "auc",
            "average_precision",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "precision_positive",
            "recall_positive",
            "f1_positive",
            "threshold",
            "class_0_count",
            "class_1_count",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        rows = []
        for model, metrics in results_dict.items():
            row = {"model": model}
            row.update({k: metrics.get(k) for k in order})
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1_weighted"], ascending=[False, False])

    order = [
        "direction_accuracy",
        "balanced_accuracy",
        "auc_from_predicted_change",
        "precision_up",
        "recall_up",
        "f1_up",
        "profit_factor",
        "sharpe_ratio",
        "win_rate",
        "max_drawdown",
        "pred_up_rate",
        "actual_up_rate",
        "pred_constant_flag",
        "pred_return_constant_flag",
        "rmse",
        "mae",
        "r2",
    ]
    rows = []
    for model, metrics in results_dict.items():
        row = {"model": model}
        row.update({k: metrics.get(k) for k in order})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction_accuracy", "profit_factor"], ascending=[False, False])
