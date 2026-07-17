from __future__ import annotations

from collections.abc import Sequence
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


EPSILON = 1e-12


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > EPSILON else np.nan


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(actual, predicted))


def _economic_metrics(returns: np.ndarray, horizon: int) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return {}
    periods_per_year = 252.0 / horizon
    cumulative = float(np.prod(1.0 + returns) - 1.0)
    annualized = (
        float((1.0 + cumulative) ** (periods_per_year / returns.size) - 1.0)
        if cumulative > -1.0
        else -1.0
    )
    volatility = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = _safe_divide(float(np.mean(returns)) * np.sqrt(periods_per_year), volatility)
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(wealth)
    max_drawdown = float(abs(np.min(wealth / peaks - 1.0)))
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _non_overlapping_economics(
    frame: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: Sequence[int],
    trend_column: str | None = None,
    prefix: str | None = None,
) -> dict[str, float]:
    if trend_column is None:
        trend_column = (
            "predicted_trend_selected"
            if "predicted_trend_selected" in frame
            else "predicted_trend"
        )
    result: dict[str, float] = {}
    label = f"{prefix}_" if prefix else ""
    for cost_bps in transaction_cost_bps:
        sleeve_metrics: list[dict[str, float]] = []
        for sleeve in range(horizon):
            subset = frame.iloc[sleeve::horizon]
            if subset.empty:
                continue
            position = np.where(subset[trend_column].to_numpy() == 1, 1.0, -1.0)
            turnover = np.empty(position.size, dtype=float)
            turnover[0] = 1.0
            if position.size > 1:
                turnover[1:] = np.abs(np.diff(position)) / 2.0
            strategy_returns = (
                position * subset["actual_return"].to_numpy(dtype=float)
                - turnover * float(cost_bps) / 10000.0
            )
            sleeve_metrics.append(_economic_metrics(strategy_returns, horizon))
        if not sleeve_metrics:
            continue
        for name in sleeve_metrics[0]:
            values = np.asarray([item[name] for item in sleeve_metrics], dtype=float)
            finite = values[np.isfinite(values)]
            base = f"economic_{label}{cost_bps}bp"
            result[f"{base}_mean_{name}"] = (
                float(np.mean(finite)) if finite.size else np.nan
            )
            result[f"{base}_median_{name}"] = (
                float(np.median(finite)) if finite.size else np.nan
            )
            result[f"{base}_worst_{name}"] = (
                float(np.max(finite) if name == "max_drawdown" else np.min(finite))
                if finite.size
                else np.nan
            )
    return result


def select_trend_threshold(
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
    candidates: Sequence[float],
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    actual_return = np.asarray(actual_return, dtype=float)
    predicted_return = np.asarray(predicted_return, dtype=float)
    actual_trend = (actual_return > 0.0).astype(int)
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        predicted_trend = (predicted_return > float(threshold)).astype(int)
        rows.append(
            {
                "candidate_threshold": float(threshold),
                "validation_balanced_accuracy": _balanced_accuracy(
                    actual_trend, predicted_trend
                ),
                "validation_mcc": float(
                    matthews_corrcoef(actual_trend, predicted_trend)
                ),
                "validation_predicted_up_rate": float(np.mean(predicted_trend)),
            }
        )
    selected_row = min(
        rows,
        key=lambda row: (
            -row["validation_balanced_accuracy"],
            -row["validation_mcc"],
            abs(row["candidate_threshold"]),
        ),
    )
    selected = float(selected_row["candidate_threshold"])
    for row in rows:
        row["is_selected"] = float(
            np.isclose(row["candidate_threshold"], selected)
        )
        row["selection_rule"] = (
            "balanced_accuracy,higher_mcc,closer_to_zero"
        )
    threshold_zero = min(rows, key=lambda row: abs(row["candidate_threshold"]))
    summary = {
        "validation_selected_threshold": selected,
        "validation_ba_threshold_0": threshold_zero[
            "validation_balanced_accuracy"
        ],
        "validation_ba_selected": selected_row["validation_balanced_accuracy"],
        "validation_mcc_selected": selected_row["validation_mcc"],
        "validation_actual_up_rate": float(np.mean(actual_trend)),
        "validation_predicted_up_rate_selected": selected_row[
            "validation_predicted_up_rate"
        ],
    }
    return selected, summary, rows


def _trend_metrics(
    actual_trend: np.ndarray,
    predicted_trend: np.ndarray,
    predicted_return: np.ndarray,
    label: str,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(
        actual_trend, predicted_trend, labels=[0, 1]
    ).ravel()
    ap = (
        float(average_precision_score(actual_trend, predicted_return))
        if np.unique(actual_trend).size > 1
        else np.nan
    )
    return {
        f"trend_{label}_direction_accuracy": float(
            np.mean(actual_trend == predicted_trend)
        ),
        f"trend_{label}_balanced_accuracy": _balanced_accuracy(
            actual_trend, predicted_trend
        ),
        f"trend_{label}_precision": float(
            precision_score(actual_trend, predicted_trend, zero_division=0)
        ),
        f"trend_{label}_recall": float(
            recall_score(actual_trend, predicted_trend, zero_division=0)
        ),
        f"trend_{label}_ap": ap,
        f"trend_{label}_f1": float(
            f1_score(actual_trend, predicted_trend, zero_division=0)
        ),
        f"trend_{label}_mcc": float(
            matthews_corrcoef(actual_trend, predicted_trend)
        ),
        f"trend_{label}_tn": float(tn),
        f"trend_{label}_fp": float(fp),
        f"trend_{label}_fn": float(fn),
        f"trend_{label}_tp": float(tp),
        f"trend_{label}_predicted_up_rate": float(np.mean(predicted_trend)),
        f"trend_{label}_constant_prediction": float(
            np.unique(predicted_trend).size == 1
        ),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    horizon: int,
    mase_scale: float | None = None,
    transaction_cost_bps: Sequence[int] = (0, 2, 5, 10),
    selected_threshold: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    del mase_scale
    frame = predictions.copy()
    actual = frame["actual_dce_corn_close"].to_numpy(dtype=float)
    predicted = frame["predicted_dce_corn_close"].to_numpy(dtype=float)
    close_t = frame["close_t"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Price arrays contain non-finite values")

    frame["actual_return"] = actual / close_t - 1.0
    frame["predicted_return"] = predicted / close_t - 1.0
    frame["actual_trend"] = (frame["actual_return"] > 0.0).astype(int)
    if selected_threshold is not None:
        frame["selected_threshold"] = float(selected_threshold)
    elif "selected_threshold" not in frame:
        frame["selected_threshold"] = 0.0
    frame["predicted_trend_threshold_0"] = (
        frame["predicted_return"] > 0.0
    ).astype(int)
    frame["predicted_trend_selected"] = (
        frame["predicted_return"] > frame["selected_threshold"]
    ).astype(int)

    actual_trend = frame["actual_trend"].to_numpy(dtype=int)
    threshold_zero = frame["predicted_trend_threshold_0"].to_numpy(dtype=int)
    selected = frame["predicted_trend_selected"].to_numpy(dtype=int)
    predicted_return = frame["predicted_return"].to_numpy(dtype=float)
    threshold_zero_metrics = _trend_metrics(
        actual_trend, threshold_zero, predicted_return, "threshold_0"
    )
    selected_metrics = _trend_metrics(
        actual_trend, selected, predicted_return, "selected"
    )
    metrics: dict[str, float] = {
        "price_mae": float(mean_absolute_error(actual, predicted)),
        "price_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "price_r2": float(r2_score(actual, predicted)),
        "test_ba_threshold_0": threshold_zero_metrics[
            "trend_threshold_0_balanced_accuracy"
        ],
        "test_ba_selected": selected_metrics["trend_selected_balanced_accuracy"],
        "test_mcc_selected": selected_metrics["trend_selected_mcc"],
        "test_actual_up_rate": float(np.mean(actual_trend)),
        "test_predicted_up_rate_selected": float(np.mean(selected)),
        "trend_direction_accuracy": selected_metrics[
            "trend_selected_direction_accuracy"
        ],
        "trend_balanced_accuracy": selected_metrics[
            "trend_selected_balanced_accuracy"
        ],
        "trend_mcc": selected_metrics["trend_selected_mcc"],
        **threshold_zero_metrics,
        **selected_metrics,
    }
    metrics.update(
        _non_overlapping_economics(
            frame,
            horizon,
            transaction_cost_bps,
            trend_column="predicted_trend_threshold_0",
            prefix="threshold_0",
        )
    )
    metrics.update(
        _non_overlapping_economics(
            frame,
            horizon,
            transaction_cost_bps,
            trend_column="predicted_trend_selected",
            prefix="selected",
        )
    )
    return metrics, frame
