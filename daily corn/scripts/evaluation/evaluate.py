from __future__ import annotations

from collections.abc import Sequence

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
    downside_deviation = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    sharpe = _safe_divide(float(np.mean(returns)) * np.sqrt(periods_per_year), volatility)
    sortino = _safe_divide(
        float(np.mean(returns)) * np.sqrt(periods_per_year), downside_deviation
    )
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    max_drawdown = float(abs(np.min(drawdowns)))
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = _safe_divide(float(gains.sum()), float(abs(losses.sum())))
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": _safe_divide(annualized, max_drawdown),
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": float(np.mean(returns > 0)),
        "average_gain": float(gains.mean()) if gains.size else 0.0,
        "average_loss": float(losses.mean()) if losses.size else 0.0,
    }


def _non_overlapping_economics(
    frame: pd.DataFrame, horizon: int, transaction_cost_bps: Sequence[int]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for cost_bps in transaction_cost_bps:
        sleeve_metrics: list[dict[str, float]] = []
        for sleeve in range(horizon):
            subset = frame.iloc[sleeve::horizon]
            if subset.empty:
                continue
            position = np.where(subset["predicted_trend"].to_numpy() == 1, 1.0, -1.0)
            turnover = np.abs(np.diff(position, prepend=0.0))
            strategy_returns = (
                position * subset["actual_return"].to_numpy(dtype=float)
                - turnover * float(cost_bps) / 10000.0
            )
            metrics = _economic_metrics(strategy_returns, horizon)
            metrics["turnover"] = float(np.mean(turnover))
            metrics["trade_count"] = float(np.sum(turnover > 0))
            sleeve_metrics.append(metrics)
        if not sleeve_metrics:
            continue
        metric_names = sleeve_metrics[0].keys()
        for name in metric_names:
            values = np.asarray([item[name] for item in sleeve_metrics], dtype=float)
            finite = values[np.isfinite(values)]
            result[f"economic_{cost_bps}bp_mean_{name}"] = (
                float(np.mean(finite)) if finite.size else np.nan
            )
            result[f"economic_{cost_bps}bp_median_{name}"] = (
                float(np.median(finite)) if finite.size else np.nan
            )
            higher_is_worse = name in {"max_drawdown", "turnover", "trade_count"}
            result[f"economic_{cost_bps}bp_worst_{name}"] = (
                float(np.max(finite) if higher_is_worse else np.min(finite))
                if finite.size
                else np.nan
            )
    return result


def evaluate_predictions(
    predictions: pd.DataFrame,
    horizon: int,
    mase_scale: float,
    transaction_cost_bps: Sequence[int] = (0, 2, 5, 10),
) -> tuple[dict[str, float], pd.DataFrame]:
    frame = predictions.copy()
    actual = frame["actual_dce_corn_close"].to_numpy(dtype=float)
    predicted = frame["predicted_dce_corn_close"].to_numpy(dtype=float)
    close_t = frame["close_t"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Price arrays contain non-finite values")
    frame["actual_return"] = actual / close_t - 1.0
    frame["predicted_return"] = predicted / close_t - 1.0
    frame["actual_trend"] = (actual > close_t).astype(int)
    frame["predicted_trend"] = (predicted > close_t).astype(int)

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, close_t)))
    actual_trend = frame["actual_trend"].to_numpy(dtype=int)
    predicted_trend = frame["predicted_trend"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(actual_trend, predicted_trend, labels=[0, 1]).ravel()
    if np.unique(actual_trend).size > 1:
        ap = float(average_precision_score(actual_trend, frame["predicted_return"]))
    else:
        ap = np.nan
    metrics: dict[str, float] = {
        "price_mae": mae,
        "price_rmse": rmse,
        "price_mape": float(np.mean(np.abs((actual - predicted) / actual))),
        "price_smape": float(
            np.mean(2.0 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
        ),
        "price_r2": float(r2_score(actual, predicted)),
        "price_mase": float(
            np.mean(
                np.abs(actual - predicted)
                / np.where(
                    np.asarray(frame.get("mase_scale", mase_scale), dtype=float) > EPSILON,
                    np.asarray(frame.get("mase_scale", mase_scale), dtype=float),
                    np.nan,
                )
            )
        ),
        "price_baseline_rmse": baseline_rmse,
        "price_baseline_improvement": 1.0 - _safe_divide(rmse, baseline_rmse),
        "trend_direction_accuracy": float(np.mean(actual_trend == predicted_trend)),
        "trend_balanced_accuracy": float(
            balanced_accuracy_score(actual_trend, predicted_trend)
        ),
        "trend_precision": float(
            precision_score(actual_trend, predicted_trend, zero_division=0)
        ),
        "trend_recall": float(recall_score(actual_trend, predicted_trend, zero_division=0)),
        "trend_ap": ap,
        "trend_f1": float(f1_score(actual_trend, predicted_trend, zero_division=0)),
        "trend_mcc": float(matthews_corrcoef(actual_trend, predicted_trend)),
        "trend_tn": float(tn),
        "trend_fp": float(fp),
        "trend_fn": float(fn),
        "trend_tp": float(tp),
        "trend_actual_up_rate": float(np.mean(actual_trend)),
        "trend_predicted_up_rate": float(np.mean(predicted_trend)),
        "trend_constant_prediction": float(np.unique(predicted_trend).size == 1),
    }
    metrics.update(_non_overlapping_economics(frame, horizon, transaction_cost_bps))
    return metrics, frame
