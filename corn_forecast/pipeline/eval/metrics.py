"""Benchmark metric calculations with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


def compute_strategy_returns(predictions: np.ndarray, actual_returns: np.ndarray) -> np.ndarray:
    """Convert binary predictions to long/short strategy returns."""

    positions = np.where(predictions == 1, 1.0, -1.0)
    return positions * actual_returns


def max_drawdown(equity: np.ndarray) -> float:
    """Compute maximum peak-to-trough drawdown."""

    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(drawdown))


def sharpe_ratio(strategy_returns: np.ndarray, annualize: int = 12) -> float:
    """Annualized Sharpe ratio."""

    std = np.std(strategy_returns)
    if std < 1e-12:
        return 0.0
    return float(np.mean(strategy_returns) / std * np.sqrt(annualize))


def sortino_ratio(strategy_returns: np.ndarray, annualize: int = 12) -> float:
    """Annualized Sortino ratio using downside deviation."""

    downside = strategy_returns[strategy_returns < 0]
    mean_ret = float(np.mean(strategy_returns))
    if len(downside) == 0:
        return float("inf") if mean_ret > 1e-12 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-12:
        return float("inf") if mean_ret > 1e-12 else 0.0
    return float(mean_ret / downside_std * np.sqrt(annualize))


def profit_factor(strategy_returns: np.ndarray) -> float:
    """Gross profit divided by gross loss."""

    pos = strategy_returns[strategy_returns > 0].sum()
    neg = abs(strategy_returns[strategy_returns < 0].sum())
    if neg < 1e-12:
        return float("inf") if pos > 1e-12 else 1.0
    return float(pos / neg)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True negative rate."""

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    return float(tn / (tn + fp)) if tn + fp > 0 else 0.0


def negative_predictive_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative predictive value."""

    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return float(tn / (tn + fn)) if tn + fn > 0 else 0.0


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC with single-class fallback."""

    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision with single-class fallback."""

    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return 0.5


def safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary log loss with clipping and single-class labels."""

    try:
        return float(log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15), labels=[0, 1]))
    except ValueError:
        return 0.0


def finite_or_value(value: float, digits: int = 4) -> float:
    """Round finite values and preserve infinities."""

    return round(value, digits) if np.isfinite(value) else value


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Compute bootstrap confidence intervals for DirAcc and Sharpe."""

    rng = np.random.RandomState(random_seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    returns = np.asarray(returns, dtype=float)
    n = len(y_true)
    if not (n == len(y_pred) == len(y_prob) == len(returns)):
        raise ValueError("All metric arrays must have the same length")
    if n_bootstrap <= 0:
        strategy_returns = compute_strategy_returns(y_pred, returns)
        dir_acc = round(float(np.mean(y_pred == y_true)), 4)
        sharpe = round(sharpe_ratio(strategy_returns), 4)
        return {
            "DirAcc_CI": [dir_acc, dir_acc],
            "Sharpe_CI": [sharpe, sharpe],
            "n_bootstrap": int(n_bootstrap),
            "ci_level": ci_level,
        }

    strategy_returns = compute_strategy_returns(y_pred, returns)
    dir_acc_boot = np.zeros(n_bootstrap)
    sharpe_boot = np.zeros(n_bootstrap)
    for idx in range(n_bootstrap):
        sample_idx = rng.randint(0, n, size=n)
        dir_acc_boot[idx] = np.mean(y_pred[sample_idx] == y_true[sample_idx])
        sharpe_boot[idx] = sharpe_ratio(strategy_returns[sample_idx])

    alpha = (1.0 - ci_level) / 2.0
    return {
        "DirAcc_CI": [
            round(float(np.percentile(dir_acc_boot, 100 * alpha)), 4),
            round(float(np.percentile(dir_acc_boot, 100 * (1.0 - alpha))), 4),
        ],
        "Sharpe_CI": [
            round(float(np.percentile(sharpe_boot, 100 * alpha)), 4),
            round(float(np.percentile(sharpe_boot, 100 * (1.0 - alpha))), 4),
        ],
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    actual_returns: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    annualize: int = 12,
    raw_reg_pred: np.ndarray | None = None,
    naive_mean_ret_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute classification, trading, calibration, health, and CI metrics."""

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    y_pred = (y_prob > 0.5).astype(int)
    strategy_returns = compute_strategy_returns(y_pred, actual_returns)
    equity = np.cumprod(1.0 + strategy_returns)
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    ci = compute_bootstrap_ci(
        y_true,
        y_pred,
        y_prob,
        actual_returns,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )
    max_dd = max_drawdown(equity)
    ann_ret = float(np.mean(strategy_returns) * annualize)
    if abs(max_dd) < 1e-12:
        calmar = float("inf") if ann_ret > 1e-12 else 0.0
    else:
        calmar = ann_ret / abs(max_dd)

    result = {
        "DirAcc": round(float(np.mean(y_pred == y_true)), 4),
        "BalancedAcc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "AUC": round(safe_auc(y_true, y_prob), 4),
        "AP": round(safe_ap(y_true, y_prob), 4),
        "Sharpe": round(sharpe_ratio(strategy_returns, annualize), 4),
        "Sortino": finite_or_value(sortino_ratio(strategy_returns, annualize)),
        "Calmar": finite_or_value(calmar),
        "AnnRet": round(ann_ret, 4),
        "ProfitFactor": finite_or_value(profit_factor(strategy_returns)),
        "WinRate": round(float(np.mean(strategy_returns > 0)), 4),
        "MaxDD": round(max_dd, 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "Specificity": round(specificity(y_true, y_pred), 4),
        "NPV": round(negative_predictive_value(y_true, y_pred), 4),
        "Brier": round(float(np.mean((y_prob - y_true.astype(float)) ** 2)), 4),
        "LogLoss": round(safe_log_loss(y_true, y_prob), 4),
        "Expectancy": round(float(np.mean(strategy_returns)), 4),
        "AvgWin": round(float(np.mean(wins)) if len(wins) else 0.0, 4),
        "AvgLoss": round(float(abs(np.mean(losses))) if len(losses) else 0.0, 4),
        "pred_constant_flag": bool(np.std(y_pred) < 1e-12),
    }
    if raw_reg_pred is not None:
        raw_reg = np.asarray(raw_reg_pred, dtype=float)
        if len(raw_reg) != len(actual_returns):
            raise ValueError("raw_reg_pred must have the same length as actual_returns")
        if naive_mean_ret_pred is not None:
            naive = np.asarray(naive_mean_ret_pred, dtype=float)
            denom = float(np.sum((actual_returns - naive) ** 2))
            if denom < 1e-12:
                r2_health = 0.0
            else:
                r2_health = 1.0 - float(np.sum((actual_returns - raw_reg) ** 2)) / denom
        else:
            r2_health = float(r2_score(actual_returns, raw_reg))
        result["R2_health"] = round(r2_health, 4)
    else:
        result["R2_health"] = None
    result.update(ci)
    return result
