import numpy as np

from corn_forecast.pipeline.eval.metrics import compute_all_metrics, compute_bootstrap_ci


def test_compute_all_metrics_contains_required_keys():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.4, 0.3])
    returns = np.array([0.02, -0.01, 0.03, -0.02])

    metrics = compute_all_metrics(y_true, y_prob, returns, n_bootstrap=50)

    required = {
        "DirAcc",
        "BalancedAcc",
        "AUC",
        "AP",
        "Sharpe",
        "Sortino",
        "Calmar",
        "AnnRet",
        "ProfitFactor",
        "WinRate",
        "MaxDD",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Specificity",
        "NPV",
        "Brier",
        "LogLoss",
        "Expectancy",
        "AvgWin",
        "AvgLoss",
        "DirAcc_CI",
        "Sharpe_CI",
    }
    assert required.issubset(metrics.keys())
    assert metrics["DirAcc"] == 0.75


def test_bootstrap_ci_is_deterministic():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.4, 0.3])
    returns = np.array([0.02, -0.01, 0.03, -0.02])

    ci1 = compute_bootstrap_ci(y_true, y_pred, y_prob, returns, n_bootstrap=50, random_seed=42)
    ci2 = compute_bootstrap_ci(y_true, y_pred, y_prob, returns, n_bootstrap=50, random_seed=42)

    assert ci1 == ci2
    assert len(ci1["DirAcc_CI"]) == 2
    assert len(ci1["Sharpe_CI"]) == 2


def test_compute_all_metrics_allows_zero_bootstrap():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.4, 0.3])
    returns = np.array([0.02, -0.01, 0.03, -0.02])

    metrics = compute_all_metrics(y_true, y_prob, returns, n_bootstrap=0)

    assert metrics["n_bootstrap"] == 0
    assert metrics["DirAcc_CI"] == [metrics["DirAcc"], metrics["DirAcc"]]


def test_compute_all_metrics_includes_r2_health_for_regression_head():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.1])
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    raw_reg_pred = np.array([0.018, -0.012, 0.025, -0.015])
    naive = np.full_like(returns, returns.mean())

    metrics = compute_all_metrics(
        y_true,
        y_prob,
        returns,
        n_bootstrap=20,
        raw_reg_pred=raw_reg_pred,
        naive_mean_ret_pred=naive,
    )

    assert "R2_health" in metrics
    assert metrics["R2_health"] > 0.0
