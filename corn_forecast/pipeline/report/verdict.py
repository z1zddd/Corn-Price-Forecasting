"""Agent-readable verdicts."""

from __future__ import annotations


def build_agent_verdict(metrics: dict, *, baseline_metrics: dict | None, primary_metric: str) -> dict:
    """Build a conservative machine-readable verdict."""

    warnings: list[str] = []
    next_actions: list[str] = []
    primary_value = float(metrics.get(primary_metric, metrics.get("DirAcc", 0.0)))
    baseline_value = float((baseline_metrics or {}).get(primary_metric, (baseline_metrics or {}).get("DirAcc", 0.0)))
    ci = metrics.get("DirAcc_CI", [primary_value, primary_value])
    ci_width = float(ci[1] - ci[0]) if isinstance(ci, list) and len(ci) == 2 else 0.0

    if metrics.get("pred_constant_flag"):
        warnings.append("Prediction is constant; model signal is invalid.")
        next_actions.append("Inspect target balance and feature variation.")
        status = "invalid"
        passed = False
    elif primary_value > baseline_value and ci_width <= 0.20 and float(metrics.get("Sharpe", 0.0)) > 0:
        status = "signal"
        passed = True
    elif primary_value > baseline_value:
        warnings.append("Primary metric is above baseline but uncertainty remains wide or Sharpe is weak.")
        next_actions.append("Run more windows or compare against additional baselines.")
        status = "weak_signal"
        passed = False
    else:
        warnings.append("Primary metric does not beat baseline.")
        next_actions.append("Review features, target horizon, and window mode.")
        status = "no_signal"
        passed = False

    return {
        "pass": passed,
        "status": status,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "baseline_value": baseline_value,
        "ci_width": ci_width,
        "warnings": warnings,
        "next_actions": next_actions,
    }
