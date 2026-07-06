"""Experiment report writer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_experiment_report(
    *,
    output_dir: str | Path,
    model_name: str,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    metrics: dict,
    verdict: dict,
    config: dict | None = None,
    write_model_output: bool = True,
) -> None:
    """Write predictions, metrics, comparison, chart, markdown, and verdict."""

    out = Path(output_dir)
    if write_model_output:
        write_model_outputs(out, model_name, predictions, metrics)
    comparison.to_csv(out / "comparison.csv", index=False)
    (out / "agent_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    if config is not None:
        (out / "config_resolved.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# Experiment Report",
        "",
        f"Model: `{model_name}`",
        "",
        "## Comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"Status: `{verdict['status']}`",
        "",
        "Backtest results are research evidence and do not promise live trading profit.",
        "",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")


def write_model_outputs(output_dir: str | Path, model_name: str, predictions: pd.DataFrame, metrics: dict) -> None:
    """Write one model's predictions, rolling metrics, summary, and chart."""

    model_out = Path(output_dir) / "model_outputs" / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(model_out / "predictions.csv", index=False)
    rolling_metrics = build_rolling_metrics(predictions)
    rolling_metrics.to_csv(model_out / "rolling_metrics.csv", index=False)
    (model_out / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_equity_curve(predictions, model_out / "equity_curve.png", model_name)
    write_rolling_chart(
        rolling_metrics,
        model_out / "rolling_dir_acc.png",
        y_col="rolling_12_dir_acc",
        title=f"{model_name} rolling direction accuracy",
        ylabel="Rolling DirAcc",
        reference=0.5,
    )
    write_rolling_chart(
        rolling_metrics,
        model_out / "rolling_sharpe.png",
        y_col="rolling_12_sharpe",
        title=f"{model_name} rolling Sharpe",
        ylabel="Rolling Sharpe",
        reference=0.0,
    )


def build_rolling_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative and rolling diagnostics for a prediction table."""

    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "date",
                "window_id",
                "cumulative_dir_acc",
                "cumulative_return",
                "rolling_12_dir_acc",
                "rolling_12_sharpe",
            ]
        )
    ordered = predictions.reset_index(drop=True).copy()
    direction = ordered["direction_correct"].astype(float)
    strategy_returns = ordered["strategy_return"].astype(float)
    equity = (1.0 + strategy_returns).cumprod()
    rolling_mean = strategy_returns.rolling(window=12, min_periods=1).mean()
    rolling_std = strategy_returns.rolling(window=12, min_periods=2).std(ddof=0)
    rolling_sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan) * np.sqrt(12.0)).fillna(0.0)
    return pd.DataFrame(
        {
            "step": range(1, len(ordered) + 1),
            "date": ordered["date"],
            "window_id": ordered["window_id"],
            "cumulative_dir_acc": direction.expanding().mean(),
            "cumulative_return": equity - 1.0,
            "rolling_12_dir_acc": direction.rolling(window=12, min_periods=1).mean(),
            "rolling_12_sharpe": rolling_sharpe,
        }
    )


def write_equity_curve(predictions: pd.DataFrame, path: str | Path, title: str) -> None:
    """Write an equity curve plot."""

    fig, ax = plt.subplots(figsize=(8, 4))
    x = pd.to_datetime(predictions["date"]) if "date" in predictions else range(len(predictions))
    ax.plot(x, predictions["equity"], label="strategy")
    if "actual_return" in predictions:
        buy_hold_equity = (1.0 + predictions["actual_return"].astype(float)).cumprod()
        ax.plot(x, buy_hold_equity, label="buy-and-hold")
    ax.set_title(f"{title} equity curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_rolling_chart(
    rolling_metrics: pd.DataFrame,
    path: str | Path,
    *,
    y_col: str,
    title: str,
    ylabel: str,
    reference: float | None = None,
) -> None:
    """Write a rolling diagnostic line chart."""

    fig, ax = plt.subplots(figsize=(8, 4))
    if not rolling_metrics.empty and y_col in rolling_metrics:
        x = pd.to_datetime(rolling_metrics["date"]) if "date" in rolling_metrics else rolling_metrics["step"]
        ax.plot(x, rolling_metrics[y_col], label=ylabel)
    if reference is not None:
        ax.axhline(reference, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
