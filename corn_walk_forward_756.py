from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = {
    "5td": ("target_corn_ret_fwd_5td", 5),
    "10td": ("target_corn_ret_fwd_10td", 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward corn futures return forecast.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-window", type=int, default=756)
    return parser.parse_args()


def max_drawdown(equity: pd.Series) -> tuple[pd.Series, float]:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown, float(drawdown.min())


def build_model(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    prep = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", prep), ("model", Ridge(alpha=5.0))])


def walk_forward(df: pd.DataFrame, target: str, horizon: int, train_window: int) -> pd.DataFrame:
    base_feature_cols = [c for c in df.columns if c not in ["date", *TARGETS["5td"][:1], *TARGETS["10td"][:1]]]
    rows = []
    last_test_idx = df[target].last_valid_index()
    if last_test_idx is None:
        raise ValueError(f"{target} has no valid labels")

    start_idx = train_window + horizon - 1
    for i in range(start_idx, last_test_idx + 1):
        train_end = i - horizon
        train_start = train_end - train_window + 1
        if train_start < 0:
            continue

        train = df.iloc[train_start : train_end + 1].dropna(subset=[target]).copy()
        if len(train) < train_window * 0.9:
            continue

        usable_cols = [c for c in base_feature_cols if not train[c].isna().all()]
        numeric_cols = [c for c in usable_cols if pd.api.types.is_numeric_dtype(train[c])]
        categorical_cols = [c for c in usable_cols if c not in numeric_cols]

        model = build_model(numeric_cols, categorical_cols)
        model.fit(train[usable_cols], train[target])
        pred = float(model.predict(df.loc[[i], usable_cols])[0])
        actual = float(df.loc[i, target])
        position = 1 if pred > 0 else -1
        rows.append(
            {
                "date": df.loc[i, "date"],
                "close": df.loc[i, "dce_corn_close"],
                "prediction_horizon": horizon,
                "train_start": df.loc[train_start, "date"],
                "train_end": df.loc[train_end, "date"],
                "predicted_return": pred,
                "actual_return": actual,
                "predicted_direction": "UP" if pred > 0 else "DOWN",
                "actual_direction": "UP" if actual > 0 else "DOWN",
                "direction_correct": int(np.sign(pred) == np.sign(actual)),
                "position": position,
                "strategy_return": position * actual,
            }
        )

    result = pd.DataFrame(rows)
    result["equity"] = (1 + result["strategy_return"]).cumprod()
    result["buy_hold_target_equity"] = (1 + result["actual_return"]).cumprod()
    result["drawdown"], _ = max_drawdown(result["equity"])
    return result


def summarize(result: pd.DataFrame, label: str) -> dict[str, object]:
    equity = result["equity"]
    drawdown, mdd = max_drawdown(equity)
    up = result["predicted_return"] > 0
    down = result["predicted_return"] < 0
    return {
        "horizon": label,
        "predictions": len(result),
        "start_date": result["date"].min().date(),
        "end_date": result["date"].max().date(),
        "direction_win_rate": result["direction_correct"].mean(),
        "mae": mean_absolute_error(result["actual_return"], result["predicted_return"]),
        "rmse": mean_squared_error(result["actual_return"], result["predicted_return"]) ** 0.5,
        "mean_strategy_return_per_forecast": result["strategy_return"].mean(),
        "median_strategy_return_per_forecast": result["strategy_return"].median(),
        "final_equity_forecast_level": equity.iloc[-1],
        "max_drawdown_forecast_level": mdd,
        "pred_up_rate": up.mean(),
        "up_signal_win_rate": result.loc[up, "direction_correct"].mean() if up.any() else np.nan,
        "down_signal_win_rate": result.loc[down, "direction_correct"].mean() if down.any() else np.nan,
    }


def plot_curves(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for label, result in outputs.items():
        axes[0].plot(result["date"], result["equity"], label=f"{label} strategy")
        axes[0].plot(result["date"], result["buy_hold_target_equity"], linestyle="--", alpha=0.6, label=f"{label} long-only target")
        axes[1].plot(result["date"], result["drawdown"], label=f"{label} drawdown")

    axes[0].set_title("Walk-forward forecast-level equity curve")
    axes[0].set_ylabel("Equity, compounded per forecast")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "walk_forward_equity_drawdown.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    outputs = {}
    summaries = []
    for label, (target, horizon) in TARGETS.items():
        result = walk_forward(df, target, horizon, args.train_window)
        outputs[label] = result
        summaries.append(summarize(result, label))
        result.to_csv(args.out_dir / f"walk_forward_predictions_{label}.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "walk_forward_summary.csv", index=False)
    plot_curves(outputs, args.out_dir)

    report = [
        "# Walk-forward 756 Trading Day Forecast",
        "",
        f"Data range: {df['date'].min().date()} to {df['date'].max().date()}, rows: {len(df):,}.",
        f"Training window: {args.train_window} trading days. Test step: 1 trading day.",
        "Model: Ridge regression with rolling refit; labels become eligible only after the forecast horizon has passed.",
        "",
        "## Summary",
        summary.to_string(index=False),
        "",
        "Outputs:",
        "- walk_forward_predictions_5td.csv",
        "- walk_forward_predictions_10td.csv",
        "- walk_forward_summary.csv",
        "- walk_forward_equity_drawdown.png",
    ]
    (args.out_dir / "walk_forward_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
