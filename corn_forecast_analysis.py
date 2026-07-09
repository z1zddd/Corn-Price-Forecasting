from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["target_corn_ret_fwd_5td", "target_corn_ret_fwd_10td"]


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true.to_numpy()) == np.sign(y_pred)).mean())


def hit_rate_long_only(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mask = y_pred > 0
    if not mask.any():
        return float("nan")
    return float((y_true.to_numpy()[mask] > 0).mean())


def make_models(n_train: int) -> dict[str, Pipeline]:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    return {
        "ridge": Pipeline(
            steps=[
                ("prep", ColumnTransformer([], remainder="drop")),
                ("model", Ridge(alpha=5.0)),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("prep", ColumnTransformer([], remainder="drop")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=350,
                        learning_rate=0.025,
                        max_depth=2,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", ColumnTransformer([], remainder="drop")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=5,
                        min_samples_leaf=max(8, n_train // 80),
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }, numeric_transformer, categorical_transformer


def configure_preprocessor(pipe: Pipeline, numeric_cols: list[str], categorical_cols: list[str],
                           numeric_transformer: Pipeline, categorical_transformer: Pipeline) -> None:
    pipe.steps[0] = (
        "prep",
        ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
        ),
    )


def signal_label(ret: float) -> str:
    if ret >= 0.01:
        return "strong bullish"
    if ret >= 0.003:
        return "mild bullish"
    if ret <= -0.01:
        return "strong bearish"
    if ret <= -0.003:
        return "mild bearish"
    return "neutral"


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    formatted = frame.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else format(x, floatfmt))
        else:
            formatted[col] = formatted[col].astype(str)
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast China corn futures short-horizon returns.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/keyizhan/Downloads/china_corn_trading_day_trend_dataset_enriched.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/keyizhan/Documents/时序/outputs/china_corn_forecast"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["date", *TARGETS]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    rows = []
    predictions = []
    latest_X = df.loc[[len(df) - 1], feature_cols]
    latest_date = df["date"].max()

    for target in TARGETS:
        modeled = df.dropna(subset=[target]).copy()
        split_idx = int(len(modeled) * 0.8)
        train = modeled.iloc[:split_idx]
        test = modeled.iloc[split_idx:]

        X_train, y_train = train[feature_cols], train[target]
        X_test, y_test = test[feature_cols], test[target]

        models, num_tx, cat_tx = make_models(len(train))
        baseline = np.full(len(test), y_train.mean())
        rows.append(
            {
                "target": target,
                "model": "train_mean_baseline",
                "test_start": test["date"].iloc[0].date(),
                "test_end": test["date"].iloc[-1].date(),
                "mae": mean_absolute_error(y_test, baseline),
                "rmse": mean_squared_error(y_test, baseline) ** 0.5,
                "directional_accuracy": directional_accuracy(y_test, baseline),
                "long_only_hit_rate_when_pred_positive": hit_rate_long_only(y_test, baseline),
                "test_pred_mean": baseline.mean(),
            }
        )

        fitted = {}
        for name, pipe in models.items():
            configure_preprocessor(pipe, numeric_cols, categorical_cols, num_tx, cat_tx)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "test_start": test["date"].iloc[0].date(),
                    "test_end": test["date"].iloc[-1].date(),
                    "mae": mean_absolute_error(y_test, pred),
                    "rmse": mean_squared_error(y_test, pred) ** 0.5,
                    "directional_accuracy": directional_accuracy(y_test, pred),
                    "long_only_hit_rate_when_pred_positive": hit_rate_long_only(y_test, pred),
                    "test_pred_mean": pred.mean(),
                }
            )
            fitted[name] = pipe

        metrics_for_target = pd.DataFrame([r for r in rows if r["target"] == target])
        best_name = metrics_for_target.sort_values(["rmse", "mae"]).iloc[0]["model"]
        if best_name == "train_mean_baseline":
            pred_latest = float(y_train.mean())
        else:
            best_pipe = fitted[best_name]
            best_pipe.fit(modeled[feature_cols], modeled[target])
            pred_latest = float(best_pipe.predict(latest_X)[0])

        horizon = "5 trading days" if target.endswith("5td") else "10 trading days"
        latest_close = float(df.loc[len(df) - 1, "dce_corn_close"])
        predictions.append(
            {
                "as_of_date": latest_date.date(),
                "target": target,
                "horizon": horizon,
                "selected_model": best_name,
                "predicted_return": pred_latest,
                "implied_close": latest_close * (1 + pred_latest),
                "signal": signal_label(pred_latest),
            }
        )

    metrics = pd.DataFrame(rows)
    pred_df = pd.DataFrame(predictions)

    recent_cols = [
        "date",
        "dce_corn_close",
        "dce_corn_ret_1d",
        "dce_corn_volume",
        "dce_corn_open_interest",
        "domestic_corn_spot_price_cny_t",
        "corn_basis_cny_t",
        "corn_basis_rate",
        "cbot_corn_close",
        "cbot_corn_ret_1d",
        "cbot_wheat_corn_ratio",
        "ne_avg_precip_mm",
        "ne_avg_t2m_c",
    ]
    recent = df[recent_cols].tail(20)

    metrics.to_csv(args.out_dir / "model_metrics.csv", index=False)
    pred_df.to_csv(args.out_dir / "latest_predictions.csv", index=False)
    recent.to_csv(args.out_dir / "recent_market_snapshot.csv", index=False)

    latest = df.iloc[-1]
    report = [
        "# China Corn Futures Forecast Plan",
        "",
        f"Data range: {df['date'].min().date()} to {df['date'].max().date()}, {len(df):,} trading rows.",
        f"Latest DCE corn close: {latest['dce_corn_close']:.0f}.",
        "",
        "## Latest Forecast",
        markdown_table(pred_df, floatfmt=".4f"),
        "",
        "## Model Backtest Metrics",
        markdown_table(metrics.sort_values(["target", "rmse"]), floatfmt=".4f"),
        "",
        "## Planning Notes",
        "- Treat this as a short-horizon statistical signal, not a standalone trading system.",
        "- Use the 5-trading-day forecast for entry timing and the 10-trading-day forecast for position bias.",
        "- When both horizons are bullish, plan staged long exposure; when both are bearish, stay flat or hedge existing inventory.",
        "- When horizons disagree, reduce size and wait for confirmation from basis, open interest, and CBOT corn direction.",
        "- Risk control: cap single signal loss around 0.8%-1.2% of futures price, and re-evaluate after each new trading day.",
        "- Refresh the dataset daily; the target columns for the latest rows are expected to be blank and should not be filled manually.",
    ]
    (args.out_dir / "forecast_plan.md").write_text("\n".join(report), encoding="utf-8")
    print((args.out_dir / "forecast_plan.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
