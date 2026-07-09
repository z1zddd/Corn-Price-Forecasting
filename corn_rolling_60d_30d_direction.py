from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use trailing 60 trading days to predict whether the next 30-day mean close rises or falls."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=30)
    return parser.parse_args()


def build_window_dataset(df: pd.DataFrame, lookback: int, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    close_col = "dce_corn_close"
    numeric_cols = [
        c
        for c in df.columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("target_")
    ]

    rows = []
    for i in range(lookback - 1, len(df) - horizon):
        hist = df.iloc[i - lookback + 1 : i + 1]
        fut = df.iloc[i + 1 : i + 1 + horizon]
        past_mean = hist[close_col].mean()
        future_mean = fut[close_col].mean()
        row = {
            "as_of_date": df.loc[i, "date"],
            "current_close": df.loc[i, close_col],
            "past60_close_mean": past_mean,
            "future30_close_mean": future_mean,
            "future30_mean_return_vs_past60_mean": future_mean / past_mean - 1,
            "target_up_vs_past60_mean": int(future_mean > past_mean),
        }
        for col in numeric_cols:
            series = hist[col]
            row[f"{col}_last"] = series.iloc[-1]
            row[f"{col}_mean60"] = series.mean()
            row[f"{col}_std60"] = series.std()
            denom = series.iloc[0]
            row[f"{col}_chg60"] = np.nan if pd.isna(denom) or denom == 0 else series.iloc[-1] / denom - 1
        rows.append(row)

    supervised = pd.DataFrame(rows)
    exclude = {
        "as_of_date",
        "future30_close_mean",
        "target_up_vs_past60_mean",
        "future30_mean_return_vs_past60_mean",
    }
    feature_cols = [c for c in supervised.columns if c not in exclude and not supervised[c].isna().all()]
    return supervised, feature_cols


def latest_features(df: pd.DataFrame, feature_cols: list[str], lookback: int) -> pd.DataFrame:
    close_col = "dce_corn_close"
    numeric_cols = [
        c
        for c in df.columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("target_")
    ]
    hist = df.iloc[-lookback:]
    row = {
        "current_close": df.iloc[-1][close_col],
        "past60_close_mean": hist[close_col].mean(),
    }
    for col in numeric_cols:
        series = hist[col]
        row[f"{col}_last"] = series.iloc[-1]
        row[f"{col}_mean60"] = series.mean()
        row[f"{col}_std60"] = series.std()
        denom = series.iloc[0]
        row[f"{col}_chg60"] = np.nan if pd.isna(denom) or denom == 0 else series.iloc[-1] / denom - 1
    return pd.DataFrame([row])[feature_cols]


def model_zoo() -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=12,
                        random_state=42,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=2, random_state=42)),
            ]
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    supervised, feature_cols = build_window_dataset(df, args.lookback, args.horizon)
    x = supervised[feature_cols]
    y = supervised["target_up_vs_past60_mean"]
    split = int(len(supervised) * 0.8)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    metrics = []
    best = None
    models = model_zoo()
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)[:, 1]
        row = {
            "model": name,
            "test_start": supervised.iloc[split]["as_of_date"].date(),
            "test_end": supervised.iloc[-1]["as_of_date"].date(),
            "accuracy": accuracy_score(y_test, pred),
            "precision_up": precision_score(y_test, pred, zero_division=0),
            "recall_up": recall_score(y_test, pred, zero_division=0),
            "pred_up_rate": float(pred.mean()),
            "avg_proba_up": float(proba.mean()),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        metrics.append(row)
        key = (row["accuracy"], row["precision_up"])
        if best is None or key > best[0]:
            best = (key, name, model)

    assert best is not None
    best_name = best[1]
    best_model = best[2]
    best_model.fit(x, y)
    latest_x = latest_features(df, feature_cols, args.lookback)
    latest_hist = df.iloc[-args.lookback:]
    up_probability = float(best_model.predict_proba(latest_x)[:, 1][0])
    pred_up = int(best_model.predict(latest_x)[0])

    prediction = pd.DataFrame(
        [
            {
                "as_of_date": df.iloc[-1]["date"].date(),
                "lookback_days": args.lookback,
                "horizon_days": args.horizon,
                "past60_start": latest_hist.iloc[0]["date"].date(),
                "past60_end": latest_hist.iloc[-1]["date"].date(),
                "past60_close_mean": latest_hist["dce_corn_close"].mean(),
                "latest_close": df.iloc[-1]["dce_corn_close"],
                "past60_close_change": latest_hist["dce_corn_close"].iloc[-1] / latest_hist["dce_corn_close"].iloc[0] - 1,
                "selected_model": best_name,
                "probability_up": up_probability,
                "probability_down": 1 - up_probability,
                "predicted_direction": "UP" if pred_up else "DOWN",
            }
        ]
    )

    metrics_df = pd.DataFrame(metrics)
    supervised.tail(200).to_csv(args.out_dir / "rolling_60d_30d_recent_windows.csv", index=False)
    metrics_df.to_csv(args.out_dir / "rolling_60d_30d_model_metrics.csv", index=False)
    prediction.to_csv(args.out_dir / "rolling_60d_30d_latest_prediction.csv", index=False)

    lines = [
        "# Rolling 60D to 30D Direction Forecast",
        "",
        f"Data range: {df['date'].min().date()} to {df['date'].max().date()}, {len(df):,} trading rows.",
        f"Supervised windows: {len(supervised):,}.",
        "",
        "## Latest Prediction",
        prediction.to_string(index=False),
        "",
        "## Backtest Metrics",
        metrics_df.to_string(index=False),
    ]
    report = "\n".join(lines)
    (args.out_dir / "rolling_60d_30d_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
