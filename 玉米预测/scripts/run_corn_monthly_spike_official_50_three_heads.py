#!/usr/bin/env python3
"""Official/standard 50-model rolling benchmark with three direction heads.

Heads:
- cls: direct up/down classifier trained on the future price direction.
- reg: return regressor, converted to direction by thresholding predicted return.
- sum: validation-thresholded fusion of classifier probability and regressor score.

All estimators come from official packages: scikit-learn, LightGBM, XGBoost,
and CatBoost. No custom neural architecture is used here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


FUTURE_COL_KEYWORDS = ("next", "future", "target", "fwd", "lead")
DEFAULT_EXCLUDE_COLS = {
    "first_trade_date",
    "last_trade_date",
    "spike",
    "dce_corn_close_next_month",
    "dce_corn_close_next_month_ret",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    package: str
    classifier_factory: Callable[[int], BaseEstimator]
    regressor_factory: Callable[[int], BaseEstimator]
    classifier_loss: str
    regressor_loss: str


class TrainOnlyStandardizer:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TrainOnlyStandardizer":
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.mean_ = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        self.scale_ = np.where((scale < 1e-12) | ~np.isfinite(scale), 1.0, scale)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Feature scaler is not fitted.")
        z = (x.astype(np.float64) - self.mean_) / self.scale_
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


class TargetStandardizer:
    def __init__(self) -> None:
        self.mean_ = 0.0
        self.scale_ = 1.0

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mean_ = float(np.nanmean(y))
        scale = float(np.nanstd(y))
        self.scale_ = scale if np.isfinite(scale) and scale >= 1e-12 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float64).reshape(-1) - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64).reshape(-1) * self.scale_ + self.mean_).astype(np.float64)


class ConstantClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-4, 1.0 - 1e-4))

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs/corn_monthly_spike_official_50_three_heads")
    parser.add_argument("--date-col", default="month")
    parser.add_argument("--date-format", default="%y-%b")
    parser.add_argument("--price-col", default="dce_corn_close")
    parser.add_argument("--label-col", default="spike")
    parser.add_argument("--lookbacks", default="1,2,3")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument(
        "--label-mode",
        choices=("direct_horizon_direction", "existing_spike_at_target"),
        default="direct_horizon_direction",
        help="direct_horizon_direction predicts price[t+h] > price[t], aligned with the regression head.",
    )
    parser.add_argument("--min-train", type=int, default=48)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=3)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--include-pca", action="store_true")
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-folds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    csv_path = Path(args.csv).expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_monthly_csv(csv_path, args.date_col, args.date_format)
    feature_cols, excluded_cols = select_feature_columns(
        df=df,
        date_col=args.date_col,
        price_col=args.price_col,
        label_col=args.label_col,
        include_pca=args.include_pca,
        extra_exclude=parse_csv_list(args.extra_exclude),
    )
    feature_cols = move_price_last(feature_cols, args.price_col)
    model_specs = select_models(args.models, args.seed)
    lookbacks = [int(x) for x in parse_csv_list(args.lookbacks)]
    horizons = [int(x) for x in parse_csv_list(args.horizons)]

    manifest = {
        "run_id": run_id,
        "csv": str(csv_path),
        "rows": int(len(df)),
        "date_min": str(df[args.date_col].min().date()),
        "date_max": str(df[args.date_col].max().date()),
        "price_col": args.price_col,
        "label_col": args.label_col,
        "label_mode": args.label_mode,
        "lookbacks": lookbacks,
        "horizons": horizons,
        "min_train": args.min_train,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "excluded_cols": excluded_cols,
        "models": [
            {
                "name": m.name,
                "family": m.family,
                "package": m.package,
                "classifier_loss": m.classifier_loss,
                "regressor_loss": m.regressor_loss,
            }
            for m in model_specs
        ],
        "heads": {
            "cls": "direct classifier probability, validation threshold selected for balanced accuracy",
            "reg": "return regressor, inverse-transformed, price direction from predicted future return",
            "sum": "logit(classifier_probability) + train-return-scaled regressor score, validation threshold selected",
        },
        "leakage_controls": [
            "future/target/next/lead columns excluded from features",
            "pca_* columns excluded by default unless --include-pca is set",
            "rolling training rows require target_idx <= cutoff anchor index",
            "feature scaler fitted on train fold only",
            "return target scaler fitted on train fold only and inverse-transformed before price/R2 diagnostics",
            "head thresholds selected on validation fold only",
        ],
        "official_sources": official_sources(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_predictions: list[dict[str, object]] = []
    all_folds: list[dict[str, object]] = []
    for lookback in lookbacks:
        for horizon in horizons:
            samples = make_samples(
                df=df,
                feature_cols=feature_cols,
                date_col=args.date_col,
                price_col=args.price_col,
                label_col=args.label_col,
                lookback=lookback,
                horizon=horizon,
                label_mode=args.label_mode,
            )
            origins = make_rolling_origins(samples, args.min_train, args.val_size, args.test_size, args.step_size, args.max_origins)
            if not origins:
                print(f"[skip] lookback={lookback} horizon={horizon}: no origins", flush=True)
                continue
            for spec in model_specs:
                print(f"[run] lookback={lookback} horizon={horizon} model={spec.name} origins={len(origins)}", flush=True)
                preds, folds = run_model_combo(samples, origins, spec, args.seed)
                all_predictions.extend(preds)
                all_folds.extend(folds)

    predictions_df = pd.DataFrame(all_predictions)
    if predictions_df.empty:
        raise RuntimeError("No predictions were produced.")
    predictions_df.to_csv(out_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize(predictions_df)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    model_summary_df = summarize_model_level(summary_df)
    model_summary_df.to_csv(out_dir / "model_level_summary.csv", index=False, encoding="utf-8-sig")
    if args.save_folds:
        pd.DataFrame(all_folds).to_csv(out_dir / "folds.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, summary_df, model_summary_df, manifest)

    print("\n=== Best By Horizon/Head ===")
    display_cols = ["horizon_months", "head", "model", "lookback_months", "auc", "average_precision", "balanced_accuracy", "reg_price_r2", "r2_status"]
    best_rows = []
    for (horizon, head), group in summary_df.groupby(["horizon_months", "head"]):
        best_rows.append(group.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False).iloc[0])
    print(pd.DataFrame(best_rows)[display_cols].to_string(index=False))
    print(f"\nSaved: {out_dir}")


def load_monthly_csv(csv_path: Path, date_col: str, date_format: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format=date_format, errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad = df.loc[df[date_col].isna(), date_col].head(5).tolist()
        raise ValueError(f"Could not parse date values: {bad}")
    return df.sort_values(date_col).reset_index(drop=True)


def select_feature_columns(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    label_col: str,
    include_pca: bool,
    extra_exclude: list[str],
) -> tuple[list[str], list[str]]:
    excluded = set(DEFAULT_EXCLUDE_COLS) | {date_col, label_col} | set(extra_exclude)
    selected: list[str] = []
    excluded_cols: list[str] = []
    for col in df.columns:
        lower = col.lower()
        should_exclude = (
            col in excluded
            or any(keyword in lower for keyword in FUTURE_COL_KEYWORDS)
            or (not include_pca and lower.startswith("pca_"))
            or not pd.api.types.is_numeric_dtype(df[col])
        )
        if should_exclude:
            excluded_cols.append(col)
            continue
        selected.append(col)
    if price_col not in selected:
        raise ValueError(f"{price_col} must be an eligible numeric feature.")
    return selected, excluded_cols


def make_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str,
    price_col: str,
    label_col: str,
    lookback: int,
    horizon: int,
    label_mode: str,
) -> dict[str, object]:
    features = df[feature_cols].to_numpy(dtype=np.float32)
    prices = df[price_col].to_numpy(dtype=np.float64)
    spike = df[label_col].to_numpy(dtype=np.int64)
    dates = pd.to_datetime(df[date_col])
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    ret_rows: list[float] = []
    meta_rows: list[dict[str, object]] = []
    for anchor_idx in range(lookback - 1, len(df) - horizon):
        start = anchor_idx - lookback + 1
        target_idx = anchor_idx + horizon
        window = features[start : anchor_idx + 1]
        direct_return = prices[target_idx] / prices[anchor_idx] - 1.0
        label = int(direct_return > 0.0) if label_mode == "direct_horizon_direction" else int(spike[target_idx])
        if not np.isfinite(window).any() or not np.isfinite(direct_return):
            continue
        x_rows.append(window)
        y_rows.append(label)
        ret_rows.append(float(direct_return))
        meta_rows.append(
            {
                "sample_id": len(x_rows) - 1,
                "anchor_idx": int(anchor_idx),
                "target_idx": int(target_idx),
                "anchor_month": str(dates.iloc[anchor_idx].date()),
                "target_month": str(dates.iloc[target_idx].date()),
                "anchor_price": float(prices[anchor_idx]),
                "actual_price": float(prices[target_idx]),
                "actual_return": float(direct_return),
                "target_existing_spike": int(spike[target_idx]),
            }
        )
    return {
        "X": np.stack(x_rows).astype(np.float32),
        "y": np.asarray(y_rows, dtype=np.int64),
        "returns": np.asarray(ret_rows, dtype=np.float64),
        "meta": pd.DataFrame(meta_rows),
        "lookback": lookback,
        "horizon": horizon,
    }


def make_rolling_origins(samples, min_train: int, val_size: int, test_size: int, step_size: int, max_origins: int):
    meta = samples["meta"]
    origins = []
    cursor = 0
    origin_id = 0
    while cursor < len(meta):
        cutoff_anchor_idx = int(meta.iloc[cursor]["anchor_idx"])
        trainval_idx = meta.index[meta["target_idx"] <= cutoff_anchor_idx].to_numpy(dtype=int)
        if len(trainval_idx) >= min_train + val_size:
            test_idx = meta.index[meta["anchor_idx"] > cutoff_anchor_idx].to_numpy(dtype=int)[:test_size]
            if len(test_idx) > 0:
                origins.append((trainval_idx[:-val_size], trainval_idx[-val_size:], test_idx, origin_id))
                origin_id += 1
                if max_origins and len(origins) >= max_origins:
                    break
                cursor = int(test_idx[-1]) + step_size
                continue
        cursor += 1
    return origins


def run_model_combo(samples, origins, spec: ModelSpec, seed: int):
    x = samples["X"]
    y = samples["y"]
    returns = samples["returns"]
    meta = samples["meta"]
    lookback = int(samples["lookback"])
    horizon = int(samples["horizon"])
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for train_idx, val_idx, test_idx, origin_id in origins:
        fold_seed = seed + stable_name_offset(spec.name) + 101 * lookback + 17 * horizon + origin_id
        scaler = TrainOnlyStandardizer().fit(x[train_idx])
        x_train = flatten(scaler.transform(x[train_idx]))
        x_val = flatten(scaler.transform(x[val_idx]))
        x_test = flatten(scaler.transform(x[test_idx]))

        clf = fit_classifier(spec, x_train, y[train_idx], fold_seed)
        cls_val_prob = predict_probability(clf, x_val)
        cls_test_prob = predict_probability(clf, x_test)
        cls_threshold = select_threshold(y[val_idx], cls_val_prob)

        y_scaler = TargetStandardizer().fit(returns[train_idx])
        reg = fit_regressor(spec, x_train, y_scaler.transform(returns[train_idx]), fold_seed)
        reg_val_return = y_scaler.inverse_transform(np.asarray(reg.predict(x_val), dtype=float))
        reg_test_return = y_scaler.inverse_transform(np.asarray(reg.predict(x_test), dtype=float))
        reg_threshold = select_score_threshold(y[val_idx], reg_val_return)
        return_scale = max(float(np.nanstd(returns[train_idx])), 1e-4)
        reg_val_prob = sigmoid(reg_val_return / return_scale)
        reg_test_prob = sigmoid(reg_test_return / return_scale)

        sum_val_score = logit(cls_val_prob) + logit(reg_val_prob)
        sum_test_score = logit(cls_test_prob) + logit(reg_test_prob)
        sum_threshold = select_score_threshold(y[val_idx], sum_val_score)
        sum_test_prob = sigmoid(sum_test_score)

        predicted_price = meta.iloc[test_idx]["anchor_price"].to_numpy(float) * (1.0 + reg_test_return)
        actual_price = meta.iloc[test_idx]["actual_price"].to_numpy(float)
        anchor_price = meta.iloc[test_idx]["anchor_price"].to_numpy(float)
        reg_price_r2 = safe_r2(actual_price, predicted_price)
        naive_price_r2 = safe_r2(actual_price, anchor_price)
        fold_meta = {
            "model": spec.name,
            "family": spec.family,
            "package": spec.package,
            "lookback_months": lookback,
            "horizon_months": horizon,
            "origin_id": origin_id,
            "cutoff_month": str(meta.iloc[val_idx[-1]]["target_month"]),
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "test_rows": len(test_idx),
            "cls_threshold": float(cls_threshold),
            "reg_threshold": float(reg_threshold),
            "sum_threshold": float(sum_threshold),
            "reg_price_r2_fold": float(reg_price_r2),
            "naive_price_r2_fold": float(naive_price_r2),
        }
        folds.append(fold_meta)

        head_payloads = [
            ("cls", cls_test_prob, cls_test_prob >= cls_threshold, cls_threshold),
            ("reg", reg_test_prob, reg_test_return >= reg_threshold, reg_threshold),
            ("sum", sum_test_prob, sum_test_score >= sum_threshold, sum_threshold),
        ]
        for offset, sample_idx in enumerate(test_idx):
            base = {
                "model": spec.name,
                "family": spec.family,
                "package": spec.package,
                "classifier_loss": spec.classifier_loss,
                "regressor_loss": spec.regressor_loss,
                "lookback_months": lookback,
                "horizon_months": horizon,
                "origin_id": origin_id,
                "anchor_month": meta.iloc[sample_idx]["anchor_month"],
                "target_month": meta.iloc[sample_idx]["target_month"],
                "anchor_idx": int(meta.iloc[sample_idx]["anchor_idx"]),
                "target_idx": int(meta.iloc[sample_idx]["target_idx"]),
                "anchor_price": float(anchor_price[offset]),
                "actual_price": float(actual_price[offset]),
                "predicted_price_reg": float(predicted_price[offset]),
                "actual_return": float(meta.iloc[sample_idx]["actual_return"]),
                "predicted_return_reg": float(reg_test_return[offset]),
                "actual_direction": int(y[sample_idx]),
                "target_existing_spike": int(meta.iloc[sample_idx]["target_existing_spike"]),
                "reg_price_r2_fold": float(reg_price_r2),
                "naive_price_r2_fold": float(naive_price_r2),
            }
            for head, prob, pred, threshold in head_payloads:
                row = dict(base)
                row.update(
                    {
                        "head": head,
                        "threshold": float(threshold),
                        "predicted_direction": int(pred[offset]),
                        "predicted_probability": float(prob[offset]),
                    }
                )
                predictions.append(row)
    return predictions, folds


def fit_classifier(spec: ModelSpec, x: np.ndarray, y: np.ndarray, seed: int):
    if len(np.unique(y)) < 2:
        return ConstantClassifier(float(np.mean(y)))
    model = spec.classifier_factory(seed)
    model.fit(x, y.astype(int))
    return model


def fit_regressor(spec: ModelSpec, x: np.ndarray, y_scaled: np.ndarray, seed: int):
    if np.nanstd(y_scaled) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y_scaled)))
    model = spec.regressor_factory(seed)
    model.fit(x, y_scaled)
    return model


def predict_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if np.asarray(proba).ndim == 2 and proba.shape[1] > 1:
            return np.clip(proba[:, 1].astype(float), 1e-6, 1.0 - 1e-6)
        return np.clip(np.asarray(proba, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
    if hasattr(model, "decision_function"):
        return sigmoid(np.asarray(model.decision_function(x), dtype=float).reshape(-1))
    pred = np.asarray(model.predict(x), dtype=float).reshape(-1)
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "family", "package", "lookback_months", "horizon_months", "head", "classifier_loss", "regressor_loss"]
    for keys, group in predictions.groupby(group_cols):
        y = group["actual_direction"].to_numpy(dtype=int)
        pred = group["predicted_direction"].to_numpy(dtype=int)
        prob = group["predicted_probability"].to_numpy(dtype=float)
        actual_price = group["actual_price"].to_numpy(dtype=float)
        pred_price = group["predicted_price_reg"].to_numpy(dtype=float)
        anchor_price = group["anchor_price"].to_numpy(dtype=float)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        reg_price_r2 = safe_r2(actual_price, pred_price)
        naive_r2 = safe_r2(actual_price, anchor_price)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_predictions": int(len(group)),
                "class_0_count": int((y == 0).sum()),
                "class_1_count": int((y == 1).sum()),
                "auc": safe_auc(y, prob),
                "average_precision": safe_ap(y, prob),
                "balanced_accuracy": safe_balanced_accuracy(y, pred),
                "accuracy": float(accuracy_score(y, pred)),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "reg_price_r2": reg_price_r2,
                "naive_price_r2": naive_r2,
                "r2_status": r2_status(reg_price_r2),
                "price_rmse": float(math.sqrt(mean_squared_error(actual_price, pred_price))),
                "price_mae": float(mean_absolute_error(actual_price, pred_price)),
                "predicted_positive_rate": float(np.mean(pred == 1)),
                "actual_positive_rate": float(np.mean(y == 1)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["horizon_months", "head", "balanced_accuracy", "auc", "average_precision"],
        ascending=[True, True, False, False, False],
    )


def summarize_model_level(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.groupby(["model", "head"])["balanced_accuracy"].idxmax()
    best = summary.loc[idx].copy()
    best = best.sort_values(["head", "balanced_accuracy", "auc", "average_precision"], ascending=[True, False, False, False])
    return best


def write_report(out_dir: Path, summary: pd.DataFrame, model_summary: pd.DataFrame, manifest: dict[str, object]) -> None:
    lines = [
        "# Corn Monthly Official 50-Model Three-Head Rolling Benchmark",
        "",
        f"- CSV: `{manifest['csv']}`",
        f"- Date range: `{manifest['date_min']}` to `{manifest['date_max']}`",
        f"- Label mode: `{manifest['label_mode']}`",
        f"- Models: `{len(manifest['models'])}` official/standard estimators",
        f"- Heads: `cls`, `reg`, `sum`",
        f"- Feature count: `{manifest['feature_count']}`; PCA included: `{any(str(c).startswith('pca_') for c in manifest['feature_cols'])}`",
        "",
        "## Leakage Controls",
        "",
    ]
    for item in manifest["leakage_controls"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Official Sources", ""])
    for name, url in manifest["official_sources"].items():
        lines.append(f"- {name}: {url}")
    lines.extend(["", "## Best By Horizon And Head", ""])
    best_rows = []
    for (_, _), group in summary.groupby(["horizon_months", "head"]):
        best_rows.append(group.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False).iloc[0])
    display_cols = ["horizon_months", "head", "model", "lookback_months", "auc", "average_precision", "balanced_accuracy", "reg_price_r2", "naive_price_r2", "r2_status"]
    lines.append(pd.DataFrame(best_rows)[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Best Per Model/Head", ""])
    lines.append(model_summary[display_cols].head(80).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "Outputs:", "", "- `summary_metrics.csv`", "- `model_level_summary.csv`", "- `rolling_predictions.csv`", "- `manifest.json`"])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_model_zoo(seed: int) -> list[ModelSpec]:
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.linear_model import (
        BayesianRidge,
        ElasticNet,
        HuberRegressor,
        Lasso,
        LinearRegression,
        LogisticRegression,
        PassiveAggressiveClassifier,
        PassiveAggressiveRegressor,
        Perceptron,
        Ridge,
        RidgeClassifier,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

    def ms(name, family, package, clf, reg, clf_loss="native", reg_loss="squared_error"):
        return ModelSpec(name, family, package, clf, reg, clf_loss, reg_loss)

    def lr(**kwargs):
        params = {"max_iter": 2000, "class_weight": "balanced", "random_state": seed}
        params.update(kwargs)
        return lambda s: LogisticRegression(**{**params, "random_state": s})

    def ridge_reg(alpha=1.0):
        return lambda s: Ridge(alpha=alpha)

    zoo = [
        ms("dummy_prior", "baseline", "sklearn", lambda s: DummyClassifier(strategy="prior"), lambda s: DummyRegressor(strategy="mean"), "prior", "mean"),
        ms("logistic_l2_lbfgs", "linear", "sklearn", lr(penalty="l2", solver="lbfgs", C=1.0), ridge_reg(1.0), "log_loss", "ridge_l2"),
        ms("logistic_l1_liblinear", "linear", "sklearn", lr(penalty="l1", solver="liblinear", C=0.5), lambda s: Lasso(alpha=0.01, max_iter=5000), "log_loss_l1", "lasso_l1"),
        ms("logistic_elasticnet_saga", "linear", "sklearn", lr(penalty="elasticnet", solver="saga", l1_ratio=0.3, C=0.5), lambda s: ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=5000), "log_loss_elasticnet", "elasticnet"),
        ms("ridge_classifier", "linear", "sklearn", lambda s: RidgeClassifier(class_weight="balanced"), ridge_reg(1.0), "squared_hinge_like", "ridge_l2"),
        ms("sgd_log_loss", "linear", "sklearn", lambda s: SGDClassifier(loss="log_loss", alpha=1e-3, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s), lambda s: SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3, max_iter=2000, tol=1e-4, random_state=s), "log_loss", "squared_error"),
        ms("sgd_modified_huber", "linear", "sklearn", lambda s: SGDClassifier(loss="modified_huber", alpha=1e-3, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s), lambda s: HuberRegressor(alpha=1e-3, max_iter=200), "modified_huber", "huber"),
        ms("passive_aggressive", "linear", "sklearn", lambda s: PassiveAggressiveClassifier(class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s), lambda s: PassiveAggressiveRegressor(max_iter=2000, tol=1e-4, random_state=s), "hinge", "pa"),
        ms("perceptron", "linear", "sklearn", lambda s: Perceptron(class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s), lambda s: PassiveAggressiveRegressor(max_iter=2000, tol=1e-4, random_state=s), "perceptron", "pa"),
        ms("linear_svc", "svm", "sklearn", lambda s: LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=s), lambda s: LinearSVR(C=0.5, max_iter=5000, random_state=s), "hinge", "epsilon_insensitive"),
        ms("svc_rbf", "svm", "sklearn", lambda s: SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=1.0, gamma="scale", kernel="rbf"), "hinge_platt", "epsilon_insensitive"),
        ms("svc_poly2", "svm", "sklearn", lambda s: SVC(C=0.7, degree=2, gamma="scale", kernel="poly", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=0.7, degree=2, gamma="scale", kernel="poly"), "hinge_platt", "epsilon_insensitive"),
        ms("svc_sigmoid", "svm", "sklearn", lambda s: SVC(C=0.7, gamma="scale", kernel="sigmoid", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=0.7, gamma="scale", kernel="sigmoid"), "hinge_platt", "epsilon_insensitive"),
        ms("nu_svc_rbf", "svm", "sklearn", lambda s: NuSVC(nu=0.35, gamma="scale", kernel="rbf", probability=True, random_state=s), lambda s: NuSVR(nu=0.35, gamma="scale", kernel="rbf"), "hinge_platt", "epsilon_insensitive"),
        ms("knn_3_uniform", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=3, weights="uniform"), lambda s: KNeighborsRegressor(n_neighbors=3, weights="uniform"), "vote", "neighbor_mean"),
        ms("knn_5_distance", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=5, weights="distance"), lambda s: KNeighborsRegressor(n_neighbors=5, weights="distance"), "vote_distance", "neighbor_distance"),
        ms("knn_9_distance", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=9, weights="distance"), lambda s: KNeighborsRegressor(n_neighbors=9, weights="distance"), "vote_distance", "neighbor_distance"),
        ms("nearest_centroid", "neighbors", "sklearn", lambda s: NearestCentroid(), ridge_reg(1.0), "centroid_distance", "ridge_l2"),
        ms("gaussian_nb", "bayes", "sklearn", lambda s: GaussianNB(), lambda s: BayesianRidge(), "gaussian_nb", "bayesian_ridge"),
        ms("bernoulli_nb", "bayes", "sklearn", lambda s: BernoulliNB(binarize=0.0, alpha=1.0), ridge_reg(1.0), "bernoulli_nb", "ridge_l2"),
        ms("lda_svd", "discriminant", "sklearn", lambda s: LinearDiscriminantAnalysis(solver="svd"), lambda s: LinearRegression(), "lda", "ols"),
        ms("lda_shrinkage", "discriminant", "sklearn", lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), ridge_reg(1.0), "lda_shrinkage", "ridge_l2"),
        ms("qda_reg", "discriminant", "sklearn", lambda s: QuadraticDiscriminantAnalysis(reg_param=0.2), ridge_reg(1.0), "qda", "ridge_l2"),
        ms("decision_tree_gini", "tree", "sklearn", lambda s: DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: DecisionTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=4, random_state=s), "gini", "squared_error"),
        ms("decision_tree_entropy", "tree", "sklearn", lambda s: DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: DecisionTreeRegressor(criterion="absolute_error", max_depth=4, min_samples_leaf=4, random_state=s), "entropy", "absolute_error"),
        ms("extra_tree_gini", "tree", "sklearn", lambda s: ExtraTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: ExtraTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=4, random_state=s), "gini_randomized", "squared_error"),
        ms("extra_tree_entropy", "tree", "sklearn", lambda s: ExtraTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: ExtraTreeRegressor(criterion="absolute_error", max_depth=4, min_samples_leaf=4, random_state=s), "entropy_randomized", "absolute_error"),
        ms("random_forest_100", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=s, n_jobs=1), "gini_bagging", "squared_error_bagging"),
        ms("random_forest_balanced", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=120, max_depth=None, min_samples_leaf=4, class_weight="balanced_subsample", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=120, max_depth=None, min_samples_leaf=4, random_state=s, n_jobs=1), "gini_balanced_bagging", "squared_error_bagging"),
        ms("random_forest_shallow", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=120, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=120, max_depth=3, min_samples_leaf=3, random_state=s, n_jobs=1), "gini_shallow", "squared_error_shallow"),
        ms("extra_trees_200", "forest", "sklearn", lambda s: ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=s, n_jobs=1), "randomized_tree_bagging", "randomized_tree_regression"),
        ms("extra_trees_balanced", "forest", "sklearn", lambda s: ExtraTreesClassifier(n_estimators=220, max_depth=None, min_samples_leaf=4, class_weight="balanced", random_state=s, n_jobs=1), lambda s: ExtraTreesRegressor(n_estimators=220, max_depth=None, min_samples_leaf=4, random_state=s, n_jobs=1), "randomized_tree_balanced", "randomized_tree_regression"),
        ms("gradient_boosting", "boosting", "sklearn", lambda s: GradientBoostingClassifier(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=s), lambda s: GradientBoostingRegressor(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=s), "deviance", "squared_error_boosting"),
        ms("gradient_boosting_shallow", "boosting", "sklearn", lambda s: GradientBoostingClassifier(n_estimators=120, learning_rate=0.025, max_depth=1, random_state=s), lambda s: GradientBoostingRegressor(n_estimators=120, learning_rate=0.025, max_depth=1, random_state=s), "deviance_stumps", "squared_error_stumps"),
        ms("hist_gradient_boosting", "boosting", "sklearn", lambda s: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=s), lambda s: HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=s), "log_loss_hist_gbdt", "squared_error_hist_gbdt"),
        ms("hist_gradient_boosting_l2", "boosting", "sklearn", lambda s: HistGradientBoostingClassifier(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=1.0, random_state=s), lambda s: HistGradientBoostingRegressor(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=1.0, random_state=s), "log_loss_hist_l2", "squared_error_hist_l2"),
        ms("ada_boost_tree", "boosting", "sklearn", lambda s: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=s), n_estimators=80, learning_rate=0.05, algorithm="SAMME", random_state=s), lambda s: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=1, random_state=s), n_estimators=80, learning_rate=0.05, random_state=s), "samme", "adaboost_square"),
        ms("bagging_tree", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=s), n_estimators=80, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=s), n_estimators=80, max_samples=0.8, random_state=s, n_jobs=1), "bagged_tree_vote", "bagged_tree_regression"),
        ms("bagging_extra_tree", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=ExtraTreeClassifier(max_depth=4, class_weight="balanced", random_state=s), n_estimators=100, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=ExtraTreeRegressor(max_depth=4, random_state=s), n_estimators=100, max_samples=0.8, random_state=s, n_jobs=1), "bagged_extra_tree_vote", "bagged_extra_tree_regression"),
        ms("bagging_logistic", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), n_estimators=30, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=30, max_samples=0.8, random_state=s, n_jobs=1), "bagged_log_loss", "bagged_ridge"),
        ms("mlp_small_relu", "neural_sklearn", "sklearn", lambda s: MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), lambda s: MLPRegressor(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), "log_loss_mlp", "squared_error_mlp"),
        ms("mlp_small_tanh", "neural_sklearn", "sklearn", lambda s: MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), lambda s: MLPRegressor(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), "log_loss_mlp", "squared_error_mlp"),
        ms("gaussian_process_rbf", "kernel", "sklearn", lambda s: GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), max_iter_predict=100, random_state=s), lambda s: GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0), alpha=1e-2, normalize_y=True, random_state=s), "laplace_log_marginal", "gp_regression"),
        ms("lightgbm_gbdt", "gbdt", "lightgbm", lightgbm_classifier("gbdt"), lightgbm_regressor("gbdt"), "binary_logloss", "l2"),
        ms("lightgbm_goss", "gbdt", "lightgbm", lightgbm_classifier("goss"), lightgbm_regressor("goss"), "binary_logloss_goss", "l2_goss"),
        ms("lightgbm_dart", "gbdt", "lightgbm", lightgbm_classifier("dart"), lightgbm_regressor("dart"), "binary_logloss_dart", "l2_dart"),
        ms("xgboost_gbtree", "gbdt", "xgboost", xgb_classifier("gbtree"), xgb_regressor("gbtree"), "logloss", "squared_error"),
        ms("xgboost_dart", "gbdt", "xgboost", xgb_classifier("dart"), xgb_regressor("dart"), "logloss_dart", "squared_error_dart"),
        ms("catboost_symmetric", "gbdt", "catboost", catboost_classifier("SymmetricTree"), catboost_regressor("SymmetricTree"), "logloss", "rmse"),
        ms("catboost_depthwise", "gbdt", "catboost", catboost_classifier("Depthwise"), catboost_regressor("Depthwise"), "logloss_depthwise", "rmse_depthwise"),
    ]
    if len(zoo) != 50:
        raise AssertionError(f"Expected 50 models, got {len(zoo)}")
    return zoo


def lightgbm_classifier(boosting_type: str):
    def factory(seed):
        from lightgbm import LGBMClassifier

        params = {
            "boosting_type": boosting_type,
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        if boosting_type == "goss":
            params.pop("subsample")
        return LGBMClassifier(**params)

    return factory


def lightgbm_regressor(boosting_type: str):
    def factory(seed):
        from lightgbm import LGBMRegressor

        params = {
            "boosting_type": boosting_type,
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        if boosting_type == "goss":
            params.pop("subsample")
        return LGBMRegressor(**params)

    return factory


def xgb_classifier(booster: str):
    def factory(seed):
        from xgboost import XGBClassifier

        return XGBClassifier(
            booster=booster,
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    return factory


def xgb_regressor(booster: str):
    def factory(seed):
        from xgboost import XGBRegressor

        return XGBRegressor(
            booster=booster,
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    return factory


def catboost_classifier(grow_policy: str):
    def factory(seed):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=100,
            learning_rate=0.035,
            depth=3,
            grow_policy=grow_policy,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return factory


def catboost_regressor(grow_policy: str):
    def factory(seed):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=100,
            learning_rate=0.035,
            depth=3,
            grow_policy=grow_policy,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return factory


def select_models(requested: str, seed: int) -> list[ModelSpec]:
    zoo = build_model_zoo(seed)
    if requested == "all":
        return zoo
    wanted = parse_csv_list(requested)
    by_name = {m.name: m for m in zoo}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"Unknown model(s): {missing}")
    return [by_name[name] for name in wanted]


def select_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    return select_score_threshold(y, prob)


def select_score_threshold(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=float).reshape(-1)
    if len(np.unique(y)) < 2 or np.nanstd(score) < 1e-12:
        return float(np.nanmedian(score))
    candidates = np.unique(np.quantile(score, np.linspace(0.05, 0.95, 31)))
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        bal = balanced_accuracy_score(y, pred)
        if bal > best_score:
            best_score = float(bal)
            best_threshold = float(threshold)
    return best_threshold


def official_sources() -> dict[str, str]:
    return {
        "scikit-learn": "https://scikit-learn.org/stable/supervised_learning.html",
        "LightGBM": "https://lightgbm.readthedocs.io/en/stable/Python-API.html",
        "XGBoost": "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
        "CatBoost": "https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier",
    }


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def move_price_last(feature_cols: list[str], price_col: str) -> list[str]:
    return [c for c in feature_cols if c != price_col] + [price_col]


def flatten(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], -1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50.0, 50.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    y_arr = np.asarray(y)
    prob_arr = np.asarray(prob, dtype=float)
    mask = np.isfinite(prob_arr)
    y_arr = y_arr[mask]
    prob_arr = prob_arr[mask]
    return float(roc_auc_score(y_arr, prob_arr)) if len(y_arr) >= 2 and len(np.unique(y_arr)) == 2 else float("nan")


def safe_ap(y: np.ndarray, prob: np.ndarray) -> float:
    y_arr = np.asarray(y)
    prob_arr = np.asarray(prob, dtype=float)
    mask = np.isfinite(prob_arr)
    y_arr = y_arr[mask]
    prob_arr = prob_arr[mask]
    return float(average_precision_score(y_arr, prob_arr)) if len(y_arr) >= 2 and len(np.unique(y_arr)) == 2 else float("nan")


def safe_balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y_arr = np.asarray(y)
    pred_arr = np.asarray(pred)
    mask = np.isfinite(pred_arr.astype(float, copy=False))
    y_arr = y_arr[mask]
    pred_arr = pred_arr[mask]
    return float(balanced_accuracy_score(y_arr, pred_arr)) if len(y_arr) >= 2 and len(np.unique(y_arr)) == 2 else float("nan")


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    return float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) >= 2 and np.nanstd(y_true_arr) > 1e-12 else float("nan")


def r2_status(r2: float) -> str:
    if not np.isfinite(r2):
        return "undefined"
    if r2 < -0.1:
        return "abnormal"
    if r2 < 0:
        return "likely_abnormal"
    return "ok"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_name_offset(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 1009


if __name__ == "__main__":
    main()
