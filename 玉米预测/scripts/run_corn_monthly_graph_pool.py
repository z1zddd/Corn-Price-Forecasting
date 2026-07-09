#!/usr/bin/env python3
"""Leakage-safe graph feature rolling benchmark for the old monthly corn data.

This script does not implement a new paper model. It uses official package
estimators from the existing 50-model pool and official graph/data APIs from
NetworkX and scikit-learn to turn each monthly feature window into graph-aware
features. Graphs are rebuilt inside every rolling fold from training rows only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.covariance import GraphicalLasso
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
from sklearn.neighbors import NearestNeighbors

from run_corn_monthly_spike_official_50_three_heads import (
    ModelSpec,
    TargetStandardizer,
    TrainOnlyStandardizer,
    fit_classifier,
    fit_regressor,
    logit,
    parse_csv_list,
    predict_probability,
    r2_status,
    safe_ap,
    safe_auc,
    safe_balanced_accuracy,
    safe_r2,
    select_models,
    sigmoid,
    stable_name_offset,
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
class GraphSpec:
    name: str
    builder: Callable[[np.ndarray, np.ndarray, list[str]], np.ndarray]
    source: str


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    feature_set: str
    graph: GraphSpec
    transform: str
    model: ModelSpec
    lookback: int
    horizon: int


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
    parser.add_argument("--out-dir", default="outputs/corn_monthly_graph_pool")
    parser.add_argument("--date-col", default="month")
    parser.add_argument("--date-format", default="%y-%b")
    parser.add_argument("--price-col", default="dce_corn_close")
    parser.add_argument("--label-col", default="spike")
    parser.add_argument("--lookbacks", default="1,2,3")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--feature-sets", default="no_pca,with_pca")
    parser.add_argument(
        "--graph-builders",
        default="corr_abs_top5,spearman_abs_top5,knn_cosine_top5,glasso_alpha05,group_prefix",
    )
    parser.add_argument("--transforms", default="raw_smooth")
    parser.add_argument("--models", default="all")
    parser.add_argument("--min-train", type=int, default=48)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--max-experiments", type=int, default=0)
    parser.add_argument("--start-experiment", type=int, default=0)
    parser.add_argument("--allow-small-pool", action="store_true", help="Only for smoke tests; full runs keep the 50+ guard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--save-folds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv).expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).expanduser().resolve()
    if args.resume and out_root.exists() and (out_root / "manifest.json").exists():
        out_dir = out_root
    else:
        out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_monthly_csv(csv_path, args.date_col, args.date_format)
    feature_sets = make_feature_sets(
        df=df,
        date_col=args.date_col,
        price_col=args.price_col,
        label_col=args.label_col,
        requested=parse_csv_list(args.feature_sets),
        extra_exclude=parse_csv_list(args.extra_exclude),
    )
    graph_specs = select_graph_specs(parse_csv_list(args.graph_builders))
    transforms = parse_csv_list(args.transforms)
    model_specs = select_models(args.models, args.seed)
    lookbacks = [int(x) for x in parse_csv_list(args.lookbacks)]
    horizons = [int(x) for x in parse_csv_list(args.horizons)]

    experiments = build_experiments(
        feature_sets=feature_sets,
        graph_specs=graph_specs,
        transforms=transforms,
        model_specs=model_specs,
        lookbacks=lookbacks,
        horizons=horizons,
    )
    if args.start_experiment:
        experiments = experiments[args.start_experiment :]
    if args.max_experiments:
        experiments = experiments[: args.max_experiments]
    if len(experiments) < 50 and not args.allow_small_pool:
        raise ValueError(f"Graph pool is too small: {len(experiments)} experiments; requested at least 50.")

    samples_by_key: dict[tuple[str, int, int], dict[str, object]] = {}
    for feature_set, feature_cols in feature_sets.items():
        for lookback in lookbacks:
            for horizon in horizons:
                samples_by_key[(feature_set, lookback, horizon)] = make_samples(
                    df=df,
                    feature_cols=feature_cols,
                    date_col=args.date_col,
                    price_col=args.price_col,
                    label_col=args.label_col,
                    lookback=lookback,
                    horizon=horizon,
                )

    manifest = {
        "run_id": out_dir.name,
        "csv": str(csv_path),
        "rows": int(len(df)),
        "date_min": str(df[args.date_col].min().date()),
        "date_max": str(df[args.date_col].max().date()),
        "target_rule": "direct price direction: price[t+horizon] > price[t]; last rows without future price are dropped",
        "price_col": args.price_col,
        "label_col": args.label_col,
        "lookbacks": lookbacks,
        "horizons": horizons,
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "feature_columns": feature_sets,
        "graph_builders": [g.name for g in graph_specs],
        "transforms": transforms,
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
        "experiment_count": len(experiments),
        "min_train": args.min_train,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "max_origins": args.max_origins,
        "decision_rules": {
            "cls": "fixed probability threshold 0.5; no validation/test threshold tuning",
            "reg": "fixed return threshold 0.0 after inverse-transform; no validation/test threshold tuning",
        },
        "leakage_controls": [
            "future/target/next/lead columns are excluded from features",
            "dce_corn_close_next_month and its return are never used as features",
            "graph adjacency is rebuilt inside every fold using train rows only",
            "feature scaler is fitted on train rows only before graph construction",
            "regression target scaler is fitted on train rows only and inverse-transformed before R2 checks",
            "test_rows defaults to 1 and each test anchor is after the fold cutoff",
            "validation rows are historical known rows and are not used for threshold selection in this script",
        ],
        "official_sources": official_sources(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = load_completed_experiments(out_dir) if args.resume else set()
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    errors: list[dict[str, object]] = load_errors(out_dir) if args.resume else []
    if args.resume:
        old_predictions = out_dir / "checkpoint_rolling_predictions.csv"
        if old_predictions.exists():
            predictions = pd.read_csv(old_predictions).to_dict("records")

    feature_cache: dict[tuple[str, int, int, str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
    for exp_idx, exp in enumerate(experiments, start=args.start_experiment):
        if exp.experiment_id in completed:
            print(f"[skip] {exp_idx} {exp.experiment_id}", flush=True)
            continue
        samples = samples_by_key[(exp.feature_set, exp.lookback, exp.horizon)]
        origins = make_rolling_origins(
            samples=samples,
            min_train=args.min_train,
            val_size=args.val_size,
            test_size=args.test_size,
            step_size=args.step_size,
            max_origins=args.max_origins,
        )
        print(
            f"[run] {exp_idx + 1}/{args.start_experiment + len(experiments)} "
            f"{exp.experiment_id} origins={len(origins)}",
            flush=True,
        )
        try:
            exp_predictions, exp_folds, exp_errors = run_experiment(
                exp=exp,
                samples=samples,
                origins=origins,
                seed=args.seed,
                feature_cache=feature_cache,
            )
            predictions.extend(exp_predictions)
            folds.extend(exp_folds)
            errors.extend(exp_errors)
        except Exception as exc:  # noqa: BLE001 - long benchmark should keep going.
            errors.append(
                {
                    "experiment_id": exp.experiment_id,
                    "model": exp.model.name,
                    "feature_set": exp.feature_set,
                    "graph_builder": exp.graph.name,
                    "transform": exp.transform,
                    "lookback_months": exp.lookback,
                    "horizon_months": exp.horizon,
                    "origin_id": None,
                    "phase": "experiment",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        write_checkpoints(out_dir, predictions, folds, errors, args.save_folds, manifest, exp_idx, len(experiments))

    if not predictions:
        raise RuntimeError("No graph benchmark predictions were produced.")
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(out_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize(pred_df)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    model_summary = summarize_model_level(summary_df)
    model_summary.to_csv(out_dir / "model_level_summary.csv", index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "model_errors.csv", index=False, encoding="utf-8-sig")
    if args.save_folds and folds:
        pd.DataFrame(folds).to_csv(out_dir / "folds.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, summary_df, model_summary, manifest, errors)

    print("\n=== Best Graph Single Experiments ===")
    display = [
        "experiment_id",
        "head",
        "n_predictions",
        "accuracy",
        "balanced_accuracy",
        "auc",
        "average_precision",
        "reg_price_r2",
        "r2_status",
    ]
    print(summary_df.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False)[display].head(20).to_string(index=False))
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


def make_feature_sets(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    label_col: str,
    requested: list[str],
    extra_exclude: list[str],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in requested:
        include_pca = name in {"with_pca", "all_numeric"}
        selected, _ = select_feature_columns(df, date_col, price_col, label_col, include_pca, extra_exclude)
        if name == "pca_only":
            selected = [col for col in selected if col.lower().startswith("pca_") or col == price_col]
        if price_col not in selected:
            selected = [*selected, price_col]
        out[name] = move_price_last(selected, price_col)
    return out


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


def move_price_last(feature_cols: list[str], price_col: str) -> list[str]:
    return [col for col in feature_cols if col != price_col] + [price_col]


def make_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str,
    price_col: str,
    label_col: str,
    lookback: int,
    horizon: int,
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
        if not np.isfinite(window).any() or not np.isfinite(direct_return):
            continue
        x_rows.append(window)
        y_rows.append(int(direct_return > 0.0))
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
        "feature_cols": feature_cols,
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
            if len(test_idx) == test_size:
                origins.append((trainval_idx[:-val_size], trainval_idx[-val_size:], test_idx, origin_id))
                origin_id += 1
                if max_origins and len(origins) >= max_origins:
                    break
                cursor = int(test_idx[-1]) + step_size
                continue
        cursor += 1
    return origins


def select_graph_specs(requested: list[str]) -> list[GraphSpec]:
    specs = {
        "corr_abs_top3": GraphSpec("corr_abs_top3", lambda x, y, cols: corr_graph(x, top_k=3, rank=False), "numpy.corrcoef"),
        "corr_abs_top5": GraphSpec("corr_abs_top5", lambda x, y, cols: corr_graph(x, top_k=5, rank=False), "numpy.corrcoef"),
        "corr_abs_top8": GraphSpec("corr_abs_top8", lambda x, y, cols: corr_graph(x, top_k=8, rank=False), "numpy.corrcoef"),
        "spearman_abs_top5": GraphSpec("spearman_abs_top5", lambda x, y, cols: corr_graph(x, top_k=5, rank=True), "pandas.rank + numpy.corrcoef"),
        "knn_cosine_top5": GraphSpec("knn_cosine_top5", lambda x, y, cols: knn_feature_graph(x, top_k=5, metric="cosine"), "sklearn.neighbors.NearestNeighbors"),
        "knn_euclidean_top5": GraphSpec("knn_euclidean_top5", lambda x, y, cols: knn_feature_graph(x, top_k=5, metric="euclidean"), "sklearn.neighbors.NearestNeighbors"),
        "glasso_alpha05": GraphSpec("glasso_alpha05", lambda x, y, cols: glasso_graph(x, alpha=0.05, top_k=5), "sklearn.covariance.GraphicalLasso"),
        "glasso_alpha10": GraphSpec("glasso_alpha10", lambda x, y, cols: glasso_graph(x, alpha=0.10, top_k=5), "sklearn.covariance.GraphicalLasso"),
        "group_prefix": GraphSpec("group_prefix", lambda x, y, cols: group_prefix_graph(cols), "NetworkX from domain feature groups"),
    }
    missing = [name for name in requested if name not in specs]
    if missing:
        raise ValueError(f"Unknown graph builder(s): {missing}")
    return [specs[name] for name in requested]


def corr_graph(x_train_flat: np.ndarray, top_k: int, rank: bool) -> np.ndarray:
    x = np.asarray(x_train_flat, dtype=np.float64)
    if rank:
        x = pd.DataFrame(x).rank(axis=0).to_numpy(dtype=np.float64)
    corr = np.corrcoef(x, rowvar=False)
    weights = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(weights, 0.0)
    return topk_symmetric(weights, top_k)


def knn_feature_graph(x_train_flat: np.ndarray, top_k: int, metric: str) -> np.ndarray:
    x = np.asarray(x_train_flat, dtype=np.float64).T
    n_features = x.shape[0]
    if n_features <= 1:
        return np.zeros((n_features, n_features), dtype=np.float64)
    n_neighbors = min(top_k + 1, n_features)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    weights = np.zeros((n_features, n_features), dtype=np.float64)
    finite_dist = distances[np.isfinite(distances)]
    scale = float(np.nanmedian(finite_dist)) if len(finite_dist) else 1.0
    scale = scale if scale > 1e-12 else 1.0
    for i in range(n_features):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            if metric == "cosine":
                weight = max(0.0, 1.0 - float(dist))
            else:
                weight = math.exp(-float(dist) / scale)
            weights[i, int(j)] = weight
    return np.maximum(weights, weights.T)


def glasso_graph(x_train_flat: np.ndarray, alpha: float, top_k: int) -> np.ndarray:
    x = np.asarray(x_train_flat, dtype=np.float64)
    try:
        model = GraphicalLasso(alpha=alpha, max_iter=100, assume_centered=False)
        model.fit(x)
        weights = np.abs(model.precision_)
        np.fill_diagonal(weights, 0.0)
        return topk_symmetric(np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), top_k)
    except Exception:
        return corr_graph(x_train_flat, top_k=top_k, rank=False)


def group_prefix_graph(feature_cols: list[str]) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for idx, col in enumerate(feature_cols):
        groups.setdefault(feature_group(col), []).append(idx)
    n_features = len(feature_cols)
    weights = np.zeros((n_features, n_features), dtype=np.float64)
    for members in groups.values():
        for i in members:
            for j in members:
                if i != j:
                    weights[i, j] = 1.0
    return weights


def feature_group(col: str) -> str:
    lower = col.lower()
    if lower.startswith("pca_"):
        return "pca"
    for prefix in (
        "dce_corn_starch",
        "dce_corn",
        "cbot_corn",
        "hlj_",
        "jilin_",
        "inner_mongolia_",
        "liaoning_",
        "ne_",
        "china_",
        "us_",
    ):
        if lower.startswith(prefix):
            return prefix.strip("_")
    return lower.split("_")[0]


def topk_symmetric(weights: np.ndarray, top_k: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).copy()
    np.fill_diagonal(w, 0.0)
    n_features = w.shape[0]
    keep = np.zeros_like(w)
    for i in range(n_features):
        row = w[i].copy()
        if not np.isfinite(row).any():
            continue
        candidates = np.argsort(row)[-(min(top_k, n_features - 1)) :]
        for j in candidates:
            if i != int(j) and row[j] > 0:
                keep[i, int(j)] = row[j]
    return np.maximum(keep, keep.T)


def normalized_adjacency(weights: np.ndarray) -> np.ndarray:
    a = np.asarray(weights, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Adjacency must be square, got {a.shape}")
    a = np.nan_to_num(np.maximum(a, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    a = a + np.eye(a.shape[0], dtype=np.float64)
    degree = np.sum(a, axis=1)
    degree = np.where(degree > 1e-12, degree, 1.0)
    inv_sqrt = 1.0 / np.sqrt(degree)
    return (inv_sqrt[:, None] * a * inv_sqrt[None, :]).astype(np.float32)


def transform_graph_features(x: np.ndarray, adjacency: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    a = np.asarray(adjacency, dtype=np.float32)
    smooth = np.einsum("ij,btj->bti", a, x)
    if transform == "raw_smooth":
        parts = [x, smooth]
    elif transform == "smooth_only":
        parts = [smooth]
    elif transform == "raw_residual":
        parts = [x, x - smooth]
    elif transform == "raw_diffuse2":
        diffuse2 = np.einsum("ij,btj->bti", a, smooth)
        parts = [x, smooth, diffuse2]
    elif transform == "centrality_pool":
        weights = graph_centrality_weights(a)
        pooled = np.einsum("btf,fk->btk", x, weights)
        return np.concatenate([x.reshape(x.shape[0], -1), pooled.reshape(x.shape[0], -1)], axis=1).astype(np.float32)
    elif transform == "graph_stats":
        smooth_delta = x - smooth
        stats = np.concatenate(
            [
                x.mean(axis=1),
                x[:, -1, :],
                smooth.mean(axis=1),
                smooth_delta.mean(axis=1),
                smooth_delta.std(axis=1),
            ],
            axis=1,
        )
        return stats.astype(np.float32)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return np.concatenate([part.reshape(x.shape[0], -1) for part in parts], axis=1).astype(np.float32)


def graph_centrality_weights(adjacency: np.ndarray) -> np.ndarray:
    raw = np.asarray(adjacency, dtype=np.float64)
    graph = nx.from_numpy_array(np.maximum(raw - np.eye(raw.shape[0]), 0.0))
    degree = np.asarray([value for _, value in nx.degree_centrality(graph).items()], dtype=np.float64)
    try:
        pagerank_dict = nx.pagerank(graph, weight="weight", max_iter=100)
        pagerank = np.asarray([pagerank_dict[i] for i in range(raw.shape[0])], dtype=np.float64)
    except Exception:
        pagerank = degree.copy()
    try:
        eigen_dict = nx.eigenvector_centrality_numpy(graph, weight="weight")
        eigen = np.asarray([eigen_dict[i] for i in range(raw.shape[0])], dtype=np.float64)
    except Exception:
        eigen = degree.copy()
    weights = np.column_stack([degree, pagerank, eigen])
    denom = np.sum(np.abs(weights), axis=0, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    return (weights / denom).astype(np.float32)


def build_experiments(
    feature_sets: dict[str, list[str]],
    graph_specs: list[GraphSpec],
    transforms: list[str],
    model_specs: list[ModelSpec],
    lookbacks: list[int],
    horizons: list[int],
) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    for feature_set in feature_sets:
        for lookback in lookbacks:
            for horizon in horizons:
                for graph in graph_specs:
                    for transform in transforms:
                        for model in model_specs:
                            experiment_id = (
                                f"{model.name}|{feature_set}|{graph.name}|{transform}|"
                                f"lb{lookback}|h{horizon}"
                            )
                            experiments.append(
                                ExperimentSpec(
                                    experiment_id=experiment_id,
                                    feature_set=feature_set,
                                    graph=graph,
                                    transform=transform,
                                    model=model,
                                    lookback=lookback,
                                    horizon=horizon,
                                )
                            )
    return experiments


def run_experiment(
    exp: ExperimentSpec,
    samples: dict[str, object],
    origins,
    seed: int,
    feature_cache: dict[tuple[str, int, int, str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    x = samples["X"]
    y = samples["y"]
    returns = samples["returns"]
    meta = samples["meta"]
    feature_cols = samples["feature_cols"]
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for train_idx, val_idx, test_idx, origin_id in origins:
        fold_seed = seed + stable_name_offset(exp.experiment_id) + 101 * exp.lookback + 17 * exp.horizon + origin_id
        feature_key = (exp.feature_set, exp.lookback, exp.horizon, exp.graph.name, exp.transform, origin_id)
        try:
            if feature_key not in feature_cache:
                feature_cache[feature_key] = build_fold_feature_matrices(
                    x=x,
                    y=y,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    feature_cols=feature_cols,
                    graph=exp.graph,
                    transform=exp.transform,
                )
            x_train, x_val, x_test, graph_info = feature_cache[feature_key]
            clf = safe_fit_classifier(exp.model, x_train, y[train_idx], fold_seed)
            cls_val_prob = predict_probability(clf, x_val)
            cls_test_prob = predict_probability(clf, x_test)

            y_scaler = TargetStandardizer().fit(returns[train_idx])
            reg = safe_fit_regressor(exp.model, x_train, y_scaler.transform(returns[train_idx]), fold_seed)
            reg_val_return = y_scaler.inverse_transform(np.asarray(reg.predict(x_val), dtype=float))
            reg_test_return = y_scaler.inverse_transform(np.asarray(reg.predict(x_test), dtype=float))
            return_scale = max(float(np.nanstd(returns[train_idx])), 1e-4)
            reg_test_prob = sigmoid(reg_test_return / return_scale)

            predicted_price = meta.iloc[test_idx]["anchor_price"].to_numpy(float) * (1.0 + reg_test_return)
            actual_price = meta.iloc[test_idx]["actual_price"].to_numpy(float)
            anchor_price = meta.iloc[test_idx]["anchor_price"].to_numpy(float)
            reg_price_r2 = safe_r2(actual_price, predicted_price)
            naive_price_r2 = safe_r2(actual_price, anchor_price)
            val_ba_05 = safe_balanced_accuracy(y[val_idx], cls_val_prob >= 0.5)
            val_reg_ba_0 = safe_balanced_accuracy(y[val_idx], reg_val_return >= 0.0)
            fold_row = {
                "experiment_id": exp.experiment_id,
                "model": exp.model.name,
                "family": exp.model.family,
                "package": exp.model.package,
                "feature_set": exp.feature_set,
                "graph_builder": exp.graph.name,
                "graph_source": exp.graph.source,
                "transform": exp.transform,
                "lookback_months": exp.lookback,
                "horizon_months": exp.horizon,
                "origin_id": origin_id,
                "cutoff_month": str(meta.iloc[val_idx[-1]]["target_month"]),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "test_rows": len(test_idx),
                "graph_edges": graph_info["graph_edges"],
                "graph_density": graph_info["graph_density"],
                "graph_components": graph_info["graph_components"],
                "val_cls_ba_at_05": val_ba_05,
                "val_reg_ba_at_0": val_reg_ba_0,
                "reg_price_r2_fold": reg_price_r2,
                "naive_price_r2_fold": naive_price_r2,
            }
            folds.append(fold_row)
            head_payloads = [
                ("cls", cls_test_prob, cls_test_prob >= 0.5, 0.5, cls_test_prob, np.full(len(test_idx), np.nan)),
                ("reg", reg_test_prob, reg_test_return >= 0.0, 0.0, reg_test_prob, reg_test_return),
            ]
            for offset, sample_idx in enumerate(test_idx):
                base = {
                    "experiment_id": exp.experiment_id,
                    "model": exp.model.name,
                    "family": exp.model.family,
                    "package": exp.model.package,
                    "classifier_loss": exp.model.classifier_loss,
                    "regressor_loss": exp.model.regressor_loss,
                    "feature_set": exp.feature_set,
                    "graph_builder": exp.graph.name,
                    "graph_source": exp.graph.source,
                    "transform": exp.transform,
                    "lookback_months": exp.lookback,
                    "horizon_months": exp.horizon,
                    "origin_id": origin_id,
                    "anchor_month": meta.iloc[sample_idx]["anchor_month"],
                    "target_month": meta.iloc[sample_idx]["target_month"],
                    "anchor_idx": int(meta.iloc[sample_idx]["anchor_idx"]),
                    "target_idx": int(meta.iloc[sample_idx]["target_idx"]),
                    "anchor_price": float(anchor_price[offset]),
                    "actual_price": float(actual_price[offset]),
                    "predicted_price_reg": float(predicted_price[offset]),
                    "actual_return": float(meta.iloc[sample_idx]["actual_return"]),
                    "actual_direction": int(y[sample_idx]),
                    "target_existing_spike": int(meta.iloc[sample_idx]["target_existing_spike"]),
                    "reg_price_r2_fold": float(reg_price_r2),
                    "naive_price_r2_fold": float(naive_price_r2),
                    "train_rows": len(train_idx),
                    "val_rows": len(val_idx),
                    "test_rows": len(test_idx),
                    "graph_edges": graph_info["graph_edges"],
                    "graph_density": graph_info["graph_density"],
                    "graph_components": graph_info["graph_components"],
                }
                for head, prob, pred, threshold, probability, predicted_return in head_payloads:
                    row = dict(base)
                    row.update(
                        {
                            "head": head,
                            "threshold": float(threshold),
                            "predicted_direction": int(pred[offset]),
                            "predicted_probability": float(probability[offset]),
                            "predicted_return_reg": float(predicted_return[offset]) if np.isfinite(predicted_return[offset]) else np.nan,
                        }
                    )
                    predictions.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "experiment_id": exp.experiment_id,
                    "model": exp.model.name,
                    "feature_set": exp.feature_set,
                    "graph_builder": exp.graph.name,
                    "transform": exp.transform,
                    "lookback_months": exp.lookback,
                    "horizon_months": exp.horizon,
                    "origin_id": origin_id,
                    "phase": "fold",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    return predictions, folds, errors


def build_fold_feature_matrices(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    graph: GraphSpec,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    scaler = TrainOnlyStandardizer().fit(x[train_idx])
    x_train_seq = scaler.transform(x[train_idx])
    x_val_seq = scaler.transform(x[val_idx])
    x_test_seq = scaler.transform(x[test_idx])
    x_train_flat_for_graph = x_train_seq.reshape(-1, x_train_seq.shape[-1])
    weights = graph.builder(x_train_flat_for_graph, y[train_idx], feature_cols)
    adjacency = normalized_adjacency(weights)
    x_train = transform_graph_features(x_train_seq, adjacency, transform)
    x_val = transform_graph_features(x_val_seq, adjacency, transform)
    x_test = transform_graph_features(x_test_seq, adjacency, transform)
    graph_plain = np.asarray(weights, dtype=float)
    graph_plain = np.maximum(graph_plain, graph_plain.T)
    np.fill_diagonal(graph_plain, 0.0)
    graph_nx = nx.from_numpy_array(graph_plain)
    graph_info = {
        "graph_edges": int(graph_nx.number_of_edges()),
        "graph_density": float(nx.density(graph_nx)) if graph_plain.shape[0] > 1 else 0.0,
        "graph_components": int(nx.number_connected_components(graph_nx)) if graph_plain.shape[0] else 0,
    }
    return x_train, x_val, x_test, graph_info


def safe_fit_classifier(spec: ModelSpec, x: np.ndarray, y: np.ndarray, seed: int) -> BaseEstimator:
    if len(np.unique(y)) < 2:
        return ConstantClassifier(float(np.mean(y)))
    try:
        return fit_classifier(spec, x, y, seed)
    except Exception:
        return ConstantClassifier(float(np.mean(y)))


def safe_fit_regressor(spec: ModelSpec, x: np.ndarray, y_scaled: np.ndarray, seed: int) -> BaseEstimator:
    if np.nanstd(y_scaled) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y_scaled)))
    try:
        return fit_regressor(spec, x, y_scaled, seed)
    except Exception:
        return ConstantRegressor(float(np.nanmean(y_scaled)))


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "experiment_id",
        "model",
        "family",
        "package",
        "feature_set",
        "graph_builder",
        "transform",
        "lookback_months",
        "horizon_months",
        "head",
        "classifier_loss",
        "regressor_loss",
    ]
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
                "first_target_month": str(group["target_month"].min()),
                "last_target_month": str(group["target_month"].max()),
                "mean_train_rows": float(group["train_rows"].mean()),
                "min_train_rows": int(group["train_rows"].min()),
                "max_train_rows": int(group["train_rows"].max()),
                "mean_graph_edges": float(group["graph_edges"].mean()),
                "mean_graph_density": float(group["graph_density"].mean()),
                "mean_graph_components": float(group["graph_components"].mean()),
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
    return best.sort_values(["head", "balanced_accuracy", "auc", "average_precision"], ascending=[True, False, False, False])


def write_checkpoints(
    out_dir: Path,
    predictions: list[dict[str, object]],
    folds: list[dict[str, object]],
    errors: list[dict[str, object]],
    save_folds: bool,
    manifest: dict[str, object],
    exp_idx: int,
    total: int,
) -> None:
    progress = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_or_attempted_experiments": exp_idx + 1,
        "total_experiments_in_this_invocation": total,
        "prediction_rows": len(predictions),
        "error_rows": len(errors),
    }
    (out_dir / "PROGRESS.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    if not predictions:
        if errors:
            pd.DataFrame(errors).to_csv(out_dir / "model_errors.csv", index=False, encoding="utf-8-sig")
        return
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(out_dir / "checkpoint_rolling_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df = summarize(pred_df)
    summary_df.to_csv(out_dir / "checkpoint_summary_metrics.csv", index=False, encoding="utf-8-sig")
    summarize_model_level(summary_df).to_csv(out_dir / "checkpoint_model_level_summary.csv", index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "model_errors.csv", index=False, encoding="utf-8-sig")
    if save_folds and folds:
        pd.DataFrame(folds).to_csv(out_dir / "checkpoint_folds.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, summary_df, summarize_model_level(summary_df), manifest, errors, checkpoint=True)


def load_completed_experiments(out_dir: Path) -> set[str]:
    pred_path = out_dir / "checkpoint_rolling_predictions.csv"
    if not pred_path.exists():
        return set()
    df = pd.read_csv(pred_path, usecols=["experiment_id"])
    return set(df["experiment_id"].dropna().astype(str).unique())


def load_errors(out_dir: Path) -> list[dict[str, object]]:
    path = out_dir / "model_errors.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    manifest: dict[str, object],
    errors: list[dict[str, object]],
    checkpoint: bool = False,
) -> None:
    title = "Corn Monthly Graph Feature Rolling Benchmark"
    if checkpoint:
        title += " Checkpoint"
    lines = [
        f"# {title}",
        "",
        f"- CSV: `{manifest['csv']}`",
        f"- Date range: `{manifest['date_min']}` to `{manifest['date_max']}`",
        f"- Target: `{manifest['target_rule']}`",
        f"- Graph experiment count: `{manifest['experiment_count']}`",
        f"- Models from official 50-pool invocation: `{len(manifest['models'])}`",
        f"- Feature sets: `{manifest['feature_sets']}`",
        f"- Graph builders: `{', '.join(manifest['graph_builders'])}`",
        f"- Transforms: `{', '.join(manifest['transforms'])}`",
        "",
        "## Leakage Controls",
        "",
    ]
    for item in manifest["leakage_controls"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Official Sources", ""])
    for name, url in manifest["official_sources"].items():
        lines.append(f"- {name}: {url}")
    lines.extend(["", "## Top Overall", ""])
    display_cols = [
        "experiment_id",
        "head",
        "n_predictions",
        "accuracy",
        "balanced_accuracy",
        "auc",
        "average_precision",
        "reg_price_r2",
        "r2_status",
        "first_target_month",
        "last_target_month",
    ]
    lines.append(
        markdown_table(
            summary.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False)[display_cols].head(30)
        )
    )
    lines.extend(["", "## Best Per Horizon/Head", ""])
    best_rows = []
    for (_, _), group in summary.groupby(["horizon_months", "head"]):
        best_rows.append(group.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False).iloc[0])
    if best_rows:
        lines.append(markdown_table(pd.DataFrame(best_rows)[display_cols]))
    lines.extend(["", "## Best Per Base Estimator/Head", ""])
    lines.append(markdown_table(model_summary[display_cols].head(80)))
    if errors:
        lines.extend(["", "## Errors", "", f"- Error rows: `{len(errors)}`; see `model_errors.csv`."])
    lines.extend(
        [
            "",
            "Outputs:",
            "",
            "- `summary_metrics.csv` or `checkpoint_summary_metrics.csv`",
            "- `model_level_summary.csv` or `checkpoint_model_level_summary.csv`",
            "- `rolling_predictions.csv` or `checkpoint_rolling_predictions.csv`",
            "- `model_errors.csv`",
            "- `manifest.json`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def official_sources() -> dict[str, str]:
    return {
        "NetworkX graph algorithms": "https://networkx.org/documentation/stable/reference/algorithms/index.html",
        "NetworkX from_numpy_array": "https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_numpy_array.html",
        "scikit-learn GraphicalLasso": "https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html",
        "scikit-learn NearestNeighbors": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html",
        "scikit-learn supervised estimators": "https://scikit-learn.org/stable/supervised_learning.html",
        "LightGBM Python API": "https://lightgbm.readthedocs.io/en/stable/Python-API.html",
        "XGBoost Python API": "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
        "CatBoost Python reference": "https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier",
        "PyTorch Geometric Temporal reference for follow-up true GNNs": "https://pytorch-geometric-temporal.readthedocs.io/",
        "Torch Spatiotemporal reference for follow-up true GNNs": "https://torch-spatiotemporal.readthedocs.io/",
        "Graph WaveNet official code reference": "https://github.com/nnzhan/Graph-WaveNet",
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows yet._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
