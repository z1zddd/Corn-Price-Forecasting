#!/usr/bin/env python3
"""Search leakage-aware ensemble methods over rolling prediction streams.

The valid methods in this script are prequential/walk-forward: each target
month is predicted using only base-model predictions and labels from earlier
target months. Diagnostic oracle rows are written separately and must not be
reported as formal backtest results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import compute_all_metrics  # noqa: E402


@dataclass(frozen=True)
class EnsembleResult:
    name: str
    horizon: int
    validation_mode: str
    method_family: str
    params: dict
    predictions: pd.DataFrame
    metrics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="all_rolling_predictions.csv or .csv.gz")
    parser.add_argument("--output-dir", default="experiments/ensemble_search")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--min-history", type=int, default=12)
    parser.add_argument("--preset", choices=["fast", "broad"], default="broad")
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated method families: simple,topk,weighted,online,dynamic,stacking,exhaustive,forward,oracle or all",
    )
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--max-logistic-candidates", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.enabled_families = parse_families(args.families)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(args.predictions)
    horizons = [int(value.strip()) for value in args.horizons.split(",") if value.strip()]
    all_rows: list[dict] = []
    all_inventory: list[dict] = []
    best_payloads: dict[str, pd.DataFrame] = {}
    config_payload = {
        "predictions": str(args.predictions),
        "horizons": horizons,
        "min_candidate_coverage": args.min_candidate_coverage,
        "min_history": args.min_history,
        "preset": args.preset,
        "families": sorted(args.enabled_families) if args.enabled_families is not None else "all",
        "validation_note": "valid_walk_forward uses only earlier target months; diagnostic_oracle is leaky upper-bound only.",
    }

    for horizon in horizons:
        matrix = build_matrix(predictions, horizon, args.min_candidate_coverage)
        all_inventory.extend(matrix["inventory"].to_dict("records"))
        results = run_horizon_search(matrix, args)
        if not results:
            continue
        summary = pd.DataFrame([result_row(result) for result in results])
        summary = summary.sort_values(["validation_mode", "BalancedAcc", "AUC", "AP"], ascending=[True, False, False, False])
        summary.to_csv(output_dir / f"h{horizon}_ensemble_search_summary.csv", index=False)
        for result in results:
            if result.validation_mode == "valid_walk_forward":
                all_rows.append(result_row(result))
        best_valid = max(
            [result for result in results if result.validation_mode == "valid_walk_forward"],
            key=lambda item: (item.metrics["BalancedAcc"], item.metrics["AUC"], item.metrics["AP"]),
        )
        best_payloads[f"h{horizon}_{safe_name(best_valid.name)}"] = best_valid.predictions
        best_valid.predictions.to_csv(output_dir / f"h{horizon}_best_valid_predictions.csv", index=False)

    pd.DataFrame(all_inventory).to_csv(output_dir / "candidate_inventory.csv", index=False)
    leaderboard = pd.DataFrame(all_rows).sort_values(["horizon", "BalancedAcc", "AUC", "AP"], ascending=[True, False, False, False])
    leaderboard.to_csv(output_dir / "valid_walk_forward_leaderboard.csv", index=False)
    (output_dir / "search_config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(output_dir, leaderboard)
    print(leaderboard.head(40).to_string(index=False))
    print(f"results written to {output_dir}")


def read_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "model",
        "feature_set",
        "lookback_months",
        "horizon_months",
        "head",
        "anchor_month",
        "target_month",
        "actual_direction",
        "actual_return",
        "predicted_direction",
        "predicted_probability",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    df = df.copy()
    df["anchor_month"] = pd.to_datetime(df["anchor_month"])
    df["target_month"] = pd.to_datetime(df["target_month"])
    df["candidate_id"] = df.apply(candidate_id, axis=1)
    return df


def candidate_id(row: pd.Series) -> str:
    feature = "news" if row["feature_set"] == "with_news_precomputed_pca" else "nonews"
    return f"{row['model']}|{feature}|lb{int(row['lookback_months'])}|h{int(row['horizon_months'])}|{row['head']}"


def build_matrix(predictions: pd.DataFrame, horizon: int, min_coverage: float) -> dict:
    part = predictions.loc[predictions["horizon_months"].astype(int).eq(horizon)].copy()
    if part.empty:
        raise ValueError(f"No predictions found for horizon={horizon}")
    index_cols = ["anchor_month", "target_month", "horizon_months", "actual_direction", "actual_return"]
    probs = part.pivot_table(index=index_cols, columns="candidate_id", values="predicted_probability", aggfunc="first")
    votes = part.pivot_table(index=index_cols, columns="candidate_id", values="predicted_direction", aggfunc="first")
    probs = probs.sort_index().reset_index()
    votes = votes.sort_index().reset_index()
    candidate_cols = [column for column in probs.columns if column not in index_cols]
    coverage = probs[candidate_cols].notna().mean().sort_values(ascending=False)
    selected = coverage.loc[coverage >= min_coverage].index.tolist()
    if not selected:
        raise ValueError(f"No candidates meet coverage >= {min_coverage} for horizon={horizon}")
    info = (
        part.drop_duplicates("candidate_id")
        .set_index("candidate_id")[["model", "feature_set", "lookback_months", "horizon_months", "head", "family", "package"]]
        .reindex(selected)
        .reset_index()
    )
    info["coverage"] = info["candidate_id"].map(coverage).astype(float)
    return {
        "horizon": horizon,
        "base": probs[index_cols].copy(),
        "prob": probs[selected].astype(float),
        "vote": votes[selected].astype(float),
        "candidates": selected,
        "inventory": info,
    }


def parse_families(value: str) -> set[str] | None:
    if str(value).strip().lower() == "all":
        return None
    families = {item.strip().lower() for item in str(value).split(",") if item.strip()}
    allowed = {"simple", "topk", "weighted", "online", "dynamic", "stacking", "exhaustive", "forward", "oracle"}
    unknown = sorted(families - allowed)
    if unknown:
        raise ValueError(f"Unknown ensemble families: {unknown}")
    return families


def enabled(args: argparse.Namespace, family: str) -> bool:
    families = getattr(args, "enabled_families", parse_families(args.families))
    return families is None or family in families


def run_horizon_search(matrix: dict, args: argparse.Namespace) -> list[EnsembleResult]:
    results: list[EnsembleResult] = []
    prob = matrix["prob"]
    vote = matrix["vote"]
    candidates = matrix["candidates"]
    y = matrix["base"]["actual_direction"].to_numpy(int)

    if enabled(args, "simple"):
        print(f"[h{matrix['horizon']}] simple", flush=True)
        simple_methods = [
            ("all_soft_mean", "soft", "mean"),
            ("all_soft_median", "soft", "median"),
            ("all_hard_mean", "hard", "mean"),
            ("all_hard_strict", "hard", "strict"),
        ]
        for name, source, reducer in simple_methods:
            score = aggregate_frame(prob if source == "soft" else vote, reducer=reducer)
            results.append(make_result(name, matrix, "valid_walk_forward", "simple_static", {"source": source, "reducer": reducer}, score, threshold=0.5, args=args))

    if args.preset == "fast":
        windows = [18, 9999]
        ks = [3, 6, 10, 20]
        topn_values = [10, 20]
        ranking_metrics = ["ba", "brier"]
    else:
        windows = [12, 18, 24, 36, 9999]
        ks = [3, 5, 6, 9, 12, 15, 20, 30, 50]
        topn_values = [10, 20, 40, 80]
        ranking_metrics = ["ba", "auc", "ap", "brier"]

    if enabled(args, "topk"):
        print(f"[h{matrix['horizon']}] rolling top-k", flush=True)
        rank_cache = build_rank_cache(prob, vote, y, ranking_metrics, windows)
        for metric in ranking_metrics:
            for window in windows:
                for k in ks:
                    if k > len(candidates):
                        continue
                    for source in ["hard", "soft"]:
                        for threshold_mode in ["fixed", "rolling_ba"]:
                            score = rolling_topk_score(prob, vote, y, metric=metric, k=k, window=window, source=source, threshold_mode=threshold_mode, min_history=args.min_history, rank_cache=rank_cache)
                            result_name = f"rolling_top{k}_{metric}_{source}_w{window}_{threshold_mode}"
                            results.append(make_result(result_name, matrix, "valid_walk_forward", "rolling_topk", {"metric": metric, "k": k, "window": window, "source": source, "threshold_mode": threshold_mode}, score, threshold=0.5, args=args))

    if enabled(args, "weighted"):
        print(f"[h{matrix['horizon']}] rolling weighted", flush=True)
        rank_cache = build_rank_cache(prob, vote, y, ["ba", "brier"], windows)
        for metric in ["ba", "brier"]:
            for window in windows:
                for k in ks:
                    if k > len(candidates):
                        continue
                    for source in ["hard", "soft"]:
                        score = rolling_weighted_score(prob, vote, y, metric=metric, k=k, window=window, source=source, min_history=args.min_history, rank_cache=rank_cache)
                        result_name = f"rolling_weighted_top{k}_{metric}_{source}_w{window}"
                        results.append(make_result(result_name, matrix, "valid_walk_forward", "rolling_weighted", {"metric": metric, "k": k, "window": window, "source": source}, score, threshold=0.5, args=args))

    if enabled(args, "online"):
        print(f"[h{matrix['horizon']}] online weighted majority", flush=True)
        for beta in [0.25, 0.5, 0.75, 0.9]:
            for source in ["hard", "soft"]:
                score = online_weighted_majority(prob, vote, y, beta=beta, source=source)
                result_name = f"online_weighted_majority_beta{beta}_{source}"
                results.append(make_result(result_name, matrix, "valid_walk_forward", "online_weighted_majority", {"beta": beta, "source": source}, score, threshold=0.5, args=args))

    dynamic_local_ks = [5, 10] if args.preset == "fast" else [5, 10, 15, 20]
    dynamic_topms = [3, 5] if args.preset == "fast" else [3, 5, 8, 12]
    if enabled(args, "dynamic"):
        print(f"[h{matrix['horizon']}] dynamic local selection", flush=True)
        for topn in topn_values:
            if topn > len(candidates):
                continue
            for local_k in dynamic_local_ks:
                for topm in dynamic_topms:
                    if topm > topn:
                        continue
                    for source in ["hard", "soft"]:
                        score = dynamic_local_selection(prob, vote, y, topn=topn, local_k=local_k, topm=topm, source=source, min_history=args.min_history)
                        result_name = f"dynamic_local_topn{topn}_nn{local_k}_topm{topm}_{source}"
                        results.append(make_result(result_name, matrix, "valid_walk_forward", "dynamic_local_selection", {"topn": topn, "local_k": local_k, "topm": topm, "source": source}, score, threshold=0.5, args=args))

    if enabled(args, "stacking"):
        print(f"[h{matrix['horizon']}] logistic stacking", flush=True)
        logistic_windows = [9999] if args.preset == "fast" else windows
        logistic_cs = [1.0] if args.preset == "fast" else [0.1, 1.0, 10.0]
        for topn in topn_values:
            topn = min(topn, args.max_logistic_candidates, len(candidates))
            if topn < 3:
                continue
            for window in logistic_windows:
                for c_value in logistic_cs:
                    score = rolling_logistic_stack(prob, vote, y, topn=topn, window=window, c_value=c_value, min_history=max(args.min_history, 16))
                    result_name = f"stack_logistic_topn{topn}_w{window}_C{c_value}"
                    results.append(make_result(result_name, matrix, "valid_walk_forward", "stacking_logistic", {"topn": topn, "window": window, "C": c_value}, score, threshold=0.5, args=args))

    if enabled(args, "exhaustive"):
        print(f"[h{matrix['horizon']}] exhaustive subsets", flush=True)
        exhaustive_topns = [8, 10] if args.preset == "fast" else [8, 10, 12]
        exhaustive_ks = [3, 5] if args.preset == "fast" else [3, 4, 5, 6]
        exhaustive_windows = [9999] if args.preset == "fast" else [18, 36, 9999]
        for topn in exhaustive_topns:
            if topn > len(candidates):
                continue
            for k in exhaustive_ks:
                if k > topn:
                    continue
                for window in exhaustive_windows:
                    score = rolling_exhaustive_subset(prob, vote, y, topn=topn, k=k, window=window, min_history=args.min_history)
                    result_name = f"exhaustive_top{topn}_k{k}_hard_w{window}"
                    results.append(make_result(result_name, matrix, "valid_walk_forward", "exhaustive_subset", {"topn": topn, "k": k, "window": window}, score, threshold=0.5, args=args))

    if enabled(args, "forward"):
        print(f"[h{matrix['horizon']}] forward ensemble selection", flush=True)
        forward_topns = [20] if args.preset == "fast" else [20, 40, 80]
        forward_sizes = [3, 6] if args.preset == "fast" else [3, 6, 10, 15]
        forward_windows = [9999] if args.preset == "fast" else [18, 36, 9999]
        for topn in forward_topns:
            if topn > len(candidates):
                continue
            for size in forward_sizes:
                for window in forward_windows:
                    score = rolling_forward_selection(prob, y, topn=topn, size=size, window=window, min_history=args.min_history)
                    result_name = f"forward_selection_top{topn}_size{size}_soft_w{window}"
                    results.append(make_result(result_name, matrix, "valid_walk_forward", "forward_ensemble_selection", {"topn": topn, "size": size, "window": window}, score, threshold=0.5, args=args))

    if enabled(args, "oracle"):
        print(f"[h{matrix['horizon']}] diagnostic oracle", flush=True)
        for metric in ranking_metrics:
            for k in ks:
                if k <= len(candidates):
                    for source in ["hard", "soft"]:
                        score = oracle_topk_score(prob, vote, y, metric=metric, k=k, source=source)
                        name = f"oracle_top{k}_{metric}_{source}"
                        results.append(make_result(name, matrix, "diagnostic_oracle", "leaky_oracle_topk", {"metric": metric, "k": k, "source": source}, score, threshold=0.5, args=args))

    return results


def make_result(
    name: str,
    matrix: dict,
    validation_mode: str,
    method_family: str,
    params: dict,
    score: np.ndarray,
    *,
    threshold: float,
    args: argparse.Namespace,
) -> EnsembleResult:
    base = matrix["base"].copy()
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    if valid.sum() < 2:
        raise ValueError(f"{name} produced fewer than two valid predictions")
    pred = (score[valid] > threshold).astype(int)
    y = base.loc[valid, "actual_direction"].to_numpy(int)
    actual_returns = base.loc[valid, "actual_return"].to_numpy(float)
    out = pd.DataFrame(
        {
            "date": base.loc[valid, "target_month"].dt.date.astype(str),
            "anchor_date": base.loc[valid, "anchor_month"].dt.date.astype(str),
            "actual_label": y,
            "predicted_label": pred,
            "predicted_probability": score[valid],
            "direction_correct": (pred == y).astype(int),
            "actual_return": actual_returns,
            "strategy_return": np.where(pred == 1, actual_returns, -actual_returns),
            "model": name,
            "window_id": np.arange(valid.sum(), dtype=int),
            "test_date": base.loc[valid, "target_month"].dt.date.astype(str),
        }
    )
    out["equity"] = (1.0 + out["strategy_return"]).cumprod()
    if int(args.bootstrap) > 0:
        metrics = compute_all_metrics(
            y,
            score[valid],
            actual_returns,
            n_bootstrap=args.bootstrap,
            ci_level=args.ci_level,
            annualize=12,
        )
    else:
        metrics = compute_search_metrics(y, score[valid], actual_returns)
    metrics["coverage"] = float(valid.mean())
    metrics["n_predictions"] = int(valid.sum())
    return EnsembleResult(name, int(matrix["horizon"]), validation_mode, method_family, params, out, metrics)


def result_row(result: EnsembleResult) -> dict:
    return {
        "horizon": result.horizon,
        "method": result.name,
        "validation_mode": result.validation_mode,
        "method_family": result.method_family,
        **result.metrics,
        "params": json.dumps(result.params, ensure_ascii=False, sort_keys=True),
    }


def aggregate_frame(frame: pd.DataFrame, *, reducer: str) -> np.ndarray:
    values = frame.to_numpy(float)
    if reducer == "mean":
        return np.nanmean(values, axis=1)
    if reducer == "median":
        return np.nanmedian(values, axis=1)
    if reducer == "strict":
        return np.nanmean(values, axis=1)
    raise ValueError(f"Unknown reducer: {reducer}")


def build_rank_cache(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, metrics: list[str], windows: list[int]) -> dict[tuple[str, int, int], list[str]]:
    cache: dict[tuple[str, int, int], list[str]] = {}
    all_candidates = list(prob.columns)
    for metric in metrics:
        for window in windows:
            for idx in range(len(y)):
                hist = history_indices(idx, window)
                if len(hist) < 2:
                    cache[(metric, window, idx)] = all_candidates
                else:
                    cache[(metric, window, idx)] = rank_candidates(prob, vote, y, hist, all_candidates, metric)
    return cache


def rolling_topk_score(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, metric: str, k: int, window: int, source: str, threshold_mode: str, min_history: int, rank_cache: dict[tuple[str, int, int], list[str]] | None = None) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    base = prob if source == "soft" else vote
    for idx in range(len(y)):
        hist = history_indices(idx, window)
        current_cols = nonmissing_columns(base.iloc[idx])
        if len(hist) < min_history or not current_cols:
            scores[idx] = float(base.iloc[idx][current_cols].mean()) if current_cols else np.nan
            continue
        ranked_all = rank_cache.get((metric, window, idx), []) if rank_cache is not None else rank_candidates(prob, vote, y, hist, current_cols, metric)
        current_set = set(current_cols)
        ranked = [col for col in ranked_all if col in current_set]
        selected = ranked[:k]
        if not selected:
            scores[idx] = float(base.iloc[idx][current_cols].mean())
            continue
        raw_score = float(base.iloc[idx][selected].mean())
        if threshold_mode == "rolling_ba":
            hist_scores = base.iloc[hist][selected].mean(axis=1).to_numpy(float)
            threshold = best_threshold(y[hist], hist_scores)
            scores[idx] = 0.500001 if raw_score > threshold else 0.499999
        else:
            scores[idx] = raw_score
    return scores


def rolling_weighted_score(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, metric: str, k: int, window: int, source: str, min_history: int, rank_cache: dict[tuple[str, int, int], list[str]] | None = None) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    base = prob if source == "soft" else vote
    for idx in range(len(y)):
        hist = history_indices(idx, window)
        current_cols = nonmissing_columns(base.iloc[idx])
        if len(hist) < min_history or not current_cols:
            scores[idx] = float(base.iloc[idx][current_cols].mean()) if current_cols else np.nan
            continue
        ranked_all = rank_cache.get((metric, window, idx), []) if rank_cache is not None else rank_candidates(prob, vote, y, hist, current_cols, metric)
        current_set = set(current_cols)
        ranked = [col for col in ranked_all if col in current_set]
        selected = ranked[:k]
        weights = candidate_weights(prob, vote, y, hist, selected, metric)
        scores[idx] = weighted_mean(base.iloc[idx][selected].to_numpy(float), weights)
    return scores


def online_weighted_majority(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, beta: float, source: str) -> np.ndarray:
    base = prob if source == "soft" else vote
    columns = list(base.columns)
    weights = pd.Series(1.0, index=columns, dtype=float)
    scores = np.full(len(y), np.nan, dtype=float)
    for idx in range(len(y)):
        current_cols = nonmissing_columns(base.iloc[idx])
        if current_cols:
            values = base.iloc[idx][current_cols].to_numpy(float)
            scores[idx] = weighted_mean(values, weights.loc[current_cols].to_numpy(float))
        truth = int(y[idx])
        available = nonmissing_columns(vote.iloc[idx])
        wrong = vote.iloc[idx][available].astype(int).ne(truth)
        weights.loc[list(wrong.index[wrong])] *= beta
        total = float(weights.sum())
        if total > 0:
            weights /= total / len(weights)
    return scores


def dynamic_local_selection(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, topn: int, local_k: int, topm: int, source: str, min_history: int) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    base = prob if source == "soft" else vote
    for idx in range(len(y)):
        hist = np.arange(idx)
        current_cols = nonmissing_columns(base.iloc[idx])
        if len(hist) < min_history or not current_cols:
            scores[idx] = float(base.iloc[idx][current_cols].mean()) if current_cols else np.nan
            continue
        ranked = rank_candidates(prob, vote, y, hist, current_cols, "ba")[:topn]
        complete_hist = [row for row in hist if prob.iloc[row][ranked].notna().all()]
        if len(complete_hist) < max(3, local_k):
            selected = ranked[:topm]
        else:
            hist_matrix = prob.iloc[complete_hist][ranked].to_numpy(float)
            current = prob.iloc[idx][ranked].to_numpy(float)
            dist = np.sqrt(np.mean((hist_matrix - current) ** 2, axis=1))
            neighbor_rows = np.asarray(complete_hist)[np.argsort(dist)[:local_k]]
            local_scores = {}
            for col in ranked:
                local_scores[col] = balanced_acc_safe(y[neighbor_rows], vote.iloc[neighbor_rows][col].to_numpy(float))
            selected = sorted(ranked, key=lambda col: local_scores[col], reverse=True)[:topm]
        scores[idx] = float(base.iloc[idx][selected].mean()) if selected else np.nan
    return scores


def rolling_logistic_stack(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, topn: int, window: int, c_value: float, min_history: int) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    for idx in range(len(y)):
        hist = history_indices(idx, window)
        current_cols = nonmissing_columns(prob.iloc[idx])
        if len(hist) < min_history or not current_cols:
            scores[idx] = float(prob.iloc[idx][current_cols].mean()) if current_cols else np.nan
            continue
        selected = rank_candidates(prob, vote, y, hist, current_cols, "ba")[:topn]
        train_rows = [row for row in hist if prob.iloc[row][selected].notna().all()]
        if len(train_rows) < min_history or len(np.unique(y[train_rows])) < 2:
            scores[idx] = float(prob.iloc[idx][selected].mean()) if selected else np.nan
            continue
        x_train = feature_matrix(prob, vote, train_rows, selected)
        x_test = feature_matrix(prob, vote, [idx], selected)
        model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=1000, solver="liblinear")
        model.fit(x_train, y[train_rows])
        classes = list(model.classes_)
        proba = model.predict_proba(x_test)
        scores[idx] = float(proba[0, classes.index(1)] if 1 in classes else proba[0, -1])
    return scores


def rolling_exhaustive_subset(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, topn: int, k: int, window: int, min_history: int) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    for idx in range(len(y)):
        hist = history_indices(idx, window)
        current_cols = nonmissing_columns(vote.iloc[idx])
        if len(hist) < min_history or not current_cols:
            scores[idx] = float(vote.iloc[idx][current_cols].mean()) if current_cols else np.nan
            continue
        ranked = rank_candidates(prob, vote, y, hist, current_cols, "ba")[:topn]
        best_combo: tuple[str, ...] | None = None
        best_score = -1.0
        for combo in itertools.combinations(ranked, k):
            hist_scores = vote.iloc[hist][list(combo)].mean(axis=1).to_numpy(float)
            score = balanced_acc_safe(y[hist], (hist_scores > 0.5).astype(int))
            if score > best_score:
                best_score = score
                best_combo = combo
        scores[idx] = float(vote.iloc[idx][list(best_combo)].mean()) if best_combo else np.nan
    return scores


def rolling_forward_selection(prob: pd.DataFrame, y: np.ndarray, *, topn: int, size: int, window: int, min_history: int) -> np.ndarray:
    scores = np.full(len(y), np.nan, dtype=float)
    for idx in range(len(y)):
        hist = history_indices(idx, window)
        current_cols = nonmissing_columns(prob.iloc[idx])
        if len(hist) < min_history or not current_cols:
            scores[idx] = float(prob.iloc[idx][current_cols].mean()) if current_cols else np.nan
            continue
        ranked = rank_candidates(prob, (prob > 0.5).astype(float), y, hist, current_cols, "ba")[:topn]
        selected: list[str] = []
        remaining = list(ranked)
        for _ in range(min(size, len(remaining))):
            best_col = None
            best_score = -1.0
            for col in remaining:
                trial = selected + [col]
                hist_scores = prob.iloc[hist][trial].mean(axis=1).to_numpy(float)
                score = balanced_acc_safe(y[hist], (hist_scores > 0.5).astype(int))
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_col is None:
                break
            selected.append(best_col)
            remaining.remove(best_col)
        scores[idx] = float(prob.iloc[idx][selected].mean()) if selected else np.nan
    return scores


def oracle_topk_score(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, *, metric: str, k: int, source: str) -> np.ndarray:
    current_cols = list(prob.columns)
    ranked = rank_candidates(prob, vote, y, np.arange(len(y)), current_cols, metric)
    selected = ranked[:k]
    base = prob if source == "soft" else vote
    return base[selected].mean(axis=1).to_numpy(float)


def rank_candidates(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, rows: np.ndarray, candidates: list[str], metric: str) -> list[str]:
    scored = []
    for col in candidates:
        mask = prob.iloc[rows][col].notna().to_numpy(bool) & vote.iloc[rows][col].notna().to_numpy(bool)
        if mask.sum() < 2:
            continue
        row_idx = rows[mask]
        y_part = y[row_idx]
        p = prob.iloc[row_idx][col].to_numpy(float)
        v = vote.iloc[row_idx][col].to_numpy(float)
        if metric == "ba":
            score = balanced_acc_safe(y_part, v)
        elif metric == "auc":
            score = auc_safe(y_part, p)
        elif metric == "ap":
            score = ap_safe(y_part, p)
        elif metric == "brier":
            score = -float(np.mean((p - y_part) ** 2))
        else:
            raise ValueError(f"Unknown candidate metric: {metric}")
        scored.append((col, score))
    return [col for col, _ in sorted(scored, key=lambda item: item[1], reverse=True)]


def candidate_weights(prob: pd.DataFrame, vote: pd.DataFrame, y: np.ndarray, rows: np.ndarray, candidates: list[str], metric: str) -> np.ndarray:
    weights = []
    for col in candidates:
        mask = prob.iloc[rows][col].notna().to_numpy(bool) & vote.iloc[rows][col].notna().to_numpy(bool)
        if mask.sum() < 2:
            weights.append(1e-6)
            continue
        row_idx = rows[mask]
        if metric == "brier":
            p = prob.iloc[row_idx][col].to_numpy(float)
            score = 1.0 / (float(np.mean((p - y[row_idx]) ** 2)) + 1e-6)
        else:
            score = max(0.0, balanced_acc_safe(y[row_idx], vote.iloc[row_idx][col].to_numpy(float)) - 0.5)
        weights.append(score + 1e-6)
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum() if weights.sum() > 0 else np.ones(len(candidates)) / max(1, len(candidates))


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if finite.sum() < 2 or len(np.unique(y_true[finite])) < 2:
        return 0.5
    candidates = np.unique(np.quantile(score[finite], np.linspace(0.05, 0.95, 31)))
    best = 0.5
    best_score = -1.0
    for threshold in candidates:
        ba = balanced_acc_safe(y_true[finite], (score[finite] > threshold).astype(int))
        if ba > best_score:
            best = float(threshold)
            best_score = ba
    return best


def compute_search_metrics(y_true: np.ndarray, y_prob: np.ndarray, actual_returns: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    y_pred = (y_prob > 0.5).astype(int)
    strategy_returns = np.where(y_pred == 1, actual_returns, -actual_returns)
    std = float(np.std(strategy_returns))
    sharpe = 0.0 if std < 1e-12 else float(np.mean(strategy_returns) / std * math.sqrt(12))
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    gross_loss = abs(float(losses.sum()))
    profit_factor = float("inf") if gross_loss < 1e-12 and wins.sum() > 0 else (float(wins.sum()) / gross_loss if gross_loss >= 1e-12 else 1.0)
    return {
        "DirAcc": round(float(np.mean(y_pred == y_true)), 4),
        "BalancedAcc": round(balanced_acc_safe(y_true, y_pred), 4),
        "AUC": round(auc_safe(y_true, y_prob), 4),
        "AP": round(ap_safe(y_true, y_prob), 4),
        "Sharpe": round(sharpe, 4),
        "ProfitFactor": round(profit_factor, 4) if np.isfinite(profit_factor) else profit_factor,
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "Brier": round(float(np.mean((y_prob - y_true.astype(float)) ** 2)), 4),
        "LogLoss": round(logloss_safe(y_true, y_prob), 4),
    }


def history_indices(idx: int, window: int) -> np.ndarray:
    if idx <= 0:
        return np.asarray([], dtype=int)
    start = 0 if window >= 9999 else max(0, idx - int(window))
    return np.arange(start, idx, dtype=int)


def nonmissing_columns(row: pd.Series) -> list[str]:
    return [column for column, value in row.items() if pd.notna(value)]


def feature_matrix(prob: pd.DataFrame, vote: pd.DataFrame, rows: list[int] | np.ndarray, cols: list[str]) -> np.ndarray:
    p = prob.iloc[rows][cols].to_numpy(float)
    v = vote.iloc[rows][cols].to_numpy(float)
    return np.concatenate([p, v], axis=1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights)
    if finite.sum() == 0:
        return float("nan")
    weights = weights[finite]
    values = values[finite]
    if weights.sum() <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def balanced_acc_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_pred)
    if finite.sum() == 0:
        return 0.5
    y_part = y_true[finite]
    pred_part = y_pred[finite].astype(int)
    if len(np.unique(y_part)) < 2:
        return 0.5
    tp = float(((y_part == 1) & (pred_part == 1)).sum())
    tn = float(((y_part == 0) & (pred_part == 0)).sum())
    pos = float((y_part == 1).sum())
    neg = float((y_part == 0).sum())
    tpr = tp / pos if pos > 0 else 0.0
    tnr = tn / neg if neg > 0 else 0.0
    return 0.5 * (tpr + tnr)


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def ap_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return 0.5


def logloss_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, np.clip(y_score, 1e-15, 1.0 - 1e-15), labels=[0, 1]))
    except ValueError:
        return 0.0


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def write_markdown_report(output_dir: Path, leaderboard: pd.DataFrame) -> None:
    lines = [
        "# Ensemble Search Report",
        "",
        "Valid rows are walk-forward/prequential: every target month uses only earlier target months for model selection, weighting, stacking, dynamic selection, and thresholding.",
        "",
        "Diagnostic oracle rows are written in per-horizon CSV files and are intentionally excluded from this leaderboard.",
        "",
    ]
    if leaderboard.empty:
        lines.append("No valid methods produced predictions.")
    else:
        display_cols = ["horizon", "method", "method_family", "n_predictions", "coverage", "BalancedAcc", "AUC", "AP", "DirAcc", "Sharpe"]
        for horizon, group in leaderboard.groupby("horizon"):
            lines.extend([f"## Horizon {horizon}", ""])
            lines.append(group[display_cols].head(20).to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
    (output_dir / "ENSEMBLE_SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
