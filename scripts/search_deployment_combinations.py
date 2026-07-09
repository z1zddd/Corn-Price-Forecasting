#!/usr/bin/env python3
"""Search deployment candidate ensembles over completed rolling predictions.

This script is intentionally different from ``search_prediction_ensembles.py``.
It answers the deployment-selection question: given all completed historical
rolling predictions, which fixed combinations would we choose now for future
use? The resulting scores are full-history discovery scores, so they should be
reported separately from strict walk-forward model-selection scores.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_forecast.pipeline.eval.metrics import compute_all_metrics  # noqa: E402
from scripts import search_prediction_ensembles as base_search  # noqa: E402


@dataclass(frozen=True)
class Pool:
    name: str
    rank_metric: str
    scope: str
    candidates: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="all_rolling_predictions.csv or .csv.gz")
    parser.add_argument("--output-dir", default="experiments/deployment_combination_search")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--max-pool-size", type=int, default=20)
    parser.add_argument("--pool-sizes", default="6,8,10,12,15,20")
    parser.add_argument("--rank-metrics", default="ba,brier,auc,ap")
    parser.add_argument(
        "--scopes",
        default="all,news,nonews,cls,reg,lb6,lb9,lb12,best_per_model,best_per_family",
        help=(
            "Comma-separated candidate pool scopes to search. Common scopes: all, news, nonews, "
            "cls, reg, lb6, lb9, lb12, best_per_model, best_per_family, or family_<name>."
        ),
    )
    parser.add_argument(
        "--aggregators",
        default="all",
        help=(
            "Comma-separated deployment aggregators. Use all, hard_vote_strict, hard_vote_tie_up, "
            "soft_mean_fixed, soft_mean_best_threshold, hard_weighted, soft_weighted, "
            "or soft_weighted_best_threshold. Weighted names expand with each pool rank metric."
        ),
    )
    parser.add_argument("--keep-top", type=int, default=500)
    parser.add_argument("--engine", choices=["python", "vectorized"], default="vectorized")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--search-modes",
        default="exhaustive",
        help=(
            "Comma-separated search modes: exhaustive, forward, forward_replacement, or all. "
            "Forward modes run metric-optimized ensemble selection over broad candidate pools."
        ),
    )
    parser.add_argument("--forward-max-k", type=int, default=80)
    parser.add_argument(
        "--forward-candidate-limit",
        type=int,
        default=0,
        help="Top-N candidates per forward pool after ranking; 0 means use the full scoped pool.",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=31,
        help="Number of quantile thresholds to test for *_best_threshold aggregators.",
    )
    parser.add_argument(
        "--forward-tie-breakers",
        default="balanced",
        help="Comma-separated forward tie-breakers: balanced, ba_only, or all.",
    )
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = base_search.read_predictions(args.predictions)
    horizons = [int(value.strip()) for value in str(args.horizons).split(",") if value.strip()]
    pool_sizes = sorted({int(value.strip()) for value in str(args.pool_sizes).split(",") if value.strip()})
    rank_metrics = [value.strip() for value in str(args.rank_metrics).split(",") if value.strip()]
    scopes = {value.strip() for value in str(args.scopes).split(",") if value.strip()}
    args.enabled_aggregators = parse_aggregators(args.aggregators)
    args.enabled_search_modes = parse_search_modes(args.search_modes)

    all_rows: list[dict] = []
    all_pool_rows: list[dict] = []
    best_prediction_payloads: dict[int, pd.DataFrame] = {}
    best_candidate_payloads: dict[int, dict] = {}

    for horizon in horizons:
        matrix = base_search.build_matrix(predictions, horizon, args.min_candidate_coverage)
        candidate_scores = score_candidates(matrix)
        write_candidate_ranking(output_dir, horizon, matrix, candidate_scores)
        pools = build_pools(matrix["inventory"], candidate_scores, rank_metrics, pool_sizes, args.max_pool_size, scopes)
        forward_pools = build_forward_pools(
            matrix["inventory"],
            candidate_scores,
            rank_metrics,
            int(args.forward_candidate_limit),
            scopes,
        )
        all_pool_rows.extend(pool_row(horizon, pool) for pool in pools)
        if args.enabled_search_modes.intersection({"forward", "forward_replacement"}):
            all_pool_rows.extend(pool_row(horizon, pool) for pool in forward_pools)
        print(
            f"[h{horizon}] searching exhaustive_pools={len(pools)} forward_pools={len(forward_pools)} "
            f"modes={','.join(sorted(args.enabled_search_modes))} engine={args.engine}",
            flush=True,
        )
        rows, predictions_by_method = search_horizon(matrix, pools, forward_pools, candidate_scores, args)
        h_rows = pd.DataFrame(rows).sort_values(
            ["BalancedAcc", "AUC", "AP", "DirAcc", "n_predictions"],
            ascending=[False, False, False, False, False],
        )
        h_rows.to_csv(output_dir / f"h{horizon}_deployment_combination_leaderboard.csv", index=False)
        all_rows.extend(h_rows.to_dict("records"))
        best_row = h_rows.iloc[0].to_dict()
        best_name = str(best_row["method"])
        best_prediction_payloads[horizon] = predictions_by_method[best_name]
        best_candidate_payloads[horizon] = {
            "horizon": horizon,
            "method": best_name,
            "search_protocol": best_row["search_protocol"],
            "selection_mode": best_row.get("selection_mode", "exhaustive"),
            "pool": best_row["pool"],
            "aggregator": best_row["aggregator"],
            "k": int(best_row["k"]),
            "threshold": float(best_row["threshold"]),
            "BalancedAcc": float(best_row["BalancedAcc"]),
            "AUC": float(best_row["AUC"]),
            "AP": float(best_row["AP"]),
            "DirAcc": float(best_row["DirAcc"]),
            "selected_candidates": json.loads(best_row["selected_candidates"]),
            "candidate_weights": json.loads(best_row.get("candidate_weights", "{}")),
        }
        best_prediction_payloads[horizon].to_csv(output_dir / f"h{horizon}_best_deployment_predictions.csv", index=False)
        (output_dir / f"h{horizon}_best_deployment_candidates.json").write_text(
            json.dumps(best_candidate_payloads[horizon], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    combined = pd.DataFrame(all_rows).sort_values(
        ["horizon", "BalancedAcc", "AUC", "AP"],
        ascending=[True, False, False, False],
    )
    combined.to_csv(output_dir / "deployment_combination_leaderboard.csv", index=False)
    pd.DataFrame(all_pool_rows).to_csv(output_dir / "deployment_candidate_pools.csv", index=False)
    write_report(output_dir, combined, best_candidate_payloads, args)
    print(combined.groupby("horizon").head(20).to_string(index=False))
    print(f"results written to {output_dir}")


def score_candidates(matrix: dict) -> dict[str, dict[str, float]]:
    prob = matrix["prob"]
    vote = matrix["vote"]
    y = matrix["base"]["actual_direction"].to_numpy(int)
    returns = matrix["base"]["actual_return"].to_numpy(float)
    scores: dict[str, dict[str, float]] = {}
    for candidate in matrix["candidates"]:
        p = prob[candidate].to_numpy(float)
        v = vote[candidate].to_numpy(float)
        valid = np.isfinite(p) & np.isfinite(v)
        if valid.sum() < 2:
            continue
        pred = v[valid].astype(int)
        strat = np.where(pred == 1, returns[valid], -returns[valid])
        scores[candidate] = {
            "ba": base_search.balanced_acc_safe(y[valid], pred),
            "auc": base_search.auc_safe(y[valid], p[valid]),
            "ap": base_search.ap_safe(y[valid], p[valid]),
            "brier": -float(np.mean((p[valid] - y[valid]) ** 2)),
            "diracc": float(np.mean(pred == y[valid])),
            "sharpe": float(np.mean(strat) / (np.std(strat) + 1e-12) * math.sqrt(12)),
            "coverage": float(valid.mean()),
        }
    return scores


def write_candidate_ranking(output_dir: Path, horizon: int, matrix: dict, candidate_scores: dict[str, dict[str, float]]) -> None:
    info = matrix["inventory"].set_index("candidate_id")
    rows = []
    for candidate, scores in candidate_scores.items():
        meta = info.loc[candidate].to_dict()
        rows.append({"candidate_id": candidate, **meta, **scores})
    pd.DataFrame(rows).sort_values(["ba", "auc", "ap"], ascending=[False, False, False]).to_csv(
        output_dir / f"h{horizon}_candidate_ranking.csv",
        index=False,
    )


def build_pools(
    inventory: pd.DataFrame,
    candidate_scores: dict[str, dict[str, float]],
    rank_metrics: list[str],
    pool_sizes: list[int],
    max_pool_size: int,
    requested_scopes: set[str],
) -> list[Pool]:
    info = inventory.set_index("candidate_id")
    pools: list[Pool] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for metric in rank_metrics:
        ranked_all = ranked_candidates(candidate_scores, metric)
        for scope_name, scoped in scope_candidates(ranked_all, info, metric, candidate_scores).items():
            if scope_name not in requested_scopes:
                continue
            for size in pool_sizes:
                limit = min(size, max_pool_size, len(scoped))
                if limit < 1:
                    continue
                selected = scoped[:limit]
                key = (f"{scope_name}_top{limit}_by_{metric}", tuple(selected))
                if key in seen:
                    continue
                seen.add(key)
                pools.append(Pool(name=key[0], rank_metric=metric, scope=scope_name, candidates=selected))
    return pools


def build_forward_pools(
    inventory: pd.DataFrame,
    candidate_scores: dict[str, dict[str, float]],
    rank_metrics: list[str],
    candidate_limit: int,
    requested_scopes: set[str],
) -> list[Pool]:
    info = inventory.set_index("candidate_id")
    pools: list[Pool] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for metric in rank_metrics:
        ranked_all = ranked_candidates(candidate_scores, metric)
        for scope_name, scoped in scope_candidates(ranked_all, info, metric, candidate_scores).items():
            if scope_name not in requested_scopes:
                continue
            if candidate_limit > 0:
                selected = scoped[: min(candidate_limit, len(scoped))]
                limit_label = f"top{len(selected)}"
            else:
                selected = list(scoped)
                limit_label = f"full{len(selected)}"
            if not selected:
                continue
            key = (f"{scope_name}_{limit_label}_by_{metric}", tuple(selected))
            if key in seen:
                continue
            seen.add(key)
            pools.append(Pool(name=key[0], rank_metric=metric, scope=scope_name, candidates=selected))
    return pools


def scope_candidates(
    ranked_all: list[str],
    info: pd.DataFrame,
    metric: str,
    candidate_scores: dict[str, dict[str, float]],
) -> dict[str, list[str]]:
    scopes: dict[str, list[str]] = {"all": ranked_all}
    for feature in ["with_news_precomputed_pca", "no_news"]:
        label = "news" if feature == "with_news_precomputed_pca" else "nonews"
        scopes[label] = [candidate for candidate in ranked_all if str(info.loc[candidate, "feature_set"]) == feature]
    for head in ["cls", "reg"]:
        scopes[head] = [candidate for candidate in ranked_all if str(info.loc[candidate, "head"]) == head]
    for lookback in sorted(info["lookback_months"].dropna().astype(int).unique()):
        scopes[f"lb{lookback}"] = [candidate for candidate in ranked_all if int(info.loc[candidate, "lookback_months"]) == lookback]
    for family in sorted(info["family"].dropna().astype(str).unique()):
        scopes[f"family_{safe_name(family)}"] = [candidate for candidate in ranked_all if str(info.loc[candidate, "family"]) == family]
    scopes["best_per_model"] = best_per_group(ranked_all, info, "model", metric, candidate_scores)
    scopes["best_per_family"] = best_per_group(ranked_all, info, "family", metric, candidate_scores)
    return scopes


def best_per_group(
    ranked_all: list[str],
    info: pd.DataFrame,
    group_col: str,
    metric: str,
    candidate_scores: dict[str, dict[str, float]],
) -> list[str]:
    best: dict[str, str] = {}
    for candidate in ranked_all:
        group = str(info.loc[candidate, group_col])
        current = best.get(group)
        if current is None or candidate_scores[candidate][metric] > candidate_scores[current][metric]:
            best[group] = candidate
    return ranked_candidates({candidate: candidate_scores[candidate] for candidate in best.values()}, metric)


def ranked_candidates(candidate_scores: dict[str, dict[str, float]], metric: str) -> list[str]:
    return [
        candidate
        for candidate, _ in sorted(
            candidate_scores.items(),
            key=lambda item: (item[1].get(metric, -1e9), item[1].get("ba", -1e9), item[1].get("auc", -1e9)),
            reverse=True,
        )
    ]


def search_horizon(
    matrix: dict,
    pools: list[Pool],
    forward_pools: list[Pool],
    candidate_scores: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    selected: dict[str, pd.DataFrame] = {}
    modes = getattr(args, "enabled_search_modes", {"exhaustive"})
    if "exhaustive" in modes:
        if getattr(args, "engine", "vectorized") == "vectorized":
            mode_rows, mode_selected = search_horizon_vectorized(matrix, pools, candidate_scores, args)
        else:
            mode_rows, mode_selected = search_horizon_python(matrix, pools, candidate_scores, args)
        rows.extend(mode_rows)
        selected.update(mode_selected)
    if "forward" in modes:
        mode_rows, mode_selected = search_forward_horizon(
            matrix,
            forward_pools,
            candidate_scores,
            args,
            allow_replacement=False,
        )
        rows.extend(mode_rows)
        selected.update(mode_selected)
    if "forward_replacement" in modes:
        mode_rows, mode_selected = search_forward_horizon(
            matrix,
            forward_pools,
            candidate_scores,
            args,
            allow_replacement=True,
        )
        rows.extend(mode_rows)
        selected.update(mode_selected)
    return rows, selected


def search_horizon_python(
    matrix: dict,
    pools: list[Pool],
    candidate_scores: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows_by_key: dict[tuple[str, str, int], tuple[tuple[float, float, float, float], Pool, str, list[str], np.ndarray, float]] = {}
    top_heap: list[tuple[tuple[float, float, float, float], int, Pool, str, list[str], np.ndarray, float]] = []
    counter = 0
    for pool in pools:
        candidates = pool.candidates
        if not candidates:
            continue
        prob_arr = matrix["prob"][candidates].to_numpy(float)
        vote_arr = matrix["vote"][candidates].to_numpy(float)
        y = matrix["base"]["actual_direction"].to_numpy(int)
        returns = matrix["base"]["actual_return"].to_numpy(float)
        aggregators = selected_aggregators(pool.rank_metric, args)
        for combo_size in range(1, len(candidates) + 1):
            for combo in itertools.combinations(range(len(candidates)), combo_size):
                combo_candidates = [candidates[idx] for idx in combo]
                for aggregator in aggregators:
                    raw_score = aggregate_score(prob_arr, vote_arr, combo, combo_candidates, aggregator, candidate_scores)
                    score, threshold = apply_threshold(raw_score, y, aggregator)
                    sort_key = quick_sort_tuple(y, score)
                    key = (pool.name, aggregator, combo_size)
                    if key not in rows_by_key or sort_key > rows_by_key[key][0]:
                        rows_by_key[key] = (sort_key, pool, aggregator, combo_candidates, score.copy(), threshold)
                    if len(top_heap) < args.keep_top:
                        heapq.heappush(top_heap, (sort_key, counter, pool, aggregator, combo_candidates, score.copy(), threshold))
                    elif sort_key > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (sort_key, counter, pool, aggregator, combo_candidates, score.copy(), threshold))
                    counter += 1
    selected: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    payloads: list[tuple[Pool, str, list[str], np.ndarray, float]] = []
    for _, pool, aggregator, combo_candidates, score, threshold in rows_by_key.values():
        payloads.append((pool, aggregator, combo_candidates, score, threshold))
    for _, _, pool, aggregator, combo_candidates, score, threshold in top_heap:
        payloads.append((pool, aggregator, combo_candidates, score, threshold))
    seen_methods: set[str] = set()
    for pool, aggregator, combo_candidates, score, threshold in payloads:
        method = method_name(pool, aggregator, combo_candidates)
        if method in seen_methods:
            continue
        seen_methods.add(method)
        row, pred_df = result_payload(matrix, score, threshold, combo_candidates, pool, aggregator, args, returns)
        rows.append(row)
        selected[row["method"]] = pred_df
    return rows, selected


def search_horizon_vectorized(
    matrix: dict,
    pools: list[Pool],
    candidate_scores: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows_by_key: dict[tuple[str, str, int], tuple[tuple[float, float, float, float], Pool, str, list[str], np.ndarray, float]] = {}
    top_heap: list[tuple[tuple[float, float, float, float], int, Pool, str, list[str], np.ndarray, float]] = []
    counter = 0
    y = matrix["base"]["actual_direction"].to_numpy(int)
    returns = matrix["base"]["actual_return"].to_numpy(float)
    for pool in pools:
        candidates = pool.candidates
        if not candidates:
            continue
        prob_arr = matrix["prob"][candidates].to_numpy(float)
        vote_arr = matrix["vote"][candidates].to_numpy(float)
        aggregators = selected_aggregators(pool.rank_metric, args)
        print(f"[h{matrix['horizon']}] pool={pool.name} candidates={len(candidates)} aggregators={','.join(aggregators)}", flush=True)
        for combo_size in range(1, len(candidates) + 1):
            for masks in mask_batches(len(candidates), combo_size, int(args.batch_size)):
                bit_matrix = masks_to_matrix(masks, len(candidates))
                for aggregator in aggregators:
                    raw_scores = aggregate_score_batch(prob_arr, vote_arr, bit_matrix, candidates, aggregator, candidate_scores)
                    scores, thresholds = apply_threshold_batch(raw_scores, y, aggregator, int(args.threshold_grid_size))
                    sort_keys = quick_sort_arrays(y, scores)
                    best_idx = int(np.argmax(sort_keys[:, 0] * 1e9 + sort_keys[:, 1] * 1e6 + sort_keys[:, 2] * 1e3 + sort_keys[:, 3]))
                    key = (pool.name, aggregator, combo_size)
                    best_tuple = tuple(float(v) for v in sort_keys[best_idx])
                    if key not in rows_by_key or best_tuple > rows_by_key[key][0]:
                        selected_candidates = selected_from_mask(int(masks[best_idx]), candidates)
                        rows_by_key[key] = (
                            best_tuple,
                            pool,
                            aggregator,
                            selected_candidates,
                            scores[best_idx].copy(),
                            float(thresholds[best_idx]),
                        )
                    top_count = min(int(args.keep_top), len(masks))
                    scalar = sort_keys[:, 0] * 1e9 + sort_keys[:, 1] * 1e6 + sort_keys[:, 2] * 1e3 + sort_keys[:, 3]
                    if top_count < len(masks):
                        top_indices = np.argpartition(scalar, -top_count)[-top_count:]
                    else:
                        top_indices = np.arange(len(masks))
                    for idx in top_indices:
                        idx = int(idx)
                        sort_tuple_value = tuple(float(v) for v in sort_keys[idx])
                        selected_candidates = selected_from_mask(int(masks[idx]), candidates)
                        payload = (
                            sort_tuple_value,
                            counter,
                            pool,
                            aggregator,
                            selected_candidates,
                            scores[idx].copy(),
                            float(thresholds[idx]),
                        )
                        if len(top_heap) < args.keep_top:
                            heapq.heappush(top_heap, payload)
                        elif sort_tuple_value > top_heap[0][0]:
                            heapq.heapreplace(top_heap, payload)
                        counter += 1
    selected: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    payloads: list[tuple[Pool, str, list[str], np.ndarray, float]] = []
    for _, pool, aggregator, combo_candidates, score, threshold in rows_by_key.values():
        payloads.append((pool, aggregator, combo_candidates, score, threshold))
    for _, _, pool, aggregator, combo_candidates, score, threshold in top_heap:
        payloads.append((pool, aggregator, combo_candidates, score, threshold))
    seen_methods: set[str] = set()
    for pool, aggregator, combo_candidates, score, threshold in payloads:
        method = method_name(pool, aggregator, combo_candidates)
        if method in seen_methods:
            continue
        seen_methods.add(method)
        row, pred_df = result_payload(matrix, score, threshold, combo_candidates, pool, aggregator, args, returns)
        rows.append(row)
        selected[row["method"]] = pred_df
    return rows, selected


def search_forward_horizon(
    matrix: dict,
    pools: list[Pool],
    candidate_scores: dict[str, dict[str, float]],
    args: argparse.Namespace,
    allow_replacement: bool,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    rows: list[dict] = []
    selected: dict[str, pd.DataFrame] = {}
    y = matrix["base"]["actual_direction"].to_numpy(int)
    returns = matrix["base"]["actual_return"].to_numpy(float)
    mode_label = "forward_replacement" if allow_replacement else "forward"
    max_k = int(args.forward_max_k)
    for pool in pools:
        candidates = pool.candidates
        if not candidates:
            continue
        prob_arr = matrix["prob"][candidates].to_numpy(float)
        vote_arr = matrix["vote"][candidates].to_numpy(float)
        aggregators = selected_aggregators(pool.rank_metric, args)
        print(
            f"[h{matrix['horizon']}] {mode_label} pool={pool.name} "
            f"candidates={len(candidates)} aggregators={','.join(aggregators)} max_k={max_k}",
            flush=True,
        )
        for aggregator in aggregators:
            values = vote_arr if aggregator.startswith("hard") else prob_arr
            filled = np.nan_to_num(values, nan=0.0)
            valid_values = np.isfinite(values).astype(float)
            weights = aggregator_candidate_weights(candidates, aggregator, candidate_scores)
            for tie_breaker in parse_forward_tie_breakers(args.forward_tie_breakers):
                mode_pool = Pool(
                    name=f"{mode_label}_{tie_breaker}_{pool.name}",
                    rank_metric=pool.rank_metric,
                    scope=pool.scope,
                    candidates=pool.candidates,
                )
                numerator = np.zeros(values.shape[0], dtype=float)
                denominator = np.zeros(values.shape[0], dtype=float)
                remaining = np.arange(len(candidates), dtype=int)
                selected_candidates: list[str] = []
                steps = max_k if allow_replacement else min(max_k, len(candidates))
                for _step in range(steps):
                    trial_indices = np.arange(len(candidates), dtype=int) if allow_replacement else remaining
                    if len(trial_indices) == 0:
                        break
                    trial_weights = weights[trial_indices]
                    trial_num = numerator.reshape(1, -1) + filled[:, trial_indices].T * trial_weights.reshape(-1, 1)
                    trial_den = denominator.reshape(1, -1) + valid_values[:, trial_indices].T * trial_weights.reshape(-1, 1)
                    raw_scores = np.full(trial_num.shape, np.nan, dtype=float)
                    np.divide(trial_num, trial_den, out=raw_scores, where=trial_den > 0)
                    scores, thresholds = apply_threshold_batch(raw_scores, y, aggregator, int(args.threshold_grid_size))
                    sort_keys = quick_sort_arrays(y, scores)
                    scalar = forward_sort_scalar(sort_keys, tie_breaker)
                    best_local = int(np.argmax(scalar))
                    best_candidate_idx = int(trial_indices[best_local])
                    best_weight = float(weights[best_candidate_idx])
                    numerator += filled[:, best_candidate_idx] * best_weight
                    denominator += valid_values[:, best_candidate_idx] * best_weight
                    selected_candidates.append(candidates[best_candidate_idx])
                    if not allow_replacement:
                        remaining = remaining[remaining != best_candidate_idx]
                    row, pred_df = result_payload(
                        matrix,
                        scores[best_local].copy(),
                        float(thresholds[best_local]),
                        list(selected_candidates),
                        mode_pool,
                        aggregator,
                        args,
                        returns,
                        selection_mode=mode_label,
                    )
                    row["forward_tie_breaker"] = tie_breaker
                    rows.append(row)
                    selected[row["method"]] = pred_df
    return rows, selected


def deployment_aggregators(rank_metric: str) -> list[str]:
    return expand_aggregators(rank_metric, None)


def parse_aggregators(value: str) -> set[str] | None:
    text = str(value).strip().lower()
    if text == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_search_modes(value: str) -> set[str]:
    text = str(value).strip().lower()
    if text == "all":
        return {"exhaustive", "forward", "forward_replacement"}
    aliases = {
        "greedy": "forward",
        "forward_no_replacement": "forward",
        "forward_with_replacement": "forward_replacement",
        "ensemble_selection": "forward_replacement",
    }
    modes = {aliases.get(item.strip(), item.strip()) for item in text.split(",") if item.strip()}
    valid = {"exhaustive", "forward", "forward_replacement"}
    unknown = modes - valid
    if unknown:
        raise ValueError(f"Unknown search modes: {sorted(unknown)}")
    return modes or {"exhaustive"}


def parse_forward_tie_breakers(value: str) -> list[str]:
    text = str(value).strip().lower()
    if text == "all":
        return ["balanced", "ba_only"]
    breakers = [item.strip() for item in text.split(",") if item.strip()]
    valid = {"balanced", "ba_only"}
    unknown = sorted(set(breakers) - valid)
    if unknown:
        raise ValueError(f"Unknown forward tie-breakers: {unknown}")
    return breakers or ["balanced"]


def expand_aggregators(rank_metric: str, enabled: set[str] | None) -> list[str]:
    specs = {
        "hard_vote_strict": "hard_vote_strict",
        "hard_vote_tie_up": "hard_vote_tie_up",
        "soft_mean_fixed": "soft_mean_fixed",
        "soft_mean_best_threshold": "soft_mean_best_threshold",
        "hard_weighted": f"hard_weighted_{rank_metric}",
        "soft_weighted": f"soft_weighted_{rank_metric}",
        "soft_weighted_best_threshold": f"soft_weighted_{rank_metric}_best_threshold",
    }
    if enabled is None:
        keys = list(specs)
    else:
        keys = [key for key in specs if key in enabled or specs[key] in enabled]
    return [
        specs[key]
        for key in keys
    ]


def selected_aggregators(rank_metric: str, args: argparse.Namespace) -> list[str]:
    return expand_aggregators(rank_metric, getattr(args, "enabled_aggregators", None))


def all_deployment_aggregators(rank_metric: str) -> list[str]:
    return [
        "hard_vote_strict",
        "hard_vote_tie_up",
        "soft_mean_fixed",
        "soft_mean_best_threshold",
        f"hard_weighted_{rank_metric}",
        f"soft_weighted_{rank_metric}",
        f"soft_weighted_{rank_metric}_best_threshold",
    ]


def aggregate_score(
    prob_arr: np.ndarray,
    vote_arr: np.ndarray,
    combo: tuple[int, ...],
    combo_candidates: list[str],
    aggregator: str,
    candidate_scores: dict[str, dict[str, float]],
) -> np.ndarray:
    if aggregator.startswith("hard"):
        base = vote_arr[:, combo]
    else:
        base = prob_arr[:, combo]
    if "_weighted_" not in aggregator:
        counts = np.isfinite(base).sum(axis=1)
        sums = np.nansum(base, axis=1)
        out = np.full(base.shape[0], np.nan, dtype=float)
        return np.divide(sums, counts, out=out, where=counts > 0)
    metric = aggregator.split("_weighted_", 1)[1].replace("_best_threshold", "")
    weights = metric_weights(combo_candidates, metric, candidate_scores)
    weighted = base * weights.reshape(1, -1)
    denom = np.nansum(np.isfinite(base) * weights.reshape(1, -1), axis=1)
    out = np.full(base.shape[0], np.nan, dtype=float)
    return np.divide(np.nansum(weighted, axis=1), denom, out=out, where=denom > 0)


def aggregator_candidate_weights(
    candidates: list[str],
    aggregator: str,
    candidate_scores: dict[str, dict[str, float]],
) -> np.ndarray:
    if "_weighted_" not in aggregator:
        return np.ones(len(candidates), dtype=float)
    metric = aggregator.split("_weighted_", 1)[1].replace("_best_threshold", "")
    weights = np.asarray([candidate_weight(candidate, metric, candidate_scores) for candidate in candidates], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.ones(len(candidates), dtype=float)
    return weights


def mask_batches(n_candidates: int, combo_size: int, batch_size: int) -> Iterable[np.ndarray]:
    batch: list[int] = []
    for combo in itertools.combinations(range(n_candidates), combo_size):
        mask = 0
        for idx in combo:
            mask |= 1 << idx
        batch.append(mask)
        if len(batch) >= batch_size:
            yield np.asarray(batch, dtype=np.uint64)
            batch = []
    if batch:
        yield np.asarray(batch, dtype=np.uint64)


def masks_to_matrix(masks: np.ndarray, n_candidates: int) -> np.ndarray:
    shifts = np.arange(n_candidates, dtype=np.uint64)
    return (((masks[:, None] >> shifts[None, :]) & 1) > 0).astype(float)


def selected_from_mask(mask: int, candidates: list[str]) -> list[str]:
    return [candidate for idx, candidate in enumerate(candidates) if mask & (1 << idx)]


def aggregate_score_batch(
    prob_arr: np.ndarray,
    vote_arr: np.ndarray,
    bit_matrix: np.ndarray,
    candidates: list[str],
    aggregator: str,
    candidate_scores: dict[str, dict[str, float]],
) -> np.ndarray:
    if aggregator.startswith("hard"):
        values = vote_arr
    else:
        values = prob_arr
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)
    if "_weighted_" not in aggregator:
        numerator = bit_matrix @ filled.T
        denominator = bit_matrix @ valid.T
    else:
        metric = aggregator.split("_weighted_", 1)[1].replace("_best_threshold", "")
        weights = np.asarray([candidate_weight(candidate, metric, candidate_scores) for candidate in candidates], dtype=float)
        numerator = (bit_matrix * weights.reshape(1, -1)) @ filled.T
        denominator = (bit_matrix * weights.reshape(1, -1)) @ valid.T
    out = np.full(numerator.shape, np.nan, dtype=float)
    return np.divide(numerator, denominator, out=out, where=denominator > 0)


def candidate_weight(candidate: str, metric: str, candidate_scores: dict[str, dict[str, float]]) -> float:
    score = candidate_scores[candidate].get(metric, 0.0)
    if metric == "brier":
        return 1.0 / max(1e-6, -score)
    return max(1e-6, score - 0.5)


def metric_weights(combo_candidates: list[str], metric: str, candidate_scores: dict[str, dict[str, float]]) -> np.ndarray:
    values = []
    for candidate in combo_candidates:
        score = candidate_scores[candidate].get(metric, 0.0)
        if metric == "brier":
            mse = max(1e-6, -score)
            values.append(1.0 / mse)
        else:
            values.append(max(1e-6, score - 0.5))
    weights = np.asarray(values, dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return np.ones(len(combo_candidates), dtype=float) / max(1, len(combo_candidates))
    return weights / weights.sum()


def apply_threshold(raw_score: np.ndarray, y: np.ndarray, aggregator: str) -> tuple[np.ndarray, float]:
    raw_score = np.asarray(raw_score, dtype=float)
    if aggregator == "hard_vote_tie_up":
        return np.where(raw_score >= 0.5, 0.500001, 0.499999), 0.5
    if aggregator.endswith("_best_threshold"):
        threshold = base_search.best_threshold(y, raw_score)
        return np.where(raw_score > threshold, 0.500001, 0.499999), float(threshold)
    return raw_score, 0.5


def apply_threshold_batch(
    raw_scores: np.ndarray,
    y: np.ndarray,
    aggregator: str,
    grid_size: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    raw_scores = np.asarray(raw_scores, dtype=float)
    thresholds = np.full(raw_scores.shape[0], 0.5, dtype=float)
    if aggregator == "hard_vote_tie_up":
        scores = np.where(raw_scores >= 0.5, 0.500001, 0.499999)
        scores[~np.isfinite(raw_scores)] = np.nan
        return scores, thresholds
    if aggregator.endswith("_best_threshold"):
        thresholds = best_threshold_batch(y, raw_scores, grid_size)
        scores = np.where(raw_scores > thresholds[:, None], 0.500001, 0.499999)
        scores[~np.isfinite(raw_scores)] = np.nan
        return scores, thresholds
    return raw_scores, thresholds


def best_threshold_batch(y: np.ndarray, raw_scores: np.ndarray, grid_size: int = 31) -> np.ndarray:
    valid = np.isfinite(raw_scores)
    thresholds = np.full(raw_scores.shape[0], 0.5, dtype=float)
    enough = valid.sum(axis=1) >= 2
    if not np.any(enough):
        return thresholds
    qs = np.linspace(0.01, 0.99, max(3, int(grid_size)))
    quantiles = np.nanquantile(np.where(valid, raw_scores, np.nan), qs, axis=1).T
    best_ba = np.full(raw_scores.shape[0], -1.0, dtype=float)
    y_row = y.reshape(1, -1)
    for col in range(quantiles.shape[1]):
        threshold = quantiles[:, col]
        pred = raw_scores > threshold[:, None]
        ba = vector_balanced_accuracy(y_row, pred, valid)
        improved = ba > best_ba
        thresholds[improved] = threshold[improved]
        best_ba[improved] = ba[improved]
    thresholds[~np.isfinite(thresholds)] = 0.5
    return thresholds


def quick_sort_arrays(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    valid = np.isfinite(scores)
    pred = scores > 0.5
    y_row = y.reshape(1, -1)
    ba = vector_balanced_accuracy(y_row, pred, valid)
    valid_counts = valid.sum(axis=1)
    correct = ((pred == y_row) & valid).sum(axis=1)
    diracc = np.divide(correct, valid_counts, out=np.zeros_like(ba), where=valid_counts > 0)
    coverage = valid_counts.astype(float) / max(1, scores.shape[1])
    pred_pos = np.divide((pred & valid).sum(axis=1), valid_counts, out=np.zeros_like(ba), where=valid_counts > 0)
    actual_pos = np.divide(((y_row == 1) & valid).sum(axis=1), valid_counts, out=np.zeros_like(ba), where=valid_counts > 0)
    positive_rate_penalty = -np.abs(pred_pos - actual_pos)
    return np.column_stack([ba, diracc, coverage, positive_rate_penalty])


def forward_sort_scalar(sort_keys: np.ndarray, tie_breaker: str) -> np.ndarray:
    if tie_breaker == "ba_only":
        return sort_keys[:, 0]
    return sort_keys[:, 0] * 1e9 + sort_keys[:, 1] * 1e6 + sort_keys[:, 2] * 1e3 + sort_keys[:, 3]


def vector_balanced_accuracy(y_row: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pos_mask = (y_row == 1) & valid
    neg_mask = (y_row == 0) & valid
    pos = pos_mask.sum(axis=1).astype(float)
    neg = neg_mask.sum(axis=1).astype(float)
    tp = (pred & pos_mask).sum(axis=1).astype(float)
    tn = ((~pred) & neg_mask).sum(axis=1).astype(float)
    tpr = np.divide(tp, pos, out=np.zeros_like(tp), where=pos > 0)
    tnr = np.divide(tn, neg, out=np.zeros_like(tn), where=neg > 0)
    missing_class = (pos == 0) | (neg == 0)
    ba = 0.5 * (tpr + tnr)
    ba[missing_class] = 0.5
    return ba


def result_payload(
    matrix: dict,
    score: np.ndarray,
    threshold: float,
    selected_candidates: list[str],
    pool: Pool,
    aggregator: str,
    args: argparse.Namespace,
    returns: np.ndarray,
    selection_mode: str = "exhaustive",
) -> tuple[dict, pd.DataFrame]:
    valid = np.isfinite(score)
    y = matrix["base"].loc[valid, "actual_direction"].to_numpy(int)
    score_valid = score[valid]
    returns_valid = returns[valid]
    metrics = compute_all_metrics(
        y,
        score_valid,
        returns_valid,
        n_bootstrap=args.bootstrap,
        ci_level=args.ci_level,
        annualize=12,
    )
    method = method_name(pool, aggregator, selected_candidates)
    pred = (score_valid > 0.5).astype(int)
    pred_df = pd.DataFrame(
        {
            "date": matrix["base"].loc[valid, "target_month"].dt.date.astype(str),
            "anchor_date": matrix["base"].loc[valid, "anchor_month"].dt.date.astype(str),
            "actual_label": y,
            "predicted_label": pred,
            "predicted_probability": score_valid,
            "direction_correct": (pred == y).astype(int),
            "actual_return": returns_valid,
            "strategy_return": np.where(pred == 1, returns_valid, -returns_valid),
            "model": method,
            "window_id": np.arange(valid.sum(), dtype=int),
            "test_date": matrix["base"].loc[valid, "target_month"].dt.date.astype(str),
        }
    )
    pred_df["equity"] = (1.0 + pred_df["strategy_return"]).cumprod()
    candidate_counts = {candidate: selected_candidates.count(candidate) for candidate in dict.fromkeys(selected_candidates)}
    candidate_weights = {
        candidate: count / max(1, len(selected_candidates))
        for candidate, count in candidate_counts.items()
    }
    row = {
        "horizon": int(matrix["horizon"]),
        "method": method,
        "search_protocol": "full_history_deployment_discovery",
        "method_family": "deployment_fixed_combination",
        "selection_mode": selection_mode,
        "pool": pool.name,
        "scope": pool.scope,
        "rank_metric": pool.rank_metric,
        "aggregator": aggregator,
        "k": len(selected_candidates),
        "threshold": round(float(threshold), 6),
        "coverage": float(valid.mean()),
        "n_predictions": int(valid.sum()),
        "selected_candidates": json.dumps(selected_candidates, ensure_ascii=False),
        "candidate_weights": json.dumps(candidate_weights, ensure_ascii=False),
        **metrics,
    }
    return row, pred_df


def sort_tuple(row: dict) -> tuple[float, float, float, float]:
    return (float(row["BalancedAcc"]), float(row["AUC"]), float(row["AP"]), float(row["DirAcc"]))


def quick_sort_tuple(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(score)
    if valid.sum() < 2:
        return (-1.0, -1.0, -1.0, -1.0)
    pred = (score[valid] > 0.5).astype(int)
    y_valid = y[valid]
    ba = base_search.balanced_acc_safe(y_valid, pred)
    diracc = float(np.mean(pred == y_valid))
    coverage = float(valid.mean())
    positive_rate_penalty = -abs(float(pred.mean()) - float(y_valid.mean()))
    return (ba, diracc, coverage, positive_rate_penalty)


def method_name(pool: Pool, aggregator: str, selected_candidates: list[str]) -> str:
    return f"deploy_{pool.name}_{aggregator}_k{len(selected_candidates)}_{short_hash(selected_candidates)}"


def pool_row(horizon: int, pool: Pool) -> dict:
    return {
        "horizon": horizon,
        "pool": pool.name,
        "scope": pool.scope,
        "rank_metric": pool.rank_metric,
        "candidate_count": len(pool.candidates),
        "candidates": json.dumps(pool.candidates, ensure_ascii=False),
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def short_hash(values: Iterable[str]) -> str:
    total = 0
    for value in values:
        for char in value:
            total = (total * 131 + ord(char)) % 1_000_000_007
    return f"{total:08x}"[-8:]


def write_report(output_dir: Path, combined: pd.DataFrame, best_payloads: dict[int, dict], args: argparse.Namespace) -> None:
    lines = [
        "# Deployment Combination Search",
        "",
        "Protocol: `full_history_deployment_discovery`.",
        "",
        "This report is for choosing fixed combinations to deploy from completed historical rolling predictions. "
        "It intentionally uses all available historical prediction outcomes for model selection, so it is a "
        "deployment discovery leaderboard rather than a strict walk-forward model-selection validation.",
        "",
        f"Max pool size: `{args.max_pool_size}`; pool sizes: `{args.pool_sizes}`; rank metrics: `{args.rank_metrics}`; scopes: `{args.scopes}`.",
        "",
        "## Best By Horizon",
        "",
    ]
    for horizon, payload in sorted(best_payloads.items()):
        lines.extend(
            [
                f"### Horizon {horizon}",
                "",
                f"- Method: `{payload['method']}`",
                f"- BA/AUC/AP/DirAcc: `{payload['BalancedAcc']:.4f}` / `{payload['AUC']:.4f}` / `{payload['AP']:.4f}` / `{payload['DirAcc']:.4f}`",
                f"- Selection mode: `{payload.get('selection_mode', 'exhaustive')}`",
                f"- Pool: `{payload['pool']}`",
                f"- Aggregator: `{payload['aggregator']}`",
                f"- k: `{payload['k']}`",
                "- Selected candidates:",
            ]
        )
        lines.extend(f"  - `{candidate}`" for candidate in payload["selected_candidates"])
        if payload.get("candidate_weights"):
            lines.extend(["", "- Candidate weights:"])
            lines.extend(
                f"  - `{candidate}`: `{weight:.4f}`"
                for candidate, weight in payload["candidate_weights"].items()
            )
        lines.append("")
    for horizon in sorted(combined["horizon"].unique()):
        top = combined.loc[combined["horizon"].eq(horizon)].head(20)
        lines.extend([f"## Horizon {horizon} Top 20", "", "```text", top.to_string(index=False), "```", ""])
    (output_dir / "DEPLOYMENT_COMBINATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
