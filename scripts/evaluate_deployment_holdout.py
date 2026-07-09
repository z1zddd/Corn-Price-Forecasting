#!/usr/bin/env python3
"""Evaluate deployment ensemble selection on a held-out calendar period."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_forecast.pipeline.eval.metrics import compute_all_metrics  # noqa: E402
from scripts import search_deployment_combinations as deploy_search  # noqa: E402
from scripts import search_prediction_ensembles as base_search  # noqa: E402


@dataclass(frozen=True)
class HorizonRecipe:
    rank_metric: str
    tie_breaker: str
    aggregator: str
    search_scope: str
    max_k: int


DEFAULT_RECIPES = {
    1: HorizonRecipe(
        rank_metric="ap",
        tie_breaker="ba_only",
        aggregator="hard_vote_strict",
        search_scope="all",
        max_k=80,
    ),
    2: HorizonRecipe(
        rank_metric="ba",
        tie_breaker="balanced",
        aggregator="hard_vote_strict",
        search_scope="all",
        max_k=80,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="all_rolling_predictions.csv or .csv.gz")
    parser.add_argument("--output-dir", default="experiments/deployment_holdout")
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--holdout-end", default="", help="Inclusive holdout end date; empty means use all later rows.")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--min-candidate-coverage", type=float, default=0.90)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = base_search.read_predictions(args.predictions)
    holdout_start = pd.Timestamp(args.holdout_start)
    holdout_end = pd.Timestamp(args.holdout_end) if str(args.holdout_end).strip() else None
    horizons = [int(value.strip()) for value in str(args.horizons).split(",") if value.strip()]

    train_predictions = predictions.loc[predictions["target_month"] < holdout_start].copy()
    holdout_mask = predictions["target_month"] >= holdout_start
    if holdout_end is not None:
        holdout_mask &= predictions["target_month"] <= holdout_end
    holdout_predictions = predictions.loc[holdout_mask].copy()

    rows: list[dict[str, Any]] = []
    selected_payloads: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        recipe = DEFAULT_RECIPES[horizon]
        train_matrix = base_search.build_matrix(train_predictions, horizon, args.min_candidate_coverage)
        full_matrix = base_search.build_matrix(predictions, horizon, 0.0)
        holdout_matrix = slice_matrix_by_dates(full_matrix, holdout_start, holdout_end)

        candidate_scores = deploy_search.score_candidates(train_matrix)
        pools = deploy_search.build_forward_pools(
            train_matrix["inventory"],
            candidate_scores,
            [recipe.rank_metric],
            candidate_limit=0,
            requested_scopes={recipe.search_scope},
        )
        search_args = argparse.Namespace(
            bootstrap=args.bootstrap,
            ci_level=args.ci_level,
            enabled_aggregators={recipe.aggregator},
            forward_max_k=recipe.max_k,
            forward_tie_breakers=recipe.tie_breaker,
            threshold_grid_size=31,
        )
        train_rows, train_predictions_by_method = deploy_search.search_forward_horizon(
            train_matrix,
            pools,
            candidate_scores,
            search_args,
            allow_replacement=True,
        )
        if not train_rows:
            raise RuntimeError(f"No train-time deployment ensemble candidates found for horizon={horizon}")
        train_table = pd.DataFrame(train_rows).sort_values(
            ["BalancedAcc", "AUC", "AP", "DirAcc", "n_predictions"],
            ascending=[False, False, False, False, False],
        )
        best_train_row = train_table.iloc[0].to_dict()
        selected_candidates = json.loads(best_train_row["selected_candidates"])
        candidate_weights = json.loads(best_train_row["candidate_weights"])
        train_pred_df = train_predictions_by_method[str(best_train_row["method"])]

        holdout_score = score_selected_candidates(
            holdout_matrix,
            selected_candidates,
            recipe.aggregator,
            candidate_scores,
        )
        holdout_row, holdout_pred_df = payload_for_fixed_selection(
            holdout_matrix,
            holdout_score,
            selected_candidates,
            candidate_weights,
            best_train_row,
            recipe,
            args,
        )

        key = f"h{horizon}"
        selected_payloads[key] = {
            "horizon": horizon,
            "holdout_start": str(holdout_start.date()),
            "holdout_end": str(holdout_end.date()) if holdout_end is not None else "",
            "selection_rows": int(len(train_matrix["base"])),
            "holdout_rows": int(len(holdout_pred_df)),
            "train_method": best_train_row["method"],
            "selection_rank_metric": recipe.rank_metric,
            "forward_tie_breaker": recipe.tie_breaker,
            "aggregator": recipe.aggregator,
            "selected_count": int(best_train_row["k"]),
            "unique_candidate_count": len(candidate_weights),
            "train_metrics": metric_subset(best_train_row),
            "holdout_metrics": metric_subset(holdout_row),
            "selected_candidates": selected_candidates,
            "candidate_weights": candidate_weights,
        }
        rows.append(
            {
                "horizon": horizon,
                "selection_period": f"<{holdout_start.date()}",
                "holdout_period": (
                    f"{holdout_start.date()}..{holdout_end.date()}"
                    if holdout_end is not None
                    else f">={holdout_start.date()}"
                ),
                "selection_rank_metric": recipe.rank_metric,
                "forward_tie_breaker": recipe.tie_breaker,
                "aggregator": recipe.aggregator,
                "selected_count": int(best_train_row["k"]),
                "unique_candidate_count": len(candidate_weights),
                "train_n": int(best_train_row["n_predictions"]),
                "holdout_n": int(holdout_row["n_predictions"]),
                **prefixed_metrics("train", best_train_row),
                **prefixed_metrics("holdout", holdout_row),
            }
        )

        train_table.to_csv(output_dir / f"h{horizon}_train_selection_leaderboard.csv", index=False)
        train_pred_df.to_csv(output_dir / f"h{horizon}_train_selected_predictions.csv", index=False)
        holdout_pred_df.to_csv(output_dir / f"h{horizon}_holdout_predictions.csv", index=False)
        (output_dir / f"h{horizon}_selected_candidates.json").write_text(
            json.dumps(selected_payloads[key], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = pd.DataFrame(rows).sort_values("horizon")
    summary.to_csv(output_dir / "holdout_summary.csv", index=False)
    (output_dir / "holdout_selected_candidates.json").write_text(
        json.dumps(selected_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, summary, selected_payloads)
    print(summary.to_string(index=False))
    print(f"results written to {output_dir}")


def slice_matrix_by_dates(matrix: dict, start: pd.Timestamp, end: pd.Timestamp | None) -> dict:
    mask = matrix["base"]["target_month"] >= start
    if end is not None:
        mask &= matrix["base"]["target_month"] <= end
    return {
        **matrix,
        "base": matrix["base"].loc[mask].reset_index(drop=True),
        "prob": matrix["prob"].loc[mask].reset_index(drop=True),
        "vote": matrix["vote"].loc[mask].reset_index(drop=True),
    }


def score_selected_candidates(
    matrix: dict,
    selected_candidates: list[str],
    aggregator: str,
    candidate_scores: dict[str, dict[str, float]],
) -> np.ndarray:
    missing = [candidate for candidate in selected_candidates if candidate not in matrix["prob"].columns]
    if missing:
        raise ValueError(f"Selected candidates missing from holdout matrix: {missing[:10]}")
    source = matrix["vote"] if aggregator.startswith("hard") else matrix["prob"]
    values = np.column_stack([source[candidate].to_numpy(float) for candidate in selected_candidates])
    valid = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)
    if "_weighted_" in aggregator:
        weights = deploy_search.aggregator_candidate_weights(selected_candidates, aggregator, candidate_scores)
    else:
        weights = np.ones(len(selected_candidates), dtype=float)
    numerator = filled @ weights
    denominator = valid @ weights
    score = np.full(len(matrix["base"]), np.nan, dtype=float)
    np.divide(numerator, denominator, out=score, where=denominator > 0)
    if aggregator == "hard_vote_tie_up":
        score = np.where(score >= 0.5, 0.500001, 0.499999)
    return score


def payload_for_fixed_selection(
    matrix: dict,
    score: np.ndarray,
    selected_candidates: list[str],
    candidate_weights: dict[str, float],
    train_row: dict[str, Any],
    recipe: HorizonRecipe,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    valid = np.isfinite(score)
    base = matrix["base"].loc[valid].reset_index(drop=True)
    score_valid = score[valid]
    y = base["actual_direction"].to_numpy(int)
    returns = base["actual_return"].to_numpy(float)
    metrics = compute_all_metrics(
        y,
        score_valid,
        returns,
        n_bootstrap=args.bootstrap,
        ci_level=args.ci_level,
        annualize=12,
    )
    pred = (score_valid > 0.5).astype(int)
    pred_df = pd.DataFrame(
        {
            "date": base["target_month"].dt.date.astype(str),
            "anchor_date": base["anchor_month"].dt.date.astype(str),
            "actual_label": y,
            "predicted_label": pred,
            "predicted_probability": score_valid,
            "direction_correct": (pred == y).astype(int),
            "actual_return": returns,
            "strategy_return": np.where(pred == 1, returns, -returns),
            "model": f"holdout_fixed_from_{train_row['method']}",
            "window_id": np.arange(len(base), dtype=int),
            "test_date": base["target_month"].dt.date.astype(str),
        }
    )
    pred_df["equity"] = (1.0 + pred_df["strategy_return"]).cumprod()
    row = {
        "horizon": int(matrix["horizon"]),
        "method": f"holdout_fixed_from_{train_row['method']}",
        "selection_protocol": "pre_holdout_deployment_discovery",
        "method_family": "deployment_fixed_combination_holdout",
        "selection_mode": "forward_replacement",
        "pool": train_row["pool"],
        "scope": train_row["scope"],
        "rank_metric": recipe.rank_metric,
        "aggregator": recipe.aggregator,
        "forward_tie_breaker": recipe.tie_breaker,
        "k": len(selected_candidates),
        "threshold": 0.5,
        "coverage": float(valid.mean()),
        "n_predictions": int(valid.sum()),
        "selected_candidates": json.dumps(selected_candidates, ensure_ascii=False),
        "candidate_weights": json.dumps(candidate_weights, ensure_ascii=False),
        **metrics,
    }
    return row, pred_df


def metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "DirAcc",
        "BalancedAcc",
        "AUC",
        "AP",
        "Sharpe",
        "AnnRet",
        "ProfitFactor",
        "MaxDD",
        "Precision",
        "Recall",
        "Specificity",
        "Brier",
    ]
    return {key: row.get(key) for key in keys if key in row}


def prefixed_metrics(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metric_subset(row).items()}


def write_report(output_dir: Path, summary: pd.DataFrame, payloads: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Deployment Holdout Evaluation",
        "",
        "Protocol: select deployment ensemble candidates before the holdout period, then apply the fixed candidates and weights to the holdout period.",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
    ]
    for key, payload in payloads.items():
        lines.extend(
            [
                f"## {key}",
                "",
                f"- Selection rows: `{payload['selection_rows']}`",
                f"- Holdout rows: `{payload['holdout_rows']}`",
                f"- Rank metric: `{payload['selection_rank_metric']}`",
                f"- Tie breaker: `{payload['forward_tie_breaker']}`",
                f"- Aggregator: `{payload['aggregator']}`",
                f"- Selected count: `{payload['selected_count']}`",
                f"- Unique candidates: `{payload['unique_candidate_count']}`",
                f"- Train metrics: `{payload['train_metrics']}`",
                f"- Holdout metrics: `{payload['holdout_metrics']}`",
                "",
            ]
        )
    (output_dir / "HOLDOUT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
