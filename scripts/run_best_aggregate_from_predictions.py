#!/usr/bin/env python3
"""Build the best hard-vote aggregate model from rolling base predictions.

This runner uses the repository's metric and report stack, while taking the
already-completed out-of-sample base model predictions as ensemble inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.metrics import compute_all_metrics  # noqa: E402
from report.verdict import build_agent_verdict  # noqa: E402
from report.writer import write_experiment_report, write_model_outputs  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    """One base rolling prediction stream used by the hard-vote ensemble."""

    model: str
    feature_set: str
    lookback_months: int
    horizon_months: int
    head: str

    @property
    def cid(self) -> str:
        news = "news" if self.feature_set == "with_news_precomputed_pca" else "nonews"
        return f"{self.model}|{news}|lb{self.lookback_months}|h{self.horizon_months}|{self.head}"


H1_CANDIDATES = [
    Candidate("mlp_small_relu", "with_news_precomputed_pca", 6, 1, "cls"),
    Candidate("sgd_modified_huber", "with_news_precomputed_pca", 6, 1, "cls"),
    Candidate("lightgbm_dart", "with_news_precomputed_pca", 12, 1, "reg"),
    Candidate("keras_lstm_u16", "no_news", 12, 1, "cls"),
    Candidate("keras_tcn_filters16_k2_d1", "no_news", 9, 1, "reg"),
    Candidate("keras_gru_u16", "with_news_precomputed_pca", 12, 1, "reg"),
]

H2_CANDIDATES = [
    Candidate("aeon_knn_euclidean", "with_news_precomputed_pca", 6, 2, "reg"),
    Candidate("knn_5_distance", "with_news_precomputed_pca", 6, 2, "reg"),
    Candidate("aeon_deep_timecnn", "with_news_precomputed_pca", 12, 2, "cls"),
    Candidate("aeon_deep_timecnn", "no_news", 12, 2, "cls"),
    Candidate("knn_3_uniform", "with_news_precomputed_pca", 12, 2, "cls"),
    Candidate("aeon_rise", "no_news", 6, 2, "cls"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="all_rolling_predictions.csv or .csv.gz")
    parser.add_argument("--output-dir", default="experiments/best_aggregate_framework")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def read_predictions(path: str | Path) -> pd.DataFrame:
    """Read rolling predictions and validate required columns."""

    predictions = pd.read_csv(path)
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
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    return predictions


def build_candidate_matrix(predictions: pd.DataFrame, name: str, candidates: list[Candidate]) -> pd.DataFrame:
    """Align candidate predictions by calendar target and add vote columns."""

    frames = []
    for candidate in candidates:
        mask = (
            predictions["model"].eq(candidate.model)
            & predictions["feature_set"].eq(candidate.feature_set)
            & predictions["lookback_months"].eq(candidate.lookback_months)
            & predictions["horizon_months"].eq(candidate.horizon_months)
            & predictions["head"].eq(candidate.head)
        )
        part = predictions.loc[mask].copy()
        if part.empty:
            raise ValueError(f"Missing candidate stream for {candidate.cid}")
        part["candidate_id"] = candidate.cid
        frames.append(part)

    aligned = pd.concat(frames, ignore_index=True)
    index_cols = ["anchor_month", "target_month", "horizon_months", "actual_direction", "actual_return"]
    probs = aligned.pivot_table(index=index_cols, columns="candidate_id", values="predicted_probability", aggfunc="first")
    votes = aligned.pivot_table(index=index_cols, columns="candidate_id", values="predicted_direction", aggfunc="first")
    probs.columns = [f"prob::{column}" for column in probs.columns]
    votes.columns = [f"vote::{column}" for column in votes.columns]
    base = pd.concat([probs, votes], axis=1).reset_index()

    expected = len(candidates)
    vote_cols = [f"vote::{candidate.cid}" for candidate in candidates]
    present = base[vote_cols].notna().sum(axis=1)
    dropped = int((present != expected).sum())
    if dropped:
        print(f"[warn] {name}: dropping {dropped} incomplete early calendar rows")
        base = base.loc[present == expected].copy()
    base["anchor_month"] = pd.to_datetime(base["anchor_month"])
    base["target_month"] = pd.to_datetime(base["target_month"])
    return base.sort_values(["anchor_month", "target_month"]).reset_index(drop=True)


def aggregate_predictions(name: str, base: pd.DataFrame, candidates: list[Candidate]) -> tuple[pd.DataFrame, dict]:
    """Create hard-vote predictions and framework metrics."""

    vote_cols = [f"vote::{candidate.cid}" for candidate in candidates]
    vote_score = base[vote_cols].astype(float).mean(axis=1).to_numpy(float)
    pred = (vote_score > 0.5).astype(int)
    y_true = base["actual_direction"].to_numpy(int)
    actual_returns = base["actual_return"].to_numpy(float)
    strategy_returns = np.where(pred == 1, actual_returns, -actual_returns)

    predictions = pd.DataFrame(
        {
            "date": base["target_month"].dt.date.astype(str),
            "anchor_date": base["anchor_month"].dt.date.astype(str),
            "actual_label": y_true,
            "predicted_label": pred,
            "predicted_probability": vote_score,
            "predicted_return": "",
            "direction_correct": (pred == y_true).astype(int),
            "actual_return": actual_returns,
            "strategy_return": strategy_returns,
            "model": name,
            "window_id": np.arange(len(base), dtype=int),
            "train_start_date": "",
            "train_end_date": "",
            "val_start_date": "",
            "val_end_date": "",
            "test_date": base["target_month"].dt.date.astype(str),
        }
    )
    predictions["equity"] = (1.0 + predictions["strategy_return"]).cumprod()
    metrics = compute_all_metrics(
        y_true,
        vote_score,
        actual_returns,
        n_bootstrap=ARGS.bootstrap,
        ci_level=ARGS.ci_level,
        annualize=12,
    )
    metrics["candidate_count"] = len(candidates)
    metrics["rule"] = "hard_vote_strict_gt_0.5"
    return predictions, metrics


def single_candidate_metrics(predictions: pd.DataFrame, candidates: list[Candidate], bootstrap: int, ci_level: float) -> list[dict]:
    """Compute exact hard-label framework metrics for selected base candidates."""

    rows: list[dict] = []
    for candidate in candidates:
        mask = (
            predictions["model"].eq(candidate.model)
            & predictions["feature_set"].eq(candidate.feature_set)
            & predictions["lookback_months"].eq(candidate.lookback_months)
            & predictions["horizon_months"].eq(candidate.horizon_months)
            & predictions["head"].eq(candidate.head)
        )
        part = predictions.loc[mask].sort_values(["anchor_month", "target_month"]).copy()
        y_true = part["actual_direction"].to_numpy(int)
        # Use hard labels as scores so framework y_prob > 0.5 reproduces the
        # validation-thresholded base predictions exactly.
        hard_score = part["predicted_direction"].to_numpy(float)
        actual_returns = part["actual_return"].to_numpy(float)
        metrics = compute_all_metrics(
            y_true,
            hard_score,
            actual_returns,
            n_bootstrap=bootstrap,
            ci_level=ci_level,
            annualize=12,
        )
        rows.append({"model": candidate.cid, "group": "base_candidate", **metrics})
    return rows


def write_outputs(output_dir: Path, comparison: pd.DataFrame, model_payloads: dict[str, tuple[pd.DataFrame, dict]]) -> None:
    """Write standard framework report outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    best_row = comparison.iloc[0].to_dict()
    best_model = str(best_row["model"])
    best_predictions, _ = model_payloads[best_model]
    baseline = comparison.loc[comparison["group"].eq("base_candidate")].head(1)
    baseline_metrics = baseline.iloc[0].to_dict() if not baseline.empty else None
    verdict = build_agent_verdict(best_row, baseline_metrics=baseline_metrics, primary_metric="BalancedAcc")
    for model_name, (predictions, metrics) in model_payloads.items():
        write_model_outputs(output_dir, model_name, predictions, metrics)
    write_experiment_report(
        output_dir=output_dir,
        model_name=best_model,
        predictions=best_predictions,
        comparison=comparison,
        metrics=best_row,
        verdict=verdict,
        config={
            "source": "rolling base predictions",
            "aggregate_rule": "hard vote: positive when vote share > 0.5",
            "note": "Base predictions are already out-of-sample rolling predictions.",
        },
        write_model_output=False,
    )


def main() -> None:
    """Run H1/H2 hard-vote aggregate evaluation."""

    predictions = read_predictions(ARGS.predictions)
    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_payloads: dict[str, tuple[pd.DataFrame, dict]] = {}
    rows: list[dict] = []
    for name, candidates in [("top6_h1_hard_vote", H1_CANDIDATES), ("top6_h2_hard_vote", H2_CANDIDATES)]:
        base = build_candidate_matrix(predictions, name, candidates)
        pred_df, metrics = aggregate_predictions(name, base, candidates)
        model_payloads[name] = (pred_df, metrics)
        rows.append({"model": name, "group": "hard_vote_ensemble", **metrics})

    rows.extend(single_candidate_metrics(predictions, H1_CANDIDATES + H2_CANDIDATES, ARGS.bootstrap, ARGS.ci_level))
    comparison = pd.DataFrame(rows).sort_values(["BalancedAcc", "DirAcc", "ProfitFactor", "Sharpe"], ascending=False)
    write_outputs(output_dir, comparison, model_payloads)
    comparison.to_csv(output_dir / "best_aggregate_comparison.csv", index=False)
    (output_dir / "candidate_sets.json").write_text(
        json.dumps(
            {
                "top6_h1": [candidate.cid for candidate in H1_CANDIDATES],
                "top6_h2": [candidate.cid for candidate in H2_CANDIDATES],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(f"results written to {output_dir}")


if __name__ == "__main__":
    ARGS = parse_args()
    main()
