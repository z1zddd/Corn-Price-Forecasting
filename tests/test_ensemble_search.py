import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import search_prediction_ensembles as ensemble_search


def _toy_predictions() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=8, freq="MS")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    candidates = [
        ("model_a", "no_news", 6, "cls", [0, 1, 0, 1, 0, 1, 0, 1], 0.75),
        ("model_b", "no_news", 6, "cls", [1, 1, 0, 0, 0, 1, 1, 1], 0.65),
        ("model_c", "with_news_precomputed_pca", 9, "reg", [0, 0, 0, 1, 1, 1, 0, 0], 0.60),
    ]
    rows = []
    for i, month in enumerate(months):
        for model, feature_set, lookback, head, votes, high_prob in candidates:
            vote = int(votes[i])
            rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "lookback_months": lookback,
                    "horizon_months": 1,
                    "head": head,
                    "anchor_month": month.strftime("%Y-%m-%d"),
                    "target_month": (month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                    "actual_direction": int(y[i]),
                    "actual_return": 0.02 if y[i] else -0.01,
                    "predicted_direction": vote,
                    "predicted_probability": high_prob if vote else 1.0 - high_prob,
                    "family": "toy",
                    "package": "toy",
                }
            )
    return pd.DataFrame(rows)


def test_rolling_topk_uses_only_prior_target_labels(tmp_path: Path):
    predictions_path = tmp_path / "toy_predictions.csv"
    _toy_predictions().to_csv(predictions_path, index=False)
    predictions = ensemble_search.read_predictions(predictions_path)
    matrix = ensemble_search.build_matrix(predictions, horizon=1, min_coverage=1.0)
    prob = matrix["prob"]
    vote = matrix["vote"]
    y = matrix["base"]["actual_direction"].to_numpy(int)

    original = ensemble_search.rolling_topk_score(
        prob,
        vote,
        y,
        metric="ba",
        k=1,
        window=9999,
        source="hard",
        threshold_mode="fixed",
        min_history=2,
    )
    changed_future = y.copy()
    changed_future[-1] = 1 - changed_future[-1]
    mutated = ensemble_search.rolling_topk_score(
        prob,
        vote,
        changed_future,
        metric="ba",
        k=1,
        window=9999,
        source="hard",
        threshold_mode="fixed",
        min_history=2,
    )

    np.testing.assert_allclose(original[:-1], mutated[:-1])


def test_diverse_greedy_uses_only_prior_target_labels(tmp_path: Path):
    predictions_path = tmp_path / "toy_predictions.csv"
    _toy_predictions().to_csv(predictions_path, index=False)
    predictions = ensemble_search.read_predictions(predictions_path)
    matrix = ensemble_search.build_matrix(predictions, horizon=1, min_coverage=1.0)
    prob = matrix["prob"]
    vote = matrix["vote"]
    y = matrix["base"]["actual_direction"].to_numpy(int)

    original = ensemble_search.rolling_diverse_greedy_score(
        prob,
        vote,
        y,
        metric="ba",
        topn=3,
        k=2,
        window=9999,
        diversity_lambda=0.1,
        source="hard",
        threshold_mode="rolling_ba",
        min_history=2,
    )
    changed_future = y.copy()
    changed_future[-1] = 1 - changed_future[-1]
    mutated = ensemble_search.rolling_diverse_greedy_score(
        prob,
        vote,
        changed_future,
        metric="ba",
        topn=3,
        k=2,
        window=9999,
        diversity_lambda=0.1,
        source="hard",
        threshold_mode="rolling_ba",
        min_history=2,
    )

    np.testing.assert_allclose(original[:-1], mutated[:-1])


def test_ensemble_search_cli_writes_walk_forward_leaderboard(tmp_path: Path):
    predictions_path = tmp_path / "toy_predictions.csv"
    output_dir = tmp_path / "ensemble_search"
    _toy_predictions().to_csv(predictions_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    result = subprocess.run(
        [
            sys.executable,
            "scripts/search_prediction_ensembles.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--horizons",
            "1",
            "--families",
            "simple,threshold,topk,oracle",
            "--preset",
            "fast",
            "--min-history",
            "2",
            "--min-candidate-coverage",
            "1.0",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    leaderboard = pd.read_csv(output_dir / "valid_walk_forward_leaderboard.csv")
    summary = pd.read_csv(output_dir / "h1_ensemble_search_summary.csv")
    assert "results written" in result.stdout
    assert not leaderboard.empty
    assert set(leaderboard["validation_mode"]) == {"valid_walk_forward"}
    assert "diagnostic_oracle" in set(summary["validation_mode"])
