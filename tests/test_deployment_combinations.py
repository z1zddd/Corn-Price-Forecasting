import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _toy_predictions() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=10, freq="MS")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    specs = [
        ("model_a", "no_news", 6, "cls", [0, 1, 0, 1, 0, 1, 0, 1, 1, 0], 0.80, "linear"),
        ("model_b", "with_news_precomputed_pca", 6, "reg", [0, 1, 1, 1, 0, 1, 0, 0, 1, 0], 0.70, "boosting"),
        ("model_c", "no_news", 9, "cls", [1, 1, 0, 1, 0, 0, 0, 1, 1, 0], 0.65, "neighbors"),
        ("model_d", "with_news_precomputed_pca", 9, "reg", [0, 0, 0, 1, 1, 1, 0, 1, 0, 0], 0.60, "deep_sequence"),
    ]
    rows = []
    for i, month in enumerate(months):
        for model, feature_set, lookback, head, votes, high_prob, family in specs:
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
                    "family": family,
                    "package": "toy",
                }
            )
    return pd.DataFrame(rows)


def test_deployment_combination_cli_writes_best_candidate(tmp_path: Path):
    predictions_path = tmp_path / "toy_predictions.csv"
    output_dir = tmp_path / "deployment_search"
    _toy_predictions().to_csv(predictions_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    result = subprocess.run(
        [
            sys.executable,
            "scripts/search_deployment_combinations.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--horizons",
            "1",
            "--max-pool-size",
            "4",
            "--pool-sizes",
            "2,4",
            "--rank-metrics",
            "ba,brier",
            "--keep-top",
            "20",
            "--min-candidate-coverage",
            "1.0",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    leaderboard = pd.read_csv(output_dir / "deployment_combination_leaderboard.csv")
    best = json.loads((output_dir / "h1_best_deployment_candidates.json").read_text(encoding="utf-8"))
    assert "results written" in result.stdout
    assert not leaderboard.empty
    assert set(leaderboard["search_protocol"]) == {"full_history_deployment_discovery"}
    assert leaderboard["k"].max() >= 2
    assert best["search_protocol"] == "full_history_deployment_discovery"
    assert len(best["selected_candidates"]) == best["k"]
