import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models.deployment_ensemble import (
    BEST_DEPLOYMENT_ENSEMBLE_SPECS,
    BEST_DEPLOYMENT_MODEL_POOL_NAME,
    DeploymentEnsembleSpec,
    BestDeploymentEnsemble,
    create_deployment_ensemble_model,
)
from models.registry import create_model, expand_model_configs


def _toy_prediction_rows() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=5, freq="MS")
    candidates = [
        ("model_a", "no_news", 6, 1, "cls", [1, 1, 0, 1, 0]),
        ("model_b", "with_news_precomputed_pca", 9, 1, "reg", [1, 0, 0, 1, 1]),
    ]
    rows = []
    y = [1, 0, 0, 1, 0]
    for idx, month in enumerate(months):
        for model, feature_set, lookback, horizon, head, votes in candidates:
            rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "lookback_months": lookback,
                    "horizon_months": horizon,
                    "head": head,
                    "anchor_month": month.strftime("%Y-%m-%d"),
                    "target_month": (month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                    "actual_direction": y[idx],
                    "actual_return": 0.02 if y[idx] else -0.01,
                    "predicted_direction": votes[idx],
                    "predicted_probability": 0.8 if votes[idx] else 0.2,
                }
            )
    return pd.DataFrame(rows)


def _best_h1_spec_prediction_rows() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=5, freq="MS")
    y = [1, 0, 0, 1, 0]
    rows = []
    spec = BEST_DEPLOYMENT_ENSEMBLE_SPECS["corn_h1_forward_replacement_ap_hard_vote"]
    for idx, month in enumerate(months):
        for candidate in spec.candidate_keys.values():
            rows.append(
                {
                    "model": candidate.model,
                    "feature_set": candidate.feature_set,
                    "lookback_months": candidate.lookback_months,
                    "horizon_months": candidate.horizon_months,
                    "head": candidate.head,
                    "anchor_month": month.strftime("%Y-%m-%d"),
                    "target_month": (month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                    "actual_direction": y[idx],
                    "actual_return": 0.02 if y[idx] else -0.01,
                    "predicted_direction": y[idx],
                    "predicted_probability": 0.8 if y[idx] else 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_best_deployment_ensemble_aggregates_weighted_hard_votes():
    spec = DeploymentEnsembleSpec(
        name="toy_deployment_ensemble",
        horizon_months=1,
        search_protocol="full_history_deployment_discovery",
        selection_mode="forward_replacement",
        ranking_seed="ba",
        forward_tie_breaker="ba_only",
        aggregator="hard_vote_strict",
        threshold=0.5,
        selected_count=4,
        candidate_weights={
            "model_a|nonews|lb6|h1|cls": 0.75,
            "model_b|news|lb9|h1|reg": 0.25,
        },
        expected_metrics={"BalancedAcc": 1.0, "AUC": 1.0, "AP": 1.0, "DirAcc": 1.0},
    )
    model = BestDeploymentEnsemble(spec)

    pred_df, metrics, audit, candidate_audit = model.aggregate_predictions(_toy_prediction_rows(), bootstrap=0)

    assert np.allclose(pred_df["predicted_probability"].to_numpy(), [1.0, 0.75, 0.0, 1.0, 0.25])
    assert pred_df["predicted_label"].tolist() == [1, 1, 0, 1, 0]
    assert metrics["selection_protocol"] == "full_history_deployment_discovery"
    assert metrics["selection_mode"] == "forward_replacement"
    assert audit["unique_candidate_count"] == 2
    assert len(candidate_audit) == 2


def test_best_deployment_model_pool_and_registry_entries_exist():
    expanded = expand_model_configs(BEST_DEPLOYMENT_MODEL_POOL_NAME)

    assert [row["name"] for row in expanded] == [
        "corn_h1_forward_replacement_ap_hard_vote",
        "corn_h2_forward_replacement_ba_hard_vote",
    ]
    assert all(row["type"] == "deployment_ensemble" for row in expanded)
    assert create_model(expanded[0]).spec.name == "corn_h1_forward_replacement_ap_hard_vote"
    assert create_deployment_ensemble_model("corn_h2_forward_replacement_ba_hard_vote").spec.horizon_months == 2


def test_best_deployment_combo_runner_writes_outputs(tmp_path: Path):
    predictions_path = tmp_path / "toy_predictions.csv"
    output_dir = tmp_path / "best_combo"
    _best_h1_spec_prediction_rows().to_csv(predictions_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_best_deployment_combo.py",
            "--predictions",
            str(predictions_path),
            "--output-dir",
            str(output_dir),
            "--horizons",
            "1",
            "--bootstrap",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "results written" in result.stdout
    comparison = pd.read_csv(output_dir / "best_deployment_combo_comparison.csv")
    assert not comparison.empty
    assert set(comparison["group"]) == {"best_deployment_ensemble"}
    assert (output_dir / "deployment_ensemble_specs.json").exists()
    assert (output_dir / "deployment_ensemble_audit.csv").exists()
