"""Deployment ensemble models built from completed rolling prediction streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from corn_forecast.pipeline.eval.metrics import compute_all_metrics


FEATURE_SET_ALIASES = {
    "news": "with_news_precomputed_pca",
    "nonews": "no_news",
    "with_news_precomputed_pca": "with_news_precomputed_pca",
    "no_news": "no_news",
}


@dataclass(frozen=True)
class DeploymentCandidateKey:
    """One base prediction stream used by a deployment ensemble."""

    model: str
    feature_set: str
    lookback_months: int
    horizon_months: int
    head: str

    @property
    def candidate_id(self) -> str:
        news = "news" if self.feature_set == "with_news_precomputed_pca" else "nonews"
        return f"{self.model}|{news}|lb{self.lookback_months}|h{self.horizon_months}|{self.head}"

    @classmethod
    def parse(cls, value: str) -> "DeploymentCandidateKey":
        model, feature, lookback, horizon, head = str(value).split("|")
        return cls(
            model=model,
            feature_set=FEATURE_SET_ALIASES[feature],
            lookback_months=int(lookback.removeprefix("lb")),
            horizon_months=int(horizon.removeprefix("h")),
            head=head,
        )


@dataclass(frozen=True)
class DeploymentEnsembleSpec:
    """Fixed full-history deployment ensemble selected from rolling predictions."""

    name: str
    horizon_months: int
    search_protocol: str
    selection_mode: str
    ranking_seed: str
    forward_tie_breaker: str
    aggregator: str
    threshold: float
    selected_count: int
    candidate_weights: dict[str, float]
    expected_metrics: dict[str, float]

    @property
    def candidate_keys(self) -> dict[str, DeploymentCandidateKey]:
        return {candidate_id: DeploymentCandidateKey.parse(candidate_id) for candidate_id in self.candidate_weights}


BEST_DEPLOYMENT_ENSEMBLE_SPECS: dict[str, DeploymentEnsembleSpec] = {
    "corn_h1_forward_replacement_ap_hard_vote": DeploymentEnsembleSpec(
        name="corn_h1_forward_replacement_ap_hard_vote",
        horizon_months=1,
        search_protocol="full_history_deployment_discovery",
        selection_mode="forward_replacement",
        ranking_seed="ap",
        forward_tie_breaker="ba_only",
        aggregator="hard_vote_strict",
        threshold=0.5,
        selected_count=32,
        candidate_weights={
            "mlp_small_relu|news|lb6|h1|cls": 0.09375,
            "keras_tcn_filters8_k2_d1|nonews|lb9|h1|reg": 0.03125,
            "keras_tcn_filters16_k2_d1|nonews|lb9|h1|reg": 0.09375,
            "svc_sigmoid|nonews|lb9|h1|cls": 0.0625,
            "keras_lstm_u16|nonews|lb12|h1|cls": 0.03125,
            "svc_rbf|news|lb6|h1|cls": 0.09375,
            "aeon_deep_mlp|news|lb6|h1|reg": 0.0625,
            "aeon_knn_dtw|news|lb9|h1|cls": 0.03125,
            "extra_tree_entropy|news|lb9|h1|cls": 0.03125,
            "keras_tcn_filters16_k2_d1|news|lb9|h1|cls": 0.03125,
            "aeon_deep_mlp|nonews|lb9|h1|cls": 0.03125,
            "aeon_deep_mlp|nonews|lb12|h1|cls": 0.03125,
            "lightgbm_goss|news|lb6|h1|cls": 0.03125,
            "hist_gradient_boosting|news|lb12|h1|cls": 0.03125,
            "gradient_boosting|nonews|lb6|h1|reg": 0.03125,
            "aeon_deep_mlp|nonews|lb9|h1|reg": 0.03125,
            "svc_rbf|nonews|lb9|h1|cls": 0.03125,
            "aeon_deep_mlp|nonews|lb6|h1|cls": 0.0625,
            "keras_lstm_u16|nonews|lb6|h1|cls": 0.03125,
            "logistic_l1_liblinear|news|lb6|h1|reg": 0.03125,
            "keras_tcn_filters8_k2_d1|news|lb6|h1|cls": 0.03125,
            "aeon_deep_mlp|news|lb12|h1|cls": 0.03125,
            "aeon_minirocket|news|lb6|h1|reg": 0.03125,
        },
        expected_metrics={"BalancedAcc": 0.9359, "AUC": 0.9293, "AP": 0.9072, "DirAcc": 0.9359},
    ),
    "corn_h2_forward_replacement_ba_hard_vote": DeploymentEnsembleSpec(
        name="corn_h2_forward_replacement_ba_hard_vote",
        horizon_months=2,
        search_protocol="full_history_deployment_discovery",
        selection_mode="forward_replacement",
        ranking_seed="ba",
        forward_tie_breaker="balanced",
        aggregator="hard_vote_strict",
        threshold=0.5,
        selected_count=34,
        candidate_weights={
            "aeon_knn_euclidean|news|lb6|h2|reg": 0.08823529411764706,
            "aeon_deep_timecnn|news|lb12|h2|cls": 0.08823529411764706,
            "svc_sigmoid|news|lb9|h2|cls": 0.14705882352941177,
            "knn_5_distance|news|lb12|h2|cls": 0.029411764705882353,
            "extra_tree_gini|news|lb12|h2|reg": 0.08823529411764706,
            "aeon_deep_timecnn|nonews|lb6|h2|cls": 0.029411764705882353,
            "gradient_boosting|news|lb9|h2|cls": 0.029411764705882353,
            "aeon_rise|nonews|lb6|h2|cls": 0.029411764705882353,
            "aeon_deep_inceptiontime|nonews|lb6|h2|cls": 0.029411764705882353,
            "aeon_deep_timecnn|nonews|lb9|h2|reg": 0.058823529411764705,
            "xgboost_dart|news|lb6|h2|cls": 0.029411764705882353,
            "extra_tree_gini|news|lb6|h2|cls": 0.029411764705882353,
            "aeon_deep_mlp|news|lb12|h2|reg": 0.029411764705882353,
            "hist_gradient_boosting|news|lb6|h2|cls": 0.029411764705882353,
            "lightgbm_dart|news|lb12|h2|reg": 0.029411764705882353,
            "aeon_deep_mlp|news|lb6|h2|reg": 0.029411764705882353,
            "aeon_deep_mlp|news|lb12|h2|cls": 0.029411764705882353,
            "sgd_log_loss|news|lb9|h2|reg": 0.029411764705882353,
            "aeon_rise|news|lb9|h2|cls": 0.029411764705882353,
            "aeon_minirocket|news|lb12|h2|reg": 0.029411764705882353,
            "logistic_l1_liblinear|news|lb9|h2|cls": 0.029411764705882353,
            "hist_gradient_boosting|nonews|lb6|h2|cls": 0.029411764705882353,
            "mlp_small_relu|news|lb12|h2|cls": 0.029411764705882353,
        },
        expected_metrics={"BalancedAcc": 0.9432, "AUC": 0.9084, "AP": 0.9550, "DirAcc": 0.9342},
    ),
}


BEST_DEPLOYMENT_MODEL_POOL_NAME = "best_deployment_forward_ensembles"


class BestDeploymentEnsemble:
    """Aggregate selected base prediction streams into a fixed deployment model."""

    def __init__(self, spec: DeploymentEnsembleSpec) -> None:
        self.spec = spec
        self.model_family = "deployment_ensemble"
        self.package = "framework"
        self.input_kind = "rolling_predictions"
        self.source_kind = spec.search_protocol

    def fit(self, *_args, **_kwargs) -> "BestDeploymentEnsemble":
        return self

    def fit_with_targets(self, *_args, **_kwargs) -> "BestDeploymentEnsemble":
        return self

    def aggregate_predictions(
        self,
        predictions: pd.DataFrame,
        *,
        bootstrap: int = 0,
        ci_level: float = 0.95,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        base, candidate_audit = self._build_aligned_matrix(predictions)
        score = self._score_matrix(base)
        valid_score = np.isfinite(score)
        base_scored = base.loc[valid_score].copy()
        score = score[valid_score]
        pred = (score > self.spec.threshold).astype(int)
        y_true = base_scored["actual_direction"].to_numpy(int)
        actual_returns = base_scored["actual_return"].to_numpy(float)
        strategy_returns = np.where(pred == 1, actual_returns, -actual_returns)
        pred_df = pd.DataFrame(
            {
                "date": base_scored["target_month"].dt.date.astype(str),
                "anchor_date": base_scored["anchor_month"].dt.date.astype(str),
                "actual_label": y_true,
                "predicted_label": pred,
                "predicted_probability": score,
                "direction_correct": (pred == y_true).astype(int),
                "actual_return": actual_returns,
                "strategy_return": strategy_returns,
                "model": self.spec.name,
                "window_id": np.arange(len(base), dtype=int),
                "test_date": base_scored["target_month"].dt.date.astype(str),
            }
        )
        pred_df["equity"] = (1.0 + pred_df["strategy_return"]).cumprod()
        metrics = compute_all_metrics(
            y_true,
            score,
            actual_returns,
            n_bootstrap=bootstrap,
            ci_level=ci_level,
            annualize=12,
        )
        metrics.update(
            {
                "selection_protocol": self.spec.search_protocol,
                "selection_mode": self.spec.selection_mode,
                "ranking_seed": self.spec.ranking_seed,
                "forward_tie_breaker": self.spec.forward_tie_breaker,
                "aggregator": self.spec.aggregator,
                "threshold": self.spec.threshold,
                "selected_count": self.spec.selected_count,
                "unique_candidate_count": len(self.spec.candidate_weights),
                "expected_BalancedAcc": self.spec.expected_metrics["BalancedAcc"],
                "expected_AUC": self.spec.expected_metrics["AUC"],
                "expected_AP": self.spec.expected_metrics["AP"],
                "expected_DirAcc": self.spec.expected_metrics["DirAcc"],
            }
        )
        audit = {
            "model": self.spec.name,
            "selection_protocol": self.spec.search_protocol,
            "selection_mode": self.spec.selection_mode,
            "ranking_seed": self.spec.ranking_seed,
            "forward_tie_breaker": self.spec.forward_tie_breaker,
            "aggregator": self.spec.aggregator,
            "selected_count": self.spec.selected_count,
            "unique_candidate_count": len(self.spec.candidate_weights),
            "raw_calendar_rows": int(base.attrs["raw_calendar_rows"]),
            "scored_calendar_rows": int(len(base_scored)),
            "dropped_no_score_calendar_rows": int((~valid_score).sum()),
            "first_target_month": str(base_scored["target_month"].min().date()) if not base_scored.empty else "",
            "last_target_month": str(base_scored["target_month"].max().date()) if not base_scored.empty else "",
        }
        return pred_df, metrics, audit, candidate_audit

    def save(self, path: str | Path) -> None:
        joblib.dump({"spec": self.spec}, path)

    def _build_aligned_matrix(self, predictions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
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
        frames = []
        candidate_audit: list[dict[str, Any]] = []
        for candidate_id, candidate in self.spec.candidate_keys.items():
            mask = (
                predictions["model"].astype(str).eq(candidate.model)
                & predictions["feature_set"].astype(str).eq(candidate.feature_set)
                & predictions["lookback_months"].astype(int).eq(candidate.lookback_months)
                & predictions["horizon_months"].astype(int).eq(candidate.horizon_months)
                & predictions["head"].astype(str).eq(candidate.head)
            )
            part = predictions.loc[mask].copy()
            if part.empty:
                raise ValueError(f"Missing deployment ensemble candidate stream: {candidate_id}")
            part["candidate_id"] = candidate_id
            candidate_audit.append(
                {
                    "ensemble": self.spec.name,
                    "candidate_id": candidate_id,
                    "weight": self.spec.candidate_weights[candidate_id],
                    "rows": int(len(part)),
                    "first_target_month": str(pd.to_datetime(part["target_month"]).min().date()),
                    "last_target_month": str(pd.to_datetime(part["target_month"]).max().date()),
                }
            )
            frames.append(part)
        aligned = pd.concat(frames, ignore_index=True)
        index_cols = ["anchor_month", "target_month", "horizon_months", "actual_direction", "actual_return"]
        value_col = "predicted_direction" if self.spec.aggregator.startswith("hard") else "predicted_probability"
        matrix = aligned.pivot_table(index=index_cols, columns="candidate_id", values=value_col, aggfunc="first")
        matrix = matrix.reset_index()
        raw_calendar_rows = int(len(matrix))
        matrix["anchor_month"] = pd.to_datetime(matrix["anchor_month"])
        matrix["target_month"] = pd.to_datetime(matrix["target_month"])
        matrix = matrix.sort_values(["anchor_month", "target_month"]).reset_index(drop=True)
        matrix.attrs["raw_calendar_rows"] = raw_calendar_rows
        return matrix, candidate_audit

    def _score_matrix(self, matrix: pd.DataFrame) -> np.ndarray:
        numerator = np.zeros(len(matrix), dtype=float)
        denominator = np.zeros(len(matrix), dtype=float)
        for candidate_id, weight in self.spec.candidate_weights.items():
            values = matrix[candidate_id].to_numpy(float)
            valid = np.isfinite(values)
            numerator += np.nan_to_num(values, nan=0.0) * float(weight)
            denominator += valid.astype(float) * float(weight)
        score = np.full(len(matrix), np.nan, dtype=float)
        np.divide(numerator, denominator, out=score, where=denominator > 0)
        return score


def deployment_ensemble_model_configs() -> list[dict[str, Any]]:
    """Return framework model configs for the best fixed deployment ensembles."""

    return [
        {"name": name, "type": "deployment_ensemble", "enabled": True}
        for name in BEST_DEPLOYMENT_ENSEMBLE_SPECS
    ]


def create_deployment_ensemble_model(model_name: str, params: dict | None = None) -> BestDeploymentEnsemble:
    """Create a fixed deployment ensemble model by name."""

    params = dict(params or {})
    spec_name = str(params.pop("spec_name", model_name))
    if params:
        raise ValueError(f"Unsupported deployment_ensemble params for {spec_name}: {sorted(params)}")
    if spec_name not in BEST_DEPLOYMENT_ENSEMBLE_SPECS:
        raise ValueError(f"Unknown deployment ensemble model: {spec_name}")
    return BestDeploymentEnsemble(BEST_DEPLOYMENT_ENSEMBLE_SPECS[spec_name])
