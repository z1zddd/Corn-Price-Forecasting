#!/usr/bin/env python3
"""Run the fixed best deployment ensembles from completed rolling predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_forecast.operator.model.families.aggregation.deployment_vote import BEST_DEPLOYMENT_ENSEMBLE_SPECS, create_deployment_ensemble_model  # noqa: E402
from corn_forecast.pipeline.report.verdict import build_agent_verdict  # noqa: E402
from corn_forecast.pipeline.report.writer import write_experiment_report, write_model_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="all_rolling_predictions.csv or .csv.gz")
    parser.add_argument("--output-dir", default="experiments/best_deployment_combo")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def read_predictions(path: str | Path) -> pd.DataFrame:
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


def selected_specs(horizons: set[int]) -> list[str]:
    names = [
        name
        for name, spec in BEST_DEPLOYMENT_ENSEMBLE_SPECS.items()
        if int(spec.horizon_months) in horizons
    ]
    if not names:
        raise ValueError(f"No best deployment ensemble specs found for horizons={sorted(horizons)}")
    return names


def write_audits(output_dir: Path, audits: list[dict], candidate_audits: list[dict]) -> None:
    pd.DataFrame(audits).to_csv(output_dir / "deployment_ensemble_audit.csv", index=False)
    pd.DataFrame(candidate_audits).to_csv(output_dir / "deployment_ensemble_candidate_audit.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = {int(value.strip()) for value in str(args.horizons).split(",") if value.strip()}
    predictions = read_predictions(args.predictions)

    rows: list[dict] = []
    audits: list[dict] = []
    candidate_audits: list[dict] = []
    model_payloads = {}
    specs_payload = {}
    for model_name in selected_specs(horizons):
        model = create_deployment_ensemble_model(model_name)
        pred_df, metrics, audit, candidate_audit = model.aggregate_predictions(
            predictions,
            bootstrap=args.bootstrap,
            ci_level=args.ci_level,
        )
        rows.append({"model": model_name, "group": "best_deployment_ensemble", **metrics})
        audits.append(audit)
        candidate_audits.extend(candidate_audit)
        model_payloads[model_name] = (pred_df, metrics)
        specs_payload[model_name] = {
            "horizon_months": model.spec.horizon_months,
            "search_protocol": model.spec.search_protocol,
            "selection_mode": model.spec.selection_mode,
            "ranking_seed": model.spec.ranking_seed,
            "forward_tie_breaker": model.spec.forward_tie_breaker,
            "aggregator": model.spec.aggregator,
            "threshold": model.spec.threshold,
            "selected_count": model.spec.selected_count,
            "candidate_weights": model.spec.candidate_weights,
            "expected_metrics": model.spec.expected_metrics,
        }

    comparison = pd.DataFrame(rows).sort_values(
        ["BalancedAcc", "AUC", "AP", "DirAcc"],
        ascending=False,
    )
    for model_name, (pred_df, metrics) in model_payloads.items():
        write_model_outputs(output_dir, model_name, pred_df, metrics)
    best_row = comparison.iloc[0].to_dict()
    verdict = build_agent_verdict(best_row, baseline_metrics=None, primary_metric="BalancedAcc")
    write_experiment_report(
        output_dir=output_dir,
        model_name=str(best_row["model"]),
        predictions=model_payloads[str(best_row["model"])][0],
        comparison=comparison,
        metrics=best_row,
        verdict=verdict,
        config={
            "source": "completed rolling base predictions",
            "model_pool": "best_deployment_forward_ensembles",
            "selection_protocol": "full_history_deployment_discovery",
            "note": (
                "These ensembles are fixed上线 candidates selected from completed historical rolling "
                "prediction streams. Report separately from strict walk-forward automatic model-selection scores."
            ),
        },
        write_model_output=False,
    )
    comparison.to_csv(output_dir / "best_deployment_combo_comparison.csv", index=False)
    (output_dir / "deployment_ensemble_specs.json").write_text(
        json.dumps(specs_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_audits(output_dir, audits, candidate_audits)
    print(comparison.to_string(index=False))
    print(f"results written to {output_dir}")


if __name__ == "__main__":
    main()
