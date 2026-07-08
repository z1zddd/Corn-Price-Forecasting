from pathlib import Path

import pandas as pd

from scripts import run_best_aggregate_from_predictions as aggregate_runner


def _aggregate_toy_predictions() -> pd.DataFrame:
    months = pd.date_range("2021-01-01", periods=4, freq="MS")
    rows = []
    candidates = [
        aggregate_runner.Candidate("model_a", "no_news", 6, 1, "cls"),
        aggregate_runner.Candidate("model_b", "with_news_precomputed_pca", 9, 1, "reg"),
    ]
    for i, month in enumerate(months):
        for candidate in candidates:
            if candidate.model == "model_b" and i == 0:
                continue
            rows.append(
                {
                    "model": candidate.model,
                    "feature_set": candidate.feature_set,
                    "lookback_months": candidate.lookback_months,
                    "horizon_months": candidate.horizon_months,
                    "head": candidate.head,
                    "anchor_month": month.strftime("%Y-%m-%d"),
                    "target_month": (month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
                    "actual_direction": int(i % 2),
                    "actual_return": 0.01,
                    "predicted_direction": int(i % 2),
                    "predicted_probability": 0.8 if i % 2 else 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_post_hoc_aggregate_matrix_records_coverage_audit():
    candidates = [
        aggregate_runner.Candidate("model_a", "no_news", 6, 1, "cls"),
        aggregate_runner.Candidate("model_b", "with_news_precomputed_pca", 9, 1, "reg"),
    ]

    base = aggregate_runner.build_candidate_matrix(_aggregate_toy_predictions(), "toy_top2", candidates)
    audit = base.attrs["audit"]

    assert audit["selection_protocol"] == "post_hoc_fixed_stream_set"
    assert audit["candidate_count"] == 2
    assert audit["raw_calendar_rows"] == 4
    assert audit["complete_calendar_rows"] == 3
    assert audit["dropped_incomplete_calendar_rows"] == 1
    assert audit["first_target_month"] == "2021-03-01"
    assert audit["last_target_month"] == "2021-05-01"
    assert audit["candidate_ids"] == [candidate.cid for candidate in candidates]


def test_post_hoc_aggregate_audit_files_are_written(tmp_path: Path):
    audit = {
        "ensemble": "toy_top2",
        "selection_protocol": "post_hoc_fixed_stream_set",
        "candidate_count": 2,
        "raw_calendar_rows": 4,
        "complete_calendar_rows": 3,
        "dropped_incomplete_calendar_rows": 1,
        "first_target_month": "2021-03-01",
        "last_target_month": "2021-05-01",
        "candidate_ids": ["a", "b"],
        "candidate_audit": [
            {"ensemble": "toy_top2", "candidate_id": "a", "rows": 4},
            {"ensemble": "toy_top2", "candidate_id": "b", "rows": 3},
        ],
    }

    aggregate_runner.write_aggregate_audit(tmp_path, [audit])

    aggregate_audit = pd.read_csv(tmp_path / "aggregate_audit.csv")
    candidate_audit = pd.read_csv(tmp_path / "aggregate_candidate_audit.csv")
    assert aggregate_audit.loc[0, "selection_protocol"] == "post_hoc_fixed_stream_set"
    assert int(aggregate_audit.loc[0, "dropped_incomplete_calendar_rows"]) == 1
    assert list(candidate_audit["candidate_id"]) == ["a", "b"]
