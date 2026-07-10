# Corn Spike Graph Temporal Results - 2026-07-09

This directory stores audited result artifacts for the corn spike graph-temporal
operator/model landing task. It intentionally contains only result tables,
configuration records, leaderboards, and audit notes.

Files:

- `experiment_manifest.yaml`: data, model, layer, and filtering metadata.
- `strict_leaderboard.csv`: strict merged graph-temporal leaderboard.
- `healthy_leaderboard.csv`: rows passing the healthy filter.
- `healthy_robust30_leaderboard.csv`: healthy rows with at least 30 predictions.
- `excluded_high_scores.csv`: high-scoring rows excluded by audit filters, with reasons.
- `audit_report.md`: audit boundary and acceptance notes.

No checkpoint, screen log, temporary smoke output, or remote training script is
stored here.
