# Audit Report

Date: 2026-07-09

## Scope

This result snapshot records graph-temporal corn spike experiments only. The
formal code landing for reusable methods is under:

- `corn_forecast/operator/layer`
- `corn_forecast/operator/model/graph`

The result directory does not contain checkpoint files, screen logs, temporary
probe output, or remote launch scripts.

## Result Tables

| file | rows | role |
| --- | ---: | --- |
| `strict_leaderboard.csv` | 6644 | Full strict comparison leaderboard. |
| `healthy_leaderboard.csv` | 2164 | Rows passing health filters. |
| `healthy_robust30_leaderboard.csv` | 2164 | Healthy rows with robust30 eligibility. |
| `excluded_high_scores.csv` | 377 | High metric rows excluded by audit criteria. |

## Audit Rules

Excluded high scores are rows from the strict leaderboard where at least one
metric is high (`BalancedAcc`, `AUC`, or `AP` >= 0.60) but the row fails the
healthy or robust30 filters. Reasons are recorded per row and include:

- `pred_rows < 30`
- `R2_health < -0.1`
- `constant prediction`
- `agent_pass != true`
- `health_flag != true`
- `robust30_flag != true`

## Architecture Boundary

Layer code contains graph construction and intermediate transformations only.
Complete model code is split into one file per graph-temporal model. No model
registry, pool file, or generic temporal-attention file was added under
`operator/model/graph`.
