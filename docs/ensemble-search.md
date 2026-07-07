# Leakage-Aware Ensemble Search

`scripts/search_prediction_ensembles.py` searches ensemble methods over already
completed rolling prediction streams such as `all_rolling_predictions.csv`.

The formal results are prequential/walk-forward: for each target month, model
ranking, model selection, weights, local competence, stacking fits, and rolling
thresholds use only earlier target months. Rows marked `diagnostic_oracle` are
leaky upper-bound diagnostics and must not be reported as final backtest
performance.

## Method Families

- `simple`: static mean/median soft probabilities and static hard votes across
  all available prediction streams.
- `threshold`: rolling threshold wrappers around static ensemble scores; each
  test month's threshold is chosen only from earlier target months.
- `topk`: rolling top-k selection by prior balanced accuracy, AUC, AP, or Brier
  score, with hard or soft aggregation.
- `weighted`: rolling top-k with historical score-based weights.
- `online`: online weighted-majority style updating after each revealed label.
- `dynamic`: local dynamic ensemble selection using nearest historical
  prediction vectors and local model competence.
- `diverse`: diversity-aware greedy ensemble selection that balances historical
  competence with pairwise hard-vote disagreement.
- `stacking`: rolling logistic stacking on historical base probabilities and
  hard votes.
- `exhaustive`: rolling exhaustive hard-vote subset search over a small
  historically ranked candidate pool.
- `forward`: rolling forward ensemble selection over soft probabilities.
- `oracle`: final-period top-k diagnostics; these rows intentionally leak test
  labels and are excluded from `valid_walk_forward_leaderboard.csv`.

## Example

```bash
python scripts/search_prediction_ensembles.py \
  --predictions /path/to/all_rolling_predictions.csv \
  --output-dir experiments/ensemble_search \
  --preset fast \
  --families all \
  --min-history 12 \
  --min-candidate-coverage 0.90
```

Useful outputs:

- `valid_walk_forward_leaderboard.csv`: formal no-leakage leaderboard.
- `combined_all_summary_with_oracle.csv`: when multiple runs are merged
  externally, keep oracle rows separate from formal rows.
- `h*_best_valid_predictions.csv`: predictions for the best valid method per
  horizon.
- `candidate_inventory.csv`: candidate coverage and metadata.
- `ENSEMBLE_SEARCH_REPORT.md`: compact markdown report.

## Corn Spike Long-Lookback Check

On the 57-model corn spike prediction pool from the long-lookback rolling run,
the searched valid ensemble methods did not reach 0.90 balanced accuracy:

- Horizon 1 best valid method:
  `diverse_greedy_topn20_k6_brier_soft_lam0.1_w18_fixed`, balanced accuracy
  0.6410, AUC 0.6141, AP 0.6215.
- Horizon 2 best valid method: `dynamic_local_topn20_nn10_topm3_hard`,
  balanced accuracy 0.5838, AUC 0.6246, AP 0.6545.
- Diagnostic oracle top-k upper bounds were 0.8077 for horizon 1 and 0.7443 for
  horizon 2, so the current base prediction pool does not show evidence that a
  non-leaky combination can plausibly reach 0.90 balanced accuracy.

The earlier hand-selected `top6_h1` and `top6_h2` hard-vote ensembles are useful
diagnostics and framework checks, but because those candidate sets were chosen
after seeing completed rolling results, they should be described as fixed
post-hoc aggregates rather than a fully automatic no-leakage model-selection
procedure.
