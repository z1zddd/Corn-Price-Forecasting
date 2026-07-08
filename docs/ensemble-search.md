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
- `expweight`: exponential loss weighting over historically ranked prediction
  streams, following the online expert-advice/Hedge family.
- `online`: online weighted-majority style updating after each revealed label.
- `dynamic`: local dynamic ensemble selection using nearest historical
  prediction vectors and local model competence.
- `diverse`: diversity-aware greedy ensemble selection that balances historical
  competence with pairwise hard-vote disagreement.
- `blend`: rolling convex weight grid search over the historically strongest
  streams, a lightweight blending/Super Learner-style check.
- `stacking`: rolling logistic stacking on historical base probabilities and
  hard votes.
- `metaml`: rolling sklearn meta-learners trained only on historical base
  prediction features, including small tree ensembles, gradient boosting, SVM,
  k-nearest neighbors, and Naive Bayes.
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
  `metaml_knn5_distance_topn10_ba_w9999_fixed`, balanced accuracy 0.6795, AUC
  0.6114, AP 0.5694.
- Horizon 2 best valid method:
  `blend_grid_top4_ba_soft_step0.25_w18_rolling_ba`, balanced accuracy 0.6065,
  AUC 0.5483, AP 0.6238.
- Diagnostic oracle top-k upper bounds were 0.8077 for horizon 1 and 0.7443 for
  horizon 2, so the current base prediction pool does not show evidence that a
  non-leaky combination can plausibly reach 0.90 balanced accuracy.

The earlier hand-selected `top6_h1` and `top6_h2` hard-vote ensembles are useful
diagnostics and framework checks, but because those candidate sets were chosen
after seeing completed rolling results, they should be described as fixed
post-hoc aggregates rather than a fully automatic no-leakage model-selection
procedure.

## Post-Hoc Aggregate Audit

`scripts/run_best_aggregate_from_predictions.py` rebuilds the earlier fixed
`top6_h1` and `top6_h2` hard-vote checks inside the framework. These rows should
be read as signal diagnostics, not as formal automatic model-selection results:

- The base model predictions are still rolling out-of-sample predictions.
- The fixed top6 stream sets were selected after inspecting the completed
  rolling result pool, so the stream-selection step is post-hoc.
- The script writes `selection_protocol=post_hoc_fixed_stream_set` into the
  comparison table and report config.
- `aggregate_audit.csv` records the aggregate candidate count, evaluation
  start/end, and any incomplete early calendar rows dropped before scoring.
- `aggregate_candidate_audit.csv` records each member stream and its own target
  month coverage.

For the corn long-lookback pool, the fixed hard-vote checks were:

| aggregate | role | balanced accuracy | evaluated target months |
| --- | --- | ---: | --- |
| `top6_h1_hard_vote` | post-hoc signal diagnostic | 0.7657 | 2020-07 to 2026-06 |
| `top6_h2_hard_vote` | post-hoc signal diagnostic | 0.7122 | 2020-09 to 2026-06 |

They are useful for understanding whether strong streams can reinforce each
other, but they should be reported separately from
`valid_walk_forward_leaderboard.csv`, where every month chooses/ranks/weights
models using only earlier target months.

## Deployment Selection Search

When the task is to choose a fixed combination for future deployment, use
`scripts/search_deployment_combinations.py`. This script keeps a separate
protocol label:

`search_protocol=full_history_deployment_discovery`

This means the completed historical rolling prediction pool is used as the
model-selection data. That is the appropriate mode for choosing an上线 candidate
from all evidence available today, but it is not the same as a strict
walk-forward validation score.

Example:

```bash
python scripts/search_deployment_combinations.py \
  --predictions /path/to/all_rolling_predictions.csv \
  --output-dir experiments/deployment_combination_search \
  --horizons 1,2 \
  --max-pool-size 10 \
  --pool-sizes 6,8,10 \
  --rank-metrics ba,brier,auc,ap \
  --scopes all,news,nonews,cls,reg,lb6,lb9,lb12,best_per_model,best_per_family \
  --min-candidate-coverage 0.90 \
  --bootstrap 0
```

The search traverses every combination size from `k=1` to the pool size for
each requested pool, and evaluates multiple fixed deployment methods:

- hard strict voting, equivalent to positive only when vote share is `> 0.5`;
- hard tie-up voting, where ties become positive;
- soft probability mean with fixed threshold;
- soft probability mean with full-history chosen threshold;
- metric-weighted hard voting;
- metric-weighted soft averaging;
- metric-weighted soft averaging with full-history chosen threshold.

For larger model libraries where exhaustive subset search is too expensive, the
script also supports forward ensemble selection:

- `forward`: greedy no-replacement ensemble selection over a scoped candidate
  library.
- `forward_replacement`: greedy ensemble selection with replacement. Repeated
  selections become candidate weights, matching the common "ensemble selection
  from libraries" deployment pattern.
- `--forward-tie-breakers`: evaluates different greedy tie paths such as
  `balanced` and `ba_only`; this matters because monthly samples are small and
  many candidate additions can tie on balanced accuracy.
- `--threshold-grid-size`: controls the quantile grid for
  `*_best_threshold` aggregators.

The pool scopes make the search model-aware rather than only rank-aware:

- `all`: top streams regardless of metadata;
- `news` / `nonews`: with or without precomputed news PCA features;
- `cls` / `reg`: classification or regression heads;
- `lb6` / `lb9` / `lb12`: lookback-specific pools;
- `best_per_model`: avoid duplicate variants from the same base model;
- `best_per_family`: choose one strong stream per model family before
  combination.

On the corn long-lookback prediction pool, the max-10 deployment discovery run
found stronger fixed combinations than the earlier hand-picked top6 checks:

| horizon | deployment discovery best | BA | AUC | AP | DirAcc | k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `all_top8_by_ba` + `hard_vote_strict` | 0.8462 | 0.8794 | 0.8395 | 0.8462 | 6 |
| 2 | `best_per_family_top10_by_ba` + `hard_weighted_ba` | 0.8423 | 0.8349 | 0.8876 | 0.8421 | 5 |

Current deployment candidates:

- Horizon 1: `mlp_small_relu|news|lb6|h1|cls`,
  `sgd_modified_huber|news|lb6|h1|cls`,
  `lightgbm_dart|news|lb12|h1|reg`,
  `keras_tcn_filters16_k2_d1|nonews|lb9|h1|reg`,
  `svc_sigmoid|nonews|lb9|h1|cls`,
  `aeon_deep_mlp|nonews|lb9|h1|reg`.
- Horizon 2: `aeon_knn_euclidean|news|lb6|h2|reg`,
  `aeon_deep_timecnn|news|lb12|h2|cls`,
  `mlp_small_relu|news|lb12|h2|cls`,
  `hist_gradient_boosting|nonews|lb6|h2|cls`,
  `svc_sigmoid|news|lb9|h2|cls`.

The max-10 run still did not reach 0.90 balanced accuracy. A deeper max-16 or
max-20 search can be run with the same script after the quick deployment
candidate is reviewed.

For deeper deployment discovery, the same script also supports a vectorized
combination engine:

```bash
python scripts/search_deployment_combinations.py \
  --predictions /path/to/all_rolling_predictions.csv \
  --output-dir experiments/deployment_combination_search_max20_hard \
  --horizons 1,2 \
  --max-pool-size 20 \
  --pool-sizes 10,12,15,20 \
  --rank-metrics ba \
  --scopes all,news,nonews,cls,reg,lb6,lb9,lb12,best_per_model,best_per_family \
  --aggregators hard_vote_strict,hard_vote_tie_up,hard_weighted \
  --min-candidate-coverage 0.90 \
  --bootstrap 0 \
  --engine vectorized \
  --batch-size 50000
```

On the same corn long-lookback prediction pool, the vectorized max-20
hard-vote/weighted search improved the deployment discovery candidates but
still stayed below the 0.90 balanced-accuracy target:

| horizon | deployment discovery best | BA | AUC | AP | DirAcc | k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `all_top20_by_ba` + `hard_vote_strict` | 0.8846 | 0.9004 | 0.8770 | 0.8846 | 12 |
| 2 | `best_per_model_top20_by_ba` + `hard_vote_strict` | 0.8807 | 0.8874 | 0.8950 | 0.8816 | 10 |

Current max-20 deployment candidates:

- Horizon 1: `mlp_small_relu|news|lb6|h1|cls`,
  `lightgbm_dart|news|lb12|h1|reg`,
  `keras_lstm_u16|nonews|lb12|h1|cls`,
  `keras_tcn_filters16_k2_d1|nonews|lb9|h1|reg`,
  `svc_sigmoid|nonews|lb9|h1|cls`,
  `keras_gru_u16|news|lb12|h1|reg`,
  `extra_tree_entropy|news|lb9|h1|cls`,
  `keras_tcn_filters16_k2_d1|news|lb9|h1|cls`,
  `keras_tcn_filters8_k2_d1|nonews|lb9|h1|reg`,
  `gaussian_nb|nonews|lb6|h1|cls`,
  `aeon_deep_fcn|news|lb12|h1|reg`,
  `keras_bilstm_u16|news|lb12|h1|cls`.
- Horizon 2: `aeon_knn_euclidean|news|lb6|h2|reg`,
  `aeon_deep_timecnn|news|lb12|h2|cls`,
  `aeon_rise|nonews|lb6|h2|cls`,
  `mlp_small_relu|news|lb12|h2|cls`,
  `hist_gradient_boosting|nonews|lb6|h2|cls`,
  `aeon_deep_fcn|news|lb12|h2|reg`,
  `xgboost_dart|news|lb12|h2|cls`,
  `aeon_deep_mlp|news|lb12|h2|reg`,
  `svc_sigmoid|news|lb9|h2|cls`,
  `keras_lstm_stack2_u32|nonews|lb9|h2|cls`.

A targeted forward-selection deployment discovery run over the full 684-stream
library reached the 0.90 balanced-accuracy target for both horizons:

```bash
python scripts/search_deployment_combinations.py \
  --predictions /path/to/all_rolling_predictions.csv \
  --output-dir experiments/deployment_combo_forward_targeted \
  --horizons 1,2 \
  --rank-metrics ba \
  --scopes all \
  --aggregators hard_vote_strict,soft_mean_best_threshold \
  --search-modes forward,forward_replacement \
  --forward-max-k 80 \
  --forward-candidate-limit 0 \
  --forward-tie-breakers all \
  --threshold-grid-size 81 \
  --min-candidate-coverage 0.90 \
  --bootstrap 0
```

| horizon | deployment discovery best | mode | BA | AUC | AP | DirAcc | k |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `all_full684_by_ba` + `hard_vote_strict` | `forward_replacement`, `ba_only` tie path | 0.9231 | 0.9195 | 0.8931 | 0.9231 | 76 |
| 2 | `all_full684_by_ba` + `hard_vote_strict` | `forward_replacement`, `balanced` tie path | 0.9432 | 0.9084 | 0.9550 | 0.9342 | 34 |

Expanding only the ranking seed while keeping the same full-library hard-vote
forward-replacement search improved horizon 1 further:

```bash
python scripts/search_deployment_combinations.py \
  --predictions /path/to/all_rolling_predictions.csv \
  --output-dir experiments/deployment_combo_forward_rankseeds \
  --horizons 1,2 \
  --rank-metrics ba,auc,ap,brier \
  --scopes all \
  --aggregators hard_vote_strict \
  --search-modes forward,forward_replacement \
  --forward-max-k 100 \
  --forward-candidate-limit 0 \
  --forward-tie-breakers all \
  --threshold-grid-size 81 \
  --min-candidate-coverage 0.90 \
  --bootstrap 0
```

| horizon | deployment discovery best | ranking seed | mode | BA | AUC | AP | DirAcc | k |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `all_full684_by_ap` + `hard_vote_strict` | AP | `forward_replacement`, `ba_only` tie path | 0.9359 | 0.9293 | 0.9072 | 0.9359 | 32 |
| 2 | `all_full684_by_ba` + `hard_vote_strict` | BA | `forward_replacement`, `balanced` tie path | 0.9432 | 0.9084 | 0.9550 | 0.9342 | 34 |

These are still `full_history_deployment_discovery` rows: they are appropriate
for selecting an上线 candidate from the completed historical rolling prediction
library, but they should be reported separately from strict walk-forward
automatic model-selection scores.
