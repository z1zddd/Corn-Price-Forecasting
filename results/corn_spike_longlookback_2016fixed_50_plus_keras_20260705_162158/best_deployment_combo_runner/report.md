# Experiment Report

Model: `corn_h2_forward_replacement_ba_hard_vote`

## Comparison

| model                                    | group                    |   DirAcc |   BalancedAcc |    AUC |     AP |   Sharpe |   Sortino |   Calmar |   AnnRet |   ProfitFactor |   WinRate |   MaxDD |   Precision |   Recall |     F1 |    MCC |   Specificity |    NPV |   Brier |   LogLoss |   Expectancy |   AvgWin |   AvgLoss | pred_constant_flag   | R2_health   | DirAcc_CI        | Sharpe_CI        |   n_bootstrap |   ci_level | selection_protocol                | selection_mode      | ranking_seed   | forward_tie_breaker   | aggregator       |   threshold |   selected_count |   unique_candidate_count |   expected_BalancedAcc |   expected_AUC |   expected_AP |   expected_DirAcc |
|:-----------------------------------------|:-------------------------|---------:|--------------:|-------:|-------:|---------:|----------:|---------:|---------:|---------------:|----------:|--------:|------------:|---------:|-------:|-------:|--------------:|-------:|--------:|----------:|-------------:|---------:|----------:|:---------------------|:------------|:-----------------|:-----------------|--------------:|-----------:|:----------------------------------|:--------------------|:---------------|:----------------------|:-----------------|------------:|-----------------:|-------------------------:|-----------------------:|---------------:|--------------:|------------------:|
| corn_h2_forward_replacement_ba_hard_vote | best_deployment_ensemble |   0.9342 |        0.9432 | 0.9084 | 0.955  |   3.3759 |    2.1531 |   2.5951 |   0.4119 |        11.1749 |    0.9342 | -0.1587 |       1     |   0.8864 | 0.9398 | 0.8755 |        1      | 0.8649 |  0.1784 |    0.5423 |       0.0343 |   0.0403 |    0.0513 | False                |             | [0.9342, 0.9342] | [3.3759, 3.3759] |             0 |       0.95 | full_history_deployment_discovery | forward_replacement | ba             | balanced              | hard_vote_strict |         0.5 |               34 |                       23 |                 0.9432 |         0.9084 |        0.955  |            0.9342 |
| corn_h1_forward_replacement_ap_hard_vote | best_deployment_ensemble |   0.9359 |        0.9359 | 0.9293 | 0.9072 |   3.2772 |    7.7545 |   5.2431 |   0.2892 |        14.6545 |    0.9359 | -0.0552 |       0.925 |   0.9487 | 0.9367 | 0.8721 |        0.9231 | 0.9474 |  0.1782 |    0.5439 |       0.0241 |   0.0276 |    0.0275 | False                |             | [0.9359, 0.9359] | [3.2772, 3.2772] |             0 |       0.95 | full_history_deployment_discovery | forward_replacement | ap             | ba_only               | hard_vote_strict |         0.5 |               32 |                       23 |                 0.9359 |         0.9293 |        0.9072 |            0.9359 |

## Verdict

Status: `signal`

Backtest results are research evidence and do not promise live trading profit.
