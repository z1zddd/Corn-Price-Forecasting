# Metrics

The framework computes classification, trading, calibration, and health metrics for out-of-sample predictions.

## Classification Metrics

- `DirAcc`: directional accuracy.
- `BalancedAcc`: balanced accuracy across both classes.
- `AUC`: ROC-AUC with a single-class fallback.
- `AP`: average precision with a single-class fallback.
- `Precision`, `Recall`, `F1`, `MCC`, `Specificity`, `NPV`.

## Trading Metrics

Predicted label `1` is treated as long and predicted label `0` is treated as short. Strategy return is:

```text
actual_return if prediction == 1 else -actual_return
```

Reported trading metrics:

- `Sharpe`: annualized Sharpe ratio.
- `Sortino`: annualized Sortino ratio.
- `Calmar`: annualized return divided by absolute maximum drawdown.
- `AnnRet`: annualized mean strategy return.
- `ProfitFactor`: gross profit divided by gross loss.
- `WinRate`: fraction of positive strategy returns.
- `MaxDD`: maximum drawdown.
- `Expectancy`: average strategy return.
- `AvgWin` and `AvgLoss`.

## Calibration And Health

- `Brier`: mean squared probability error.
- `LogLoss`: clipped binary log loss.
- `pred_constant_flag`: true when predicted labels are constant.
- `R2_health`: regression-head health against a naive mean-return predictor when raw regression predictions exist.

`pred_constant_flag` is important because a high score from a constant predictor is usually not a tradable signal.

`R2_health` is diagnostic only. It should not override out-of-sample direction, trading, and confidence-interval checks.

## Confidence Intervals

Bootstrap confidence intervals are computed for:

- `DirAcc_CI`
- `Sharpe_CI`

The number of bootstrap samples and CI level are controlled by:

```yaml
evaluation:
  ci_level: 0.95
  ci_bootstrap_samples: 200
```

For small datasets, confidence intervals can be wide. The verdict writer treats a wide interval as weak evidence even when point metrics look good.

## Rolling Diagnostics

Each model output includes `rolling_metrics.csv`, `rolling_dir_acc.png`, and `rolling_sharpe.png`.

- `rolling_12_dir_acc`: rolling directional accuracy over the last 12 predictions.
- `rolling_12_sharpe`: rolling annualized Sharpe over the last 12 strategy returns.
