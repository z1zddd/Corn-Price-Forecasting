# all_rolling_predictions.csv Schema

`all_rolling_predictions.csv` is a prediction-library material. It contains
completed rolling prediction streams that can be consumed by deployment
aggregation operators.

Required fields:

| field | meaning |
| --- | --- |
| `model` | Base model or prediction stream name. |
| `feature_set` | Feature material variant, for example `no_news` or `with_news_precomputed_pca`. |
| `lookback_months` | Lookback window length used for the stream. |
| `horizon_months` | Forecast horizon in monthly periods. |
| `head` | Prediction head, commonly `cls` or `reg`. |
| `anchor_month` | Information month when the prediction is anchored. |
| `target_month` | Forward month being predicted. |
| `actual_direction` | Realized binary direction label, normally `0` or `1`. |
| `actual_return` | Realized forward return for the target month. |
| `predicted_direction` | Binary direction emitted by the prediction stream. |
| `predicted_probability` | Positive-class score or probability emitted by the prediction stream. |

Notes:

- This file is not a raw modeling table.
- Deployment vote aggregation consumes these streams after they have already
  been generated.
- Large generated prediction libraries should not be committed by default.
