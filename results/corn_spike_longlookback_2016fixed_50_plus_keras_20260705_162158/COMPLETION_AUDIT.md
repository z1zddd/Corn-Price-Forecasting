# Completion Audit

- PASS: summary_rows: 1368/1368
- PASS: valid_rows: 1368/1368
- PASS: model_count: summary=57 valid=57
- PASS: feature_sets: ['no_news', 'with_news_precomputed_pca']
- PASS: lookbacks: [6, 9, 12]
- PASS: horizons: [1, 2]
- PASS: heads: ['cls', 'reg']
- PASS: combo_uniqueness: unique=1368 max=1
- PASS: fold_identity_unique: duplicates=0 folds=50616
- PASS: test_rows_one: bad=0
- PASS: val_rows_12: bad=0
- PASS: train_rows_min24: min=24
- PASS: origin_counts_by_lb_h: {(6, 1): 78, (6, 2): 76, (9, 1): 75, (9, 2): 73, (12, 1): 72, (12, 2): 70}
- PASS: prediction_identity_unique: duplicates=0 predictions=101232
- PASS: prediction_counts_by_lb_h_head: all match expected origins
- PASS: fold_no_leakage_date_order: bad=0
- PASS: prediction_target_after_anchor: bad=0
- PASS: r2_status_rules: abnormal=106 likely_abnormal=6 ok=1256
- PASS: manifest_leakage_controls_sample: /root/corn_spike_sota/outputs/longlookback_2016fixed_50_plus_keras_20260705_162158/parts/mlp_small_tanh/20260705_162159/manifest.json

## Top 10
| model                     | feature_set               |   horizon_months | head   |   lookback_months |   balanced_accuracy |      auc |   average_precision |   reg_price_r2 | r2_status   |
|:--------------------------|:--------------------------|-----------------:|:-------|------------------:|--------------------:|---------:|--------------------:|---------------:|:------------|
| aeon_knn_euclidean        | with_news_precomputed_pca |                2 | reg    |                 6 |            0.661932 | 0.53125  |            0.629213 |       0.706061 | ok          |
| mlp_small_relu            | with_news_precomputed_pca |                1 | cls    |                 6 |            0.653846 | 0.627876 |            0.665741 |       0.529097 | ok          |
| aeon_deep_timecnn         | with_news_precomputed_pca |                2 | cls    |                12 |            0.647204 | 0.494243 |            0.570336 |       0.685808 | ok          |
| knn_5_distance            | with_news_precomputed_pca |                2 | reg    |                 6 |            0.642045 | 0.534091 |            0.632108 |       0.706161 | ok          |
| sgd_modified_huber        | with_news_precomputed_pca |                1 | cls    |                 6 |            0.641026 | 0.583498 |            0.545701 |       0.748123 | ok          |
| lightgbm_dart             | with_news_precomputed_pca |                1 | reg    |                12 |            0.637529 | 0.573427 |            0.526648 |       0.827701 | ok          |
| aeon_deep_timecnn         | no_news                   |                2 | cls    |                12 |            0.629112 | 0.490132 |            0.559754 |       0.687905 | ok          |
| keras_lstm_u16            | no_news                   |                1 | cls    |                12 |            0.628205 | 0.494949 |            0.451994 |       0.819362 | ok          |
| knn_3_uniform             | with_news_precomputed_pca |                2 | cls    |                12 |            0.626645 | 0.515625 |            0.559565 |       0.524838 | ok          |
| keras_tcn_filters16_k2_d1 | no_news                   |                1 | reg    |                 9 |            0.626068 | 0.60114  |            0.593998 |       0.568749 | ok          |
