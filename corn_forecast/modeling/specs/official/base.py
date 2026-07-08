"""Shared declarations for the official 57-model pool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


Factory = Callable[[int], object]


@dataclass(frozen=True)
class OfficialPoolSpec:
    """One official or package-native model-pool entry."""

    name: str
    family: str
    package: str
    classifier_factory: Factory | None
    regressor_factory: Factory | None
    classifier_loss: str
    regressor_loss: str
    input_kind: str
    source_kind: str


AEON_MODEL_NAMES = [
    "aeon_catch22",
    "aeon_deep_fcn",
    "aeon_deep_inceptiontime",
    "aeon_deep_mlp",
    "aeon_deep_timecnn",
    "aeon_knn_dtw",
    "aeon_knn_euclidean",
    "aeon_minirocket",
    "aeon_multirocket",
    "aeon_multirocket_hydra",
    "aeon_rdst",
    "aeon_rise",
    "aeon_summary",
    "aeon_tsf",
]


KERAS_SEQUENCE_MODEL_NAMES = [
    "keras_bilstm_u16",
    "keras_gru_u16",
    "keras_lstm_stack2_u32",
    "keras_lstm_u16",
    "keras_lstm_u32",
    "keras_tcn_filters16_k2_d1",
    "keras_tcn_filters8_k2_d1",
]


TABULAR_MODEL_NAMES = [
    "ada_boost_tree",
    "bagging_extra_tree",
    "bagging_logistic",
    "bagging_tree",
    "catboost_depthwise",
    "decision_tree_entropy",
    "decision_tree_gini",
    "extra_tree_entropy",
    "extra_tree_gini",
    "gaussian_nb",
    "gaussian_process_rbf",
    "gradient_boosting",
    "hist_gradient_boosting",
    "hist_gradient_boosting_l2",
    "knn_3_uniform",
    "knn_5_distance",
    "knn_9_distance",
    "lda_shrinkage",
    "lightgbm_dart",
    "lightgbm_gbdt",
    "lightgbm_goss",
    "logistic_l1_liblinear",
    "mlp_small_relu",
    "mlp_small_tanh",
    "nearest_centroid",
    "perceptron",
    "random_forest_100",
    "random_forest_balanced",
    "random_forest_shallow",
    "sgd_log_loss",
    "sgd_modified_huber",
    "svc_poly2",
    "svc_rbf",
    "svc_sigmoid",
    "xgboost_dart",
    "xgboost_gbtree",
]


OFFICIAL_57_MODEL_NAMES = [
    "ada_boost_tree",
    "aeon_catch22",
    "aeon_deep_fcn",
    "aeon_deep_inceptiontime",
    "aeon_deep_mlp",
    "aeon_deep_timecnn",
    "aeon_knn_dtw",
    "aeon_knn_euclidean",
    "aeon_minirocket",
    "aeon_multirocket",
    "aeon_multirocket_hydra",
    "aeon_rdst",
    "aeon_rise",
    "aeon_summary",
    "aeon_tsf",
    "bagging_extra_tree",
    "bagging_logistic",
    "bagging_tree",
    "catboost_depthwise",
    "decision_tree_entropy",
    "decision_tree_gini",
    "extra_tree_entropy",
    "extra_tree_gini",
    "gaussian_nb",
    "gaussian_process_rbf",
    "gradient_boosting",
    "hist_gradient_boosting",
    "hist_gradient_boosting_l2",
    "keras_bilstm_u16",
    "keras_gru_u16",
    "keras_lstm_stack2_u32",
    "keras_lstm_u16",
    "keras_lstm_u32",
    "keras_tcn_filters16_k2_d1",
    "keras_tcn_filters8_k2_d1",
    "knn_3_uniform",
    "knn_5_distance",
    "knn_9_distance",
    "lda_shrinkage",
    "lightgbm_dart",
    "lightgbm_gbdt",
    "lightgbm_goss",
    "logistic_l1_liblinear",
    "mlp_small_relu",
    "mlp_small_tanh",
    "nearest_centroid",
    "perceptron",
    "random_forest_100",
    "random_forest_balanced",
    "random_forest_shallow",
    "sgd_log_loss",
    "sgd_modified_huber",
    "svc_poly2",
    "svc_rbf",
    "svc_sigmoid",
    "xgboost_dart",
    "xgboost_gbtree",
]
