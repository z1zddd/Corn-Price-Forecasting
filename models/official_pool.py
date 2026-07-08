"""Official model-pool adapters for rolling commodity benchmarks."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from models.sklearn_models import flatten_windows


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


class ConstantClassifier:
    """Fallback classifier for one-class rolling folds."""

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def fit(self, x, y):
        return self

    def predict_proba(self, x) -> np.ndarray:
        n = len(x)
        p = np.full(n, self.positive_probability, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class ConstantRegressor:
    """Fallback regressor for constant return folds."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, x, y):
        return self

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


class OfficialPoolAdapter:
    """Framework adapter around official sklearn, aeon, and Keras estimators."""

    def __init__(self, spec: OfficialPoolSpec, *, seed: int = 42) -> None:
        self.spec = spec
        self.seed = int(seed)
        self.classifier_ = None
        self.regressor_ = None
        self.model_family = spec.family
        self.package = spec.package
        self.input_kind = spec.input_kind
        self.source_kind = spec.source_kind

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        return self.fit_with_targets(x_train, y_train, np.asarray(y_train, dtype=float), x_val, y_val, None)

    def fit_with_targets(self, x_train, y_class_train, y_return_train, x_val=None, y_class_val=None, y_return_val=None):
        x = format_input(x_train, self.spec.input_kind)
        y_class = np.asarray(y_class_train, dtype=int).reshape(-1)
        y_return = np.asarray(y_return_train, dtype=float).reshape(-1)
        if self.spec.classifier_factory is not None:
            self.classifier_ = fit_classifier(self.spec, x, y_class, self.seed)
        if self.spec.regressor_factory is not None:
            self.regressor_ = fit_regressor(self.spec, x, y_return, self.seed)
        return self

    def predict_proba(self, x_test) -> np.ndarray:
        x = format_input(x_test, self.spec.input_kind)
        if self.classifier_ is not None:
            return positive_probability(self.classifier_, x)
        raw = self.predict_regression(x_test)
        if raw is None:
            raise RuntimeError(f"{self.spec.name} has neither fitted classifier nor fitted regressor")
        scale = float(np.nanstd(raw))
        scale = scale if scale > 1e-12 else 1.0
        return sigmoid(raw / scale)

    def predict_regression(self, x_test) -> np.ndarray | None:
        if self.regressor_ is None:
            return None
        x = format_input(x_test, self.spec.input_kind)
        return np.asarray(self.regressor_.predict(x), dtype=float).reshape(-1)

    def predict(self, x_test) -> np.ndarray:
        return (self.predict_proba(x_test) > 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "spec": self.spec,
                "seed": self.seed,
                "classifier": self.classifier_,
                "regressor": self.regressor_,
            },
            path,
        )


def create_official_pool_model(model_name: str, params: dict | None = None) -> OfficialPoolAdapter:
    """Create a framework adapter for one model in the official 57-model pool."""

    params = dict(params or {})
    pool_model = str(params.pop("pool_model", model_name))
    seed = int(params.pop("seed", params.pop("random_state", 42)))
    if params:
        raise ValueError(f"Unsupported official_pool params for {pool_model}: {sorted(params)}")
    specs = {spec.name: spec for spec in build_official_model_pool()}
    if pool_model not in specs:
        raise ValueError(f"Unknown official_pool model: {pool_model}")
    return OfficialPoolAdapter(specs[pool_model], seed=seed)


def expand_model_pool(name: str) -> list[dict]:
    """Expand a named model pool into framework model configs."""

    if name != "official_57":
        raise ValueError(f"Unknown model pool: {name}")
    return [{"name": model_name, "type": "official_pool", "enabled": True} for model_name in OFFICIAL_57_MODEL_NAMES]


def build_official_model_pool() -> list[OfficialPoolSpec]:
    """Return the 57-model pool used by the long-lookback corn benchmark."""

    specs = {spec.name: spec for spec in [*build_tabular_specs(), *build_aeon_specs(), *build_keras_sequence_specs()]}
    missing = [name for name in OFFICIAL_57_MODEL_NAMES if name not in specs]
    if missing:
        raise AssertionError(f"Official model pool is missing entries: {missing}")
    return [specs[name] for name in OFFICIAL_57_MODEL_NAMES]


def build_tabular_specs() -> list[OfficialPoolSpec]:
    """Build sklearn/LightGBM/XGBoost/CatBoost entries from package-native estimators."""

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (
        AdaBoostClassifier,
        AdaBoostRegressor,
        BaggingClassifier,
        BaggingRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.linear_model import (
        BayesianRidge,
        HuberRegressor,
        Lasso,
        LogisticRegression,
        PassiveAggressiveRegressor,
        Perceptron,
        Ridge,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

    def spec(name, family, package, clf, reg, clf_loss="native", reg_loss="squared_error"):
        return OfficialPoolSpec(name, family, package, clf, reg, clf_loss, reg_loss, "tabular_flat", "official_tabular_package")

    return [
        spec(
            "logistic_l1_liblinear",
            "linear",
            "sklearn",
            lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s, penalty="l1", solver="liblinear", C=0.5),
            lambda s: Lasso(alpha=0.01, max_iter=5000),
            "log_loss_l1",
            "lasso_l1",
        ),
        spec(
            "sgd_log_loss",
            "linear",
            "sklearn",
            lambda s: SGDClassifier(loss="log_loss", alpha=1e-3, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s),
            lambda s: SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3, max_iter=2000, tol=1e-4, random_state=s),
            "log_loss",
            "squared_error",
        ),
        spec(
            "sgd_modified_huber",
            "linear",
            "sklearn",
            lambda s: SGDClassifier(loss="modified_huber", alpha=1e-3, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s),
            lambda s: HuberRegressor(alpha=1e-3, max_iter=200),
            "modified_huber",
            "huber",
        ),
        spec(
            "perceptron",
            "linear",
            "sklearn",
            lambda s: Perceptron(class_weight="balanced", max_iter=2000, tol=1e-4, random_state=s),
            lambda s: PassiveAggressiveRegressor(max_iter=2000, tol=1e-4, random_state=s),
            "perceptron",
            "pa",
        ),
        spec("svc_rbf", "svm", "sklearn", lambda s: SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=1.0, gamma="scale", kernel="rbf"), "hinge_platt", "epsilon_insensitive"),
        spec("svc_poly2", "svm", "sklearn", lambda s: SVC(C=0.7, degree=2, gamma="scale", kernel="poly", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=0.7, degree=2, gamma="scale", kernel="poly"), "hinge_platt", "epsilon_insensitive"),
        spec("svc_sigmoid", "svm", "sklearn", lambda s: SVC(C=0.7, gamma="scale", kernel="sigmoid", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=0.7, gamma="scale", kernel="sigmoid"), "hinge_platt", "epsilon_insensitive"),
        spec("knn_3_uniform", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=3, weights="uniform"), lambda s: KNeighborsRegressor(n_neighbors=3, weights="uniform"), "vote", "neighbor_mean"),
        spec("knn_5_distance", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=5, weights="distance"), lambda s: KNeighborsRegressor(n_neighbors=5, weights="distance"), "vote_distance", "neighbor_distance"),
        spec("knn_9_distance", "neighbors", "sklearn", lambda s: KNeighborsClassifier(n_neighbors=9, weights="distance"), lambda s: KNeighborsRegressor(n_neighbors=9, weights="distance"), "vote_distance", "neighbor_distance"),
        spec("nearest_centroid", "neighbors", "sklearn", lambda s: NearestCentroid(), lambda s: Ridge(alpha=1.0), "centroid_distance", "ridge_l2"),
        spec("gaussian_nb", "bayes", "sklearn", lambda s: GaussianNB(), lambda s: BayesianRidge(), "gaussian_nb", "bayesian_ridge"),
        spec("lda_shrinkage", "discriminant", "sklearn", lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), lambda s: Ridge(alpha=1.0), "lda_shrinkage", "ridge_l2"),
        spec("decision_tree_gini", "tree", "sklearn", lambda s: DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: DecisionTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=4, random_state=s), "gini", "squared_error"),
        spec("decision_tree_entropy", "tree", "sklearn", lambda s: DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: DecisionTreeRegressor(criterion="absolute_error", max_depth=4, min_samples_leaf=4, random_state=s), "entropy", "absolute_error"),
        spec("extra_tree_gini", "tree", "sklearn", lambda s: ExtraTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: ExtraTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=4, random_state=s), "gini_randomized", "squared_error"),
        spec("extra_tree_entropy", "tree", "sklearn", lambda s: ExtraTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=s), lambda s: ExtraTreeRegressor(criterion="absolute_error", max_depth=4, min_samples_leaf=4, random_state=s), "entropy_randomized", "absolute_error"),
        spec("random_forest_100", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=s, n_jobs=1), "gini_bagging", "squared_error_bagging"),
        spec("random_forest_balanced", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=120, max_depth=None, min_samples_leaf=4, class_weight="balanced_subsample", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=120, max_depth=None, min_samples_leaf=4, random_state=s, n_jobs=1), "gini_balanced_bagging", "squared_error_bagging"),
        spec("random_forest_shallow", "forest", "sklearn", lambda s: RandomForestClassifier(n_estimators=120, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=s, n_jobs=1), lambda s: RandomForestRegressor(n_estimators=120, max_depth=3, min_samples_leaf=3, random_state=s, n_jobs=1), "gini_shallow", "squared_error_shallow"),
        spec("gradient_boosting", "boosting", "sklearn", lambda s: GradientBoostingClassifier(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=s), lambda s: GradientBoostingRegressor(n_estimators=80, learning_rate=0.04, max_depth=2, random_state=s), "deviance", "squared_error_boosting"),
        spec("hist_gradient_boosting", "boosting", "sklearn", lambda s: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=s), lambda s: HistGradientBoostingRegressor(max_iter=120, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=s), "log_loss_hist_gbdt", "squared_error_hist_gbdt"),
        spec("hist_gradient_boosting_l2", "boosting", "sklearn", lambda s: HistGradientBoostingClassifier(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=1.0, random_state=s), lambda s: HistGradientBoostingRegressor(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, l2_regularization=1.0, random_state=s), "log_loss_hist_l2", "squared_error_hist_l2"),
        spec("ada_boost_tree", "boosting", "sklearn", adaboost_tree_classifier_factory(AdaBoostClassifier, DecisionTreeClassifier), lambda s: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=1, random_state=s), n_estimators=80, learning_rate=0.05, random_state=s), "samme", "adaboost_square"),
        spec("bagging_tree", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=s), n_estimators=80, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=s), n_estimators=80, max_samples=0.8, random_state=s, n_jobs=1), "bagged_tree_vote", "bagged_tree_regression"),
        spec("bagging_extra_tree", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=ExtraTreeClassifier(max_depth=4, class_weight="balanced", random_state=s), n_estimators=100, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=ExtraTreeRegressor(max_depth=4, random_state=s), n_estimators=100, max_samples=0.8, random_state=s, n_jobs=1), "bagged_extra_tree_vote", "bagged_extra_tree_regression"),
        spec("bagging_logistic", "bagging", "sklearn", lambda s: BaggingClassifier(estimator=LogisticRegression(max_iter=1000, class_weight="balanced"), n_estimators=30, max_samples=0.8, random_state=s, n_jobs=1), lambda s: BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=30, max_samples=0.8, random_state=s, n_jobs=1), "bagged_log_loss", "bagged_ridge"),
        spec("mlp_small_relu", "neural_sklearn", "sklearn", lambda s: MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), lambda s: MLPRegressor(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), "log_loss_mlp", "squared_error_mlp"),
        spec("mlp_small_tanh", "neural_sklearn", "sklearn", lambda s: MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), lambda s: MLPRegressor(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=s), "log_loss_mlp", "squared_error_mlp"),
        spec("gaussian_process_rbf", "kernel", "sklearn", lambda s: GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), max_iter_predict=100, random_state=s), lambda s: GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0), alpha=1e-2, normalize_y=True, random_state=s), "laplace_log_marginal", "gp_regression"),
        spec("lightgbm_gbdt", "gbdt", "lightgbm", lightgbm_classifier("gbdt"), lightgbm_regressor("gbdt"), "binary_logloss", "l2"),
        spec("lightgbm_goss", "gbdt", "lightgbm", lightgbm_classifier("goss"), lightgbm_regressor("goss"), "binary_logloss_goss", "l2_goss"),
        spec("lightgbm_dart", "gbdt", "lightgbm", lightgbm_classifier("dart"), lightgbm_regressor("dart"), "binary_logloss_dart", "l2_dart"),
        spec("xgboost_gbtree", "gbdt", "xgboost", xgb_classifier("gbtree"), xgb_regressor("gbtree"), "logloss", "squared_error"),
        spec("xgboost_dart", "gbdt", "xgboost", xgb_classifier("dart"), xgb_regressor("dart"), "logloss_dart", "squared_error_dart"),
        spec("catboost_depthwise", "gbdt", "catboost", catboost_classifier("Depthwise"), catboost_regressor("Depthwise"), "logloss_depthwise", "rmse_depthwise"),
    ]


def build_aeon_specs() -> list[OfficialPoolSpec]:
    """Build aeon entries using aeon's official estimator classes."""

    aeon_kernels = 384
    aeon_estimators = 64
    deep_epochs = 12
    deep_batch_size = 16
    return [
        spec_pair("aeon_minirocket", "tsc_convolution", "MiniRocketClassifier", "MiniRocketRegressor", "aeon.classification.convolution_based", "aeon.regression.convolution_based", {"n_kernels": aeon_kernels, "n_jobs": 1}, {"n_kernels": aeon_kernels, "n_jobs": 1}, "minirocket_ridge", "minirocket_ridge", input_kind="aeon_collection_pad10"),
        spec_pair("aeon_multirocket", "tsc_convolution", "MultiRocketClassifier", "MultiRocketRegressor", "aeon.classification.convolution_based", "aeon.regression.convolution_based", {"n_kernels": aeon_kernels, "n_features_per_kernel": 4, "n_jobs": 1}, {"n_kernels": aeon_kernels, "n_features_per_kernel": 4, "n_jobs": 1}, "multirocket_ridge", "multirocket_ridge", input_kind="aeon_collection_pad10"),
        spec_pair("aeon_multirocket_hydra", "tsc_convolution", "MultiRocketHydraClassifier", "MultiRocketHydraRegressor", "aeon.classification.convolution_based", "aeon.regression.convolution_based", {"n_kernels": 8, "n_groups": 16, "n_jobs": 1}, {"n_kernels": 8, "n_groups": 16, "n_jobs": 1}, "multirocket_hydra_ridge", "multirocket_hydra_ridge", input_kind="aeon_collection_pad10"),
        spec_pair("aeon_tsf", "tsc_interval", "TimeSeriesForestClassifier", "TimeSeriesForestRegressor", "aeon.classification.interval_based", "aeon.regression.interval_based", {"n_estimators": aeon_estimators, "min_interval_length": 1, "n_jobs": 1}, {"n_estimators": aeon_estimators, "min_interval_length": 1, "n_jobs": 1}, "interval_forest", "interval_forest"),
        spec_pair("aeon_rise", "tsc_interval", "RandomIntervalSpectralEnsembleClassifier", "RandomIntervalSpectralEnsembleRegressor", "aeon.classification.interval_based", "aeon.regression.interval_based", {"n_estimators": max(16, aeon_estimators // 2), "min_interval_length": 1, "acf_lag": 1, "acf_min_values": 1, "n_jobs": 1}, {"n_estimators": max(16, aeon_estimators // 2), "min_interval_length": 1, "acf_lag": 1, "acf_min_values": 1, "n_jobs": 1}, "rise", "rise"),
        spec_pair("aeon_catch22", "tsc_feature", "Catch22Classifier", "Catch22Regressor", "aeon.classification.feature_based", "aeon.regression.feature_based", {"catch24": True, "replace_nans": True, "n_jobs": 1}, {"catch24": True, "replace_nans": True, "n_jobs": 1}, "catch22_random_forest", "catch22_random_forest"),
        spec_pair("aeon_summary", "tsc_feature", "SummaryClassifier", "SummaryRegressor", "aeon.classification.feature_based", "aeon.regression.feature_based", {"n_jobs": 1}, {"n_jobs": 1}, "summary_random_forest", "summary_random_forest"),
        spec_pair("aeon_rdst", "tsc_shapelet", "RDSTClassifier", "RDSTRegressor", "aeon.classification.shapelet_based", "aeon.regression.shapelet_based", {"max_shapelets": 256, "shapelet_lengths": np.array([2], dtype=np.int64), "n_jobs": 1}, {"max_shapelets": 256, "shapelet_lengths": np.array([2], dtype=np.int64), "n_jobs": 1}, "random_dilated_shapelet", "random_dilated_shapelet", input_kind="aeon_collection_pad10_float64"),
        spec_pair("aeon_knn_dtw", "tsc_distance", "KNeighborsTimeSeriesClassifier", "KNeighborsTimeSeriesRegressor", "aeon.classification.distance_based", "aeon.regression.distance_based", {"n_neighbors": 3, "distance": "dtw", "n_jobs": 1}, {"n_neighbors": 3, "distance": "dtw", "n_jobs": 1}, "dtw_vote", "dtw_mean"),
        spec_pair("aeon_knn_euclidean", "tsc_distance", "KNeighborsTimeSeriesClassifier", "KNeighborsTimeSeriesRegressor", "aeon.classification.distance_based", "aeon.regression.distance_based", {"n_neighbors": 5, "distance": "euclidean", "n_jobs": 1}, {"n_neighbors": 5, "distance": "euclidean", "n_jobs": 1}, "euclidean_vote", "euclidean_mean"),
        deep_pair("aeon_deep_mlp", "MLPClassifier", "MLPRegressor", deep_epochs, deep_batch_size, {"n_layers": 2, "n_units": 64}, {"n_layers": 2, "n_units": 64}),
        deep_pair("aeon_deep_fcn", "FCNClassifier", "FCNRegressor", deep_epochs, deep_batch_size, {"n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]}, {"n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]}),
        deep_pair("aeon_deep_inceptiontime", "InceptionTimeClassifier", "InceptionTimeRegressor", deep_epochs, deep_batch_size, {"n_classifiers": 1, "n_filters": 16, "kernel_size": 3, "depth": 3, "bottleneck_size": 8}, {"n_regressors": 1, "n_filters": 16, "kernel_size": 3, "depth": 3, "bottleneck_size": 8}),
        deep_pair("aeon_deep_timecnn", "TimeCNNClassifier", "TimeCNNRegressor", deep_epochs, deep_batch_size, {"kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16]}, {"kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16]}),
    ]


def build_keras_sequence_specs() -> list[OfficialPoolSpec]:
    """Build Keras recurrent/TCN variants from official TensorFlow/Keras layers."""

    epochs = 12
    batch_size = 16
    return [
        keras_sequence_pair("keras_lstm_u16", "lstm", {"units": 16}, epochs, batch_size),
        keras_sequence_pair("keras_lstm_u32", "lstm", {"units": 32}, epochs, batch_size),
        keras_sequence_pair("keras_lstm_stack2_u32", "lstm", {"units": [32, 16]}, epochs, batch_size),
        keras_sequence_pair("keras_gru_u16", "gru", {"units": 16}, epochs, batch_size),
        keras_sequence_pair("keras_bilstm_u16", "bilstm", {"units": 16}, epochs, batch_size),
        keras_sequence_pair("keras_tcn_filters8_k2_d1", "tcn", {"nb_filters": 8, "kernel_size": 2, "dilations": (1,)}, epochs, batch_size, package="keras-tcn"),
        keras_sequence_pair("keras_tcn_filters16_k2_d1", "tcn", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,)}, epochs, batch_size, package="keras-tcn"),
    ]


def adaboost_tree_classifier_factory(ada_cls, tree_cls) -> Factory:
    def factory(seed: int):
        params = {
            "estimator": tree_cls(max_depth=1, class_weight="balanced", random_state=seed),
            "n_estimators": 80,
            "learning_rate": 0.05,
            "random_state": seed,
        }
        if "algorithm" in inspect.signature(ada_cls).parameters:
            params["algorithm"] = "SAMME"
        return ada_cls(**params)

    return factory


def lightgbm_classifier(boosting_type: str) -> Factory:
    def factory(seed: int):
        from lightgbm import LGBMClassifier

        params = {
            "boosting_type": boosting_type,
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        if boosting_type == "goss":
            params.pop("subsample")
        return LGBMClassifier(**params)

    return factory


def lightgbm_regressor(boosting_type: str) -> Factory:
    def factory(seed: int):
        from lightgbm import LGBMRegressor

        params = {
            "boosting_type": boosting_type,
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        if boosting_type == "goss":
            params.pop("subsample")
        return LGBMRegressor(**params)

    return factory


def xgb_classifier(booster: str) -> Factory:
    def factory(seed: int):
        from xgboost import XGBClassifier

        return XGBClassifier(
            booster=booster,
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    return factory


def xgb_regressor(booster: str) -> Factory:
    def factory(seed: int):
        from xgboost import XGBRegressor

        return XGBRegressor(
            booster=booster,
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    return factory


def catboost_classifier(grow_policy: str) -> Factory:
    def factory(seed: int):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=100,
            learning_rate=0.035,
            depth=3,
            grow_policy=grow_policy,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return factory


def catboost_regressor(grow_policy: str) -> Factory:
    def factory(seed: int):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=100,
            learning_rate=0.035,
            depth=3,
            grow_policy=grow_policy,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return factory


def spec_pair(
    name: str,
    family: str,
    cls_name: str,
    reg_name: str,
    cls_module: str,
    reg_module: str,
    cls_kwargs: dict[str, object],
    reg_kwargs: dict[str, object],
    classifier_loss: str,
    regressor_loss: str,
    input_kind: str = "aeon_collection",
) -> OfficialPoolSpec:
    return OfficialPoolSpec(
        name=name,
        family=family,
        package="aeon",
        classifier_factory=aeon_factory(cls_module, cls_name, cls_kwargs),
        regressor_factory=aeon_factory(reg_module, reg_name, reg_kwargs),
        classifier_loss=classifier_loss,
        regressor_loss=regressor_loss,
        input_kind=input_kind,
        source_kind="official_aeon",
    )


def deep_pair(
    name: str,
    cls_name: str,
    reg_name: str,
    epochs: int,
    batch_size: int,
    cls_extra: dict[str, object],
    reg_extra: dict[str, object],
) -> OfficialPoolSpec:
    base = {
        "n_epochs": epochs,
        "batch_size": batch_size,
        "verbose": False,
        "save_best_model": False,
        "save_last_model": False,
        "save_init_model": False,
    }
    return spec_pair(
        name,
        "tsc_deep_learning",
        cls_name,
        reg_name,
        "aeon.classification.deep_learning",
        "aeon.regression.deep_learning",
        {**base, **cls_extra},
        {**base, **reg_extra},
        "deep_categorical_crossentropy",
        "deep_mean_squared_error",
    )


def keras_sequence_pair(
    name: str,
    architecture: str,
    params: dict[str, object],
    epochs: int,
    batch_size: int,
    package: str = "tensorflow.keras",
) -> OfficialPoolSpec:
    return OfficialPoolSpec(
        name=name,
        family="deep_sequence",
        package=package,
        classifier_factory=lambda seed: KerasSequenceClassifier(architecture, dict(params), epochs, batch_size, seed),
        regressor_factory=lambda seed: KerasSequenceRegressor(architecture, dict(params), epochs, batch_size, seed),
        classifier_loss="binary_crossentropy",
        regressor_loss="mean_squared_error",
        input_kind="keras_sequence",
        source_kind="official_keras_layers",
    )


def aeon_factory(module_name: str, class_name: str, kwargs: dict[str, object]) -> Factory:
    def factory(seed: int):
        if "deep_learning" in module_name:
            configure_tensorflow_runtime()
        module = importlib.import_module(module_name)
        estimator_cls = getattr(module, class_name)
        params = dict(kwargs)
        try:
            signature = inspect.signature(estimator_cls)
            if "random_state" in signature.parameters and "random_state" not in params:
                params["random_state"] = seed
        except (TypeError, ValueError):
            pass
        return estimator_cls(**params)

    return factory


def format_input(x: np.ndarray, input_kind: str) -> np.ndarray:
    """Convert framework windows [N, V, T] to the estimator's expected layout."""

    x = np.asarray(x)
    if input_kind == "tabular_flat":
        return flatten_windows(x)
    if input_kind == "keras_sequence":
        return np.ascontiguousarray(np.transpose(x, (0, 2, 1)).astype(np.float32))
    if input_kind == "aeon_collection":
        return np.ascontiguousarray(x.astype(np.float32))
    if input_kind == "aeon_collection_pad10":
        return pad_time_axis(np.ascontiguousarray(x.astype(np.float32)), min_timepoints=10)
    if input_kind == "aeon_collection_pad10_float64":
        return pad_time_axis(np.ascontiguousarray(x.astype(np.float64)), min_timepoints=10)
    raise ValueError(f"Unknown official_pool input_kind: {input_kind}")


def pad_time_axis(x: np.ndarray, *, min_timepoints: int) -> np.ndarray:
    if x.shape[-1] >= min_timepoints:
        return x
    pad_width = [(0, 0), (0, 0), (0, min_timepoints - x.shape[-1])]
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=0.0)


def fit_classifier(spec: OfficialPoolSpec, x: np.ndarray, y: np.ndarray, seed: int):
    if len(np.unique(y)) < 2:
        return ConstantClassifier(float(np.mean(y)))
    if spec.classifier_factory is None:
        raise RuntimeError(f"{spec.name} does not define a classifier")
    model = spec.classifier_factory(seed)
    model.fit(x, y.astype(int))
    return model


def fit_regressor(spec: OfficialPoolSpec, x: np.ndarray, y: np.ndarray, seed: int):
    if np.nanstd(y) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y)))
    if spec.regressor_factory is None:
        return None
    model = spec.regressor_factory(seed)
    model.fit(x, y.astype(float))
    return model


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)
        classes = list(getattr(model, "classes_", []))
        if proba.ndim == 2 and proba.shape[1] > 1:
            if 1 in classes:
                return proba[:, classes.index(1)]
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        return sigmoid(np.asarray(model.decision_function(x), dtype=float).reshape(-1))
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


class KerasSequenceClassifier:
    """Keras sequence classifier built from official TensorFlow/Keras layers."""

    def __init__(self, architecture: str, params: dict[str, object], epochs: int, batch_size: int, seed: int) -> None:
        self.architecture = architecture
        self.params = dict(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model_ = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        tf, keras = import_keras()
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = build_keras_sequence_model(self.architecture, np.asarray(x).shape[1:], self.params, "classification", self.seed)
        class_weight = None
        unique, counts = np.unique(y.astype(int), return_counts=True)
        if len(unique) == 2 and counts.min() > 0:
            total = float(counts.sum())
            class_weight = {int(cls): total / (2.0 * float(count)) for cls, count in zip(unique, counts)}
        self.model_.fit(
            np.asarray(x, dtype=np.float32),
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(1, len(y))),
            verbose=0,
            shuffle=False,
            class_weight=class_weight,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Keras classifier is not fitted")
        return np.clip(np.asarray(self.model_.predict(np.asarray(x, dtype=np.float32), verbose=0), dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


class KerasSequenceRegressor:
    """Keras sequence regressor built from official TensorFlow/Keras layers."""

    def __init__(self, architecture: str, params: dict[str, object], epochs: int, batch_size: int, seed: int) -> None:
        self.architecture = architecture
        self.params = dict(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model_ = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        tf, keras = import_keras()
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = build_keras_sequence_model(self.architecture, np.asarray(x).shape[1:], self.params, "regression", self.seed)
        self.model_.fit(
            np.asarray(x, dtype=np.float32),
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(1, len(y))),
            verbose=0,
            shuffle=False,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Keras regressor is not fitted")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=np.float32), verbose=0), dtype=float).reshape(-1)


def import_keras():
    configure_tensorflow_runtime()
    import tensorflow as tf
    from tensorflow import keras

    configure_tensorflow_runtime()
    return tf, keras


def configure_tensorflow_runtime() -> None:
    try:
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass
    except Exception:
        pass


def build_keras_sequence_model(architecture: str, input_shape: tuple[int, ...], params: dict[str, object], task: str, seed: int):
    tf, keras = import_keras()
    _ = tf
    inputs = keras.Input(shape=input_shape)
    x = inputs
    architecture = architecture.lower()
    if architecture in {"lstm", "gru", "bilstm", "bigru"}:
        layer_name = "GRU" if "gru" in architecture else "LSTM"
        layer_cls = getattr(keras.layers, layer_name)
        bidirectional = architecture.startswith("bi") or bool(params.get("bidirectional", False))
        units = as_int_list(params.get("units", 32))
        dropout = float(params.get("dropout", 0.0))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
        for idx, unit in enumerate(units):
            recurrent = layer_cls(
                int(unit),
                activation=str(params.get("activation", "tanh")),
                recurrent_activation=str(params.get("recurrent_activation", "sigmoid")),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=idx < len(units) - 1,
            )
            x = keras.layers.Bidirectional(recurrent)(x) if bidirectional else recurrent(x)
    elif architecture == "tcn":
        from tcn import TCN

        x = TCN(
            nb_filters=int(params.get("nb_filters", 16)),
            kernel_size=int(params.get("kernel_size", 2)),
            nb_stacks=int(params.get("nb_stacks", 1)),
            dilations=tuple(int(v) for v in params.get("dilations", (1,))),
            padding=str(params.get("padding", "causal")),
            use_skip_connections=bool(params.get("use_skip_connections", True)),
            dropout_rate=float(params.get("dropout_rate", 0.0)),
            return_sequences=False,
            activation=str(params.get("activation", "relu")),
            use_batch_norm=bool(params.get("use_batch_norm", False)),
            use_layer_norm=bool(params.get("use_layer_norm", False)),
            name=f"tcn_{seed % 100000}",
        )(x)
    else:
        raise ValueError(f"Unsupported Keras sequence architecture: {architecture}")

    for unit in as_int_list(params.get("dense_units", [])):
        x = keras.layers.Dense(int(unit), activation=str(params.get("dense_activation", "relu")))(x)
        if float(params.get("dense_dropout", 0.0)) > 0:
            x = keras.layers.Dropout(float(params.get("dense_dropout", 0.0)))(x)
    if task == "classification":
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        loss = str(params.get("loss", "binary_crossentropy"))
        metrics = ["accuracy"]
    elif task == "regression":
        outputs = keras.layers.Dense(1, activation="linear")(x)
        loss = str(params.get("loss", "mse"))
        metrics = ["mse"]
    else:
        raise ValueError(f"Unknown Keras task: {task}")
    model = keras.Model(inputs=inputs, outputs=outputs)
    learning_rate = float(params.get("learning_rate", 1e-3))
    optimizer_name = str(params.get("optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=float(params.get("weight_decay", 1e-4)))
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def as_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
