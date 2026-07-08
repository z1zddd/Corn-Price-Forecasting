"""Official 57-model pool, split by model family."""

from corn_forecast.modeling.specs.official.adapter import (
    ConstantClassifier,
    ConstantRegressor,
    OfficialPoolAdapter,
    create_official_pool_model,
    fit_classifier,
    fit_regressor,
)
from corn_forecast.modeling.specs.official.aeon import aeon_factory, build_aeon_specs, deep_pair, spec_pair
from corn_forecast.modeling.specs.official.base import (
    AEON_MODEL_NAMES,
    KERAS_SEQUENCE_MODEL_NAMES,
    OFFICIAL_57_MODEL_NAMES,
    TABULAR_MODEL_NAMES,
    Factory,
    OfficialPoolSpec,
)
from corn_forecast.modeling.specs.official.io import format_input, pad_time_axis, positive_probability, sigmoid
from corn_forecast.modeling.specs.official.keras import (
    KerasSequenceClassifier,
    KerasSequenceRegressor,
    as_int_list,
    build_keras_sequence_model,
    build_keras_sequence_specs,
    configure_tensorflow_runtime,
    import_keras,
    keras_sequence_pair,
)
from corn_forecast.modeling.specs.official.pool import build_official_model_pool, expand_model_pool
from corn_forecast.modeling.specs.official.tabular import build_tabular_specs
from corn_forecast.modeling.specs.official.tabular.common import adaboost_tree_classifier_factory
from corn_forecast.modeling.specs.official.tabular.gbdt import (
    catboost_classifier,
    catboost_regressor,
    lightgbm_classifier,
    lightgbm_regressor,
    xgb_classifier,
    xgb_regressor,
)

__all__ = [
    "AEON_MODEL_NAMES",
    "KERAS_SEQUENCE_MODEL_NAMES",
    "OFFICIAL_57_MODEL_NAMES",
    "TABULAR_MODEL_NAMES",
    "ConstantClassifier",
    "ConstantRegressor",
    "Factory",
    "KerasSequenceClassifier",
    "KerasSequenceRegressor",
    "OfficialPoolAdapter",
    "OfficialPoolSpec",
    "adaboost_tree_classifier_factory",
    "aeon_factory",
    "as_int_list",
    "build_aeon_specs",
    "build_keras_sequence_model",
    "build_keras_sequence_specs",
    "build_official_model_pool",
    "build_tabular_specs",
    "catboost_classifier",
    "catboost_regressor",
    "configure_tensorflow_runtime",
    "create_official_pool_model",
    "deep_pair",
    "expand_model_pool",
    "fit_classifier",
    "fit_regressor",
    "format_input",
    "import_keras",
    "keras_sequence_pair",
    "lightgbm_classifier",
    "lightgbm_regressor",
    "pad_time_axis",
    "positive_probability",
    "sigmoid",
    "spec_pair",
    "xgb_classifier",
    "xgb_regressor",
]
