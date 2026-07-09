"""Official 57-model pool, split by model family."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AEON_MODEL_NAMES": "corn_forecast.operator.model.families.official.base",
    "KERAS_SEQUENCE_MODEL_NAMES": "corn_forecast.operator.model.families.official.base",
    "OFFICIAL_57_MODEL_NAMES": "corn_forecast.operator.model.families.official.base",
    "TABULAR_MODEL_NAMES": "corn_forecast.operator.model.families.official.base",
    "Factory": "corn_forecast.operator.model.families.official.base",
    "OfficialPoolSpec": "corn_forecast.operator.model.families.official.base",
    "ConstantClassifier": "corn_forecast.operator.model.families.official.adapter",
    "ConstantRegressor": "corn_forecast.operator.model.families.official.adapter",
    "OfficialPoolAdapter": "corn_forecast.operator.model.families.official.adapter",
    "create_official_pool_model": "corn_forecast.operator.model.families.official.adapter",
    "fit_classifier": "corn_forecast.operator.model.families.official.adapter",
    "fit_regressor": "corn_forecast.operator.model.families.official.adapter",
    "format_input": "corn_forecast.operator.model.families.official.io",
    "pad_time_axis": "corn_forecast.operator.model.families.official.io",
    "positive_probability": "corn_forecast.operator.model.families.official.io",
    "sigmoid": "corn_forecast.operator.model.families.official.io",
    "build_official_model_pool": "corn_forecast.operator.model.families.official.pool",
    "expand_model_pool": "corn_forecast.operator.model.families.official.pool",
    "aeon_factory": "corn_forecast.operator.model.families.official.aeon",
    "build_aeon_specs": "corn_forecast.operator.model.families.official.aeon",
    "deep_pair": "corn_forecast.operator.model.families.official.aeon",
    "spec_pair": "corn_forecast.operator.model.families.official.aeon",
    "KerasSequenceClassifier": "corn_forecast.operator.model.families.official.keras",
    "KerasSequenceRegressor": "corn_forecast.operator.model.families.official.keras",
    "as_int_list": "corn_forecast.operator.model.families.official.keras",
    "build_keras_sequence_model": "corn_forecast.operator.model.families.official.keras",
    "build_keras_sequence_specs": "corn_forecast.operator.model.families.official.keras",
    "configure_tensorflow_runtime": "corn_forecast.operator.model.families.official.keras",
    "import_keras": "corn_forecast.operator.model.families.official.keras",
    "keras_sequence_pair": "corn_forecast.operator.model.families.official.keras",
    "build_tabular_specs": "corn_forecast.operator.model.families.official.tabular",
    "adaboost_tree_classifier_factory": "corn_forecast.operator.model.families.official.tabular.common",
    "catboost_classifier": "corn_forecast.operator.model.families.official.tabular.gbdt",
    "catboost_regressor": "corn_forecast.operator.model.families.official.tabular.gbdt",
    "lightgbm_classifier": "corn_forecast.operator.model.families.official.tabular.gbdt",
    "lightgbm_regressor": "corn_forecast.operator.model.families.official.tabular.gbdt",
    "xgb_classifier": "corn_forecast.operator.model.families.official.tabular.gbdt",
    "xgb_regressor": "corn_forecast.operator.model.families.official.tabular.gbdt",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
