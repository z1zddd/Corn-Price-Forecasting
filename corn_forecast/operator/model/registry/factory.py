"""Model factory registry."""

from __future__ import annotations


MODEL_TYPE_ALIASES = {
    "last_return": "baseline",
    "mean_return": "baseline",
    "mean_direction": "baseline",
    "random_forest": "sklearn_random_forest",
    "logistic_regression": "sklearn_logistic_regression",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "official_pool": "official_pool",
    "deployment_ensemble": "deployment_ensemble",
    "regression_mse_sign": "regression_mse_sign",
    "regression_mae_sign": "regression_mae_sign",
    "regression_huber_sign": "regression_huber_sign",
    "dual_head_mse_bce": "dual_head_mse_bce",
    "focal_logistic": "focal_logistic",
    "lstm": "lstm",
    "gru": "gru",
    "transformer": "transformer",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
    "dlinear": "dlinear",
    "dual_stream_lstm": "dual_stream_lstm",
    "simpletm": "simpletm",
    "timemixer": "timemixer",
    "tide": "tide",
    "xlinear": "xlinear",
}


def expand_model_configs(models_config) -> list:
    """Expand named model pools into explicit model configs."""

    from corn_forecast.operator.model.families.official.pool import expand_model_pool

    if isinstance(models_config, str):
        return expand_model_pool(models_config)
    if isinstance(models_config, dict) and "pool" in models_config:
        pool_name = str(models_config["pool"])
        expanded = expand_model_pool(pool_name)
        disabled = set(models_config.get("disable", []) or [])
        enabled_only = set(models_config.get("enable_only", []) or [])
        for model in expanded:
            if model["name"] in disabled:
                model["enabled"] = False
            if enabled_only:
                model["enabled"] = model["name"] in enabled_only
        return expanded
    return list(models_config)


def normalize_model_config(model_config: dict | str) -> dict:
    """Normalize string, name-only, and typed YAML model configs."""

    if isinstance(model_config, str):
        name = model_config
        return {"name": name, "type": MODEL_TYPE_ALIASES.get(name, name), "params": {}, "enabled": True}
    normalized = dict(model_config)
    name = normalized.get("name")
    if "type" not in normalized and name is not None:
        normalized["type"] = MODEL_TYPE_ALIASES.get(str(name), str(name))
    normalized.setdefault("params", {})
    normalized.setdefault("enabled", True)
    return normalized


def create_model(model_config: dict | str):
    """Create a model from a YAML model config."""

    config = normalize_model_config(model_config)
    model_type = config.get("type")
    params = dict(config.get("params") or {})
    if model_type == "baseline":
        from corn_forecast.operator.model.families.baseline import LastReturnBaseline, MeanDirectionBaseline

        name = config.get("name")
        if name == "last_return":
            return LastReturnBaseline()
        if name in {"mean_return", "mean_direction"}:
            return MeanDirectionBaseline()
        raise ValueError(f"Unknown baseline model name: {name}")
    if model_type == "sklearn_random_forest":
        from corn_forecast.operator.model.families.classical.sklearn import create_random_forest

        return create_random_forest(params)
    if model_type == "sklearn_logistic_regression":
        from corn_forecast.operator.model.families.classical.sklearn import create_logistic_regression

        return create_logistic_regression(params)
    if model_type == "lightgbm":
        from corn_forecast.operator.model.families.classical.sklearn import create_lightgbm

        return create_lightgbm(params)
    if model_type == "xgboost":
        from corn_forecast.operator.model.families.classical.sklearn import create_xgboost

        return create_xgboost(params)
    if model_type == "catboost":
        from corn_forecast.operator.model.families.classical.sklearn import create_catboost

        return create_catboost(params)
    if model_type == "official_pool":
        from corn_forecast.operator.model.families.official.adapter import create_official_pool_model

        return create_official_pool_model(str(config.get("name")), params)
    if model_type == "deployment_ensemble":
        from corn_forecast.operator.model.families.aggregation.deployment_vote import create_deployment_ensemble_model

        return create_deployment_ensemble_model(str(config.get("name")), params)
    if model_type == "regression_mse_sign":
        from corn_forecast.operator.model.losses import create_regression_mse_sign

        return create_regression_mse_sign(params)
    if model_type == "regression_mae_sign":
        from corn_forecast.operator.model.losses import create_regression_mae_sign

        return create_regression_mae_sign(params)
    if model_type == "regression_huber_sign":
        from corn_forecast.operator.model.losses import create_regression_huber_sign

        return create_regression_huber_sign(params)
    if model_type == "dual_head_mse_bce":
        from corn_forecast.operator.model.losses import create_dual_head_mse_bce

        return create_dual_head_mse_bce(params)
    if model_type == "focal_logistic":
        from corn_forecast.operator.model.losses import create_focal_logistic

        return create_focal_logistic(params)
    from corn_forecast.operator.model.families.sequence import DEEP_MODEL_NAMES, create_deep_model

    if model_type == "deep":
        return create_deep_model(str(config.get("name")), params)
    if model_type in DEEP_MODEL_NAMES:
        return create_deep_model(str(model_type), params)
    raise ValueError(f"Unknown model type: {model_type}")
