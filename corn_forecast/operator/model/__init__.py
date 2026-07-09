"""Model operators, registries, wrappers, losses, and model families."""

__all__ = [
    "create_model",
    "expand_model_configs",
    "normalize_model_config",
]


def __getattr__(name: str):
    if name in __all__:
        from corn_forecast.operator.model import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
