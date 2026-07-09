"""Legacy compatibility package for `corn_forecast.operator.model`.

New code should import model operators from `corn_forecast.operator.model`.
This package stays light at import time and resolves common registry symbols
on demand.
"""

__all__ = [
    "create_model",
    "expand_model_configs",
    "normalize_model_config",
]


def __getattr__(name: str):
    if name in __all__:
        from corn_forecast.operator import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
