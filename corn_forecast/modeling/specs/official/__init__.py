"""Legacy compatibility shim for `corn_forecast.modeling.specs.official`."""

from corn_forecast.operator.model.families import official as _official

__all__ = _official.__all__


def __getattr__(name: str):
    if name in __all__:
        return getattr(_official, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
