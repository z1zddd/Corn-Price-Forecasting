"""Optional torch-backed deep sequence model factories."""

from __future__ import annotations

from importlib import import_module


DEEP_MODEL_NAMES = {"lstm", "gru", "transformer", "patchtst", "itransformer", "dlinear", "dual_stream_lstm"}

_FACTORIES = {
    "lstm": ("corn_forecast.modeling.sequence.lstm", "create_lstm"),
    "gru": ("corn_forecast.modeling.sequence.gru", "create_gru"),
    "transformer": ("corn_forecast.modeling.sequence.transformer", "create_transformer"),
    "patchtst": ("corn_forecast.modeling.sequence.patchtst", "create_patchtst"),
    "itransformer": ("corn_forecast.modeling.sequence.itransformer", "create_itransformer"),
    "dlinear": ("corn_forecast.modeling.sequence.dlinear", "create_dlinear"),
    "dual_stream_lstm": ("corn_forecast.modeling.sequence.dual_stream_lstm", "create_dual_stream_lstm"),
}


def create_deep_model(name: str, params: dict):
    """Create an optional torch sequence classifier by model name."""

    if name not in _FACTORIES:
        raise ValueError(f"Unknown deep model name: {name}")
    module_name, factory_name = _FACTORIES[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        if exc.name == "torch":
            raise ImportError("torch is required for deep sequence models. Install with: pip install -e .[deep]") from exc
        raise
    return getattr(module, factory_name)(params)
