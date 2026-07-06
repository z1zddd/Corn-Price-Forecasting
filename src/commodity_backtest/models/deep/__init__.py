"""Optional torch-backed deep sequence model factories."""

from __future__ import annotations

from importlib import import_module


DEEP_MODEL_NAMES = {"lstm", "gru", "transformer", "patchtst", "itransformer", "dlinear"}

_FACTORIES = {
    "lstm": ("commodity_backtest.models.deep.lstm", "create_lstm"),
    "gru": ("commodity_backtest.models.deep.gru", "create_gru"),
    "transformer": ("commodity_backtest.models.deep.transformer", "create_transformer"),
    "patchtst": ("commodity_backtest.models.deep.patchtst", "create_patchtst"),
    "itransformer": ("commodity_backtest.models.deep.itransformer", "create_itransformer"),
    "dlinear": ("commodity_backtest.models.deep.dlinear", "create_dlinear"),
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
