"""Shared model adapters."""

from corn_forecast.operator.model.wrappers.torch import TorchSequenceClassifierAdapter
from corn_forecast.operator.model.wrappers.source_tree import SourceTreeSequenceRegressorAdapter

__all__ = ["SourceTreeSequenceRegressorAdapter", "TorchSequenceClassifierAdapter"]
