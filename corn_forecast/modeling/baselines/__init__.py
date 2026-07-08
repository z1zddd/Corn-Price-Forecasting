"""Baseline model implementations."""

from corn_forecast.modeling.baselines.simple import LastReturnBaseline, MeanDirectionBaseline

__all__ = ["LastReturnBaseline", "MeanDirectionBaseline"]
