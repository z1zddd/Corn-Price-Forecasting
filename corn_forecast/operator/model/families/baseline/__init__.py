"""Baseline model implementations."""

from corn_forecast.operator.model.families.baseline.simple import LastReturnBaseline, MeanDirectionBaseline

__all__ = ["LastReturnBaseline", "MeanDirectionBaseline"]
