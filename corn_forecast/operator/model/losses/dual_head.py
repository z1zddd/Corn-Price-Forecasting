"""Dual-head direction and return model variant."""

from corn_forecast.operator.model.losses.variants import DualHeadMseBceModel, create_dual_head_mse_bce

__all__ = ["DualHeadMseBceModel", "create_dual_head_mse_bce"]
