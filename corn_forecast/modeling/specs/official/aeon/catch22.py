"""Catch22 official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import spec_pair


def build_catch22_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_catch22", "tsc_feature", "Catch22Classifier", "Catch22Regressor", "aeon.classification.feature_based", "aeon.regression.feature_based", {"catch24": True, "replace_nans": True, "n_jobs": 1}, {"catch24": True, "replace_nans": True, "n_jobs": 1}, "catch22_random_forest", "catch22_random_forest"),
    ]
