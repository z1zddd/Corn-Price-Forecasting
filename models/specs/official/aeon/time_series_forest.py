"""TimeSeriesForest official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_ESTIMATORS, spec_pair


def build_time_series_forest_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_tsf", "tsc_interval", "TimeSeriesForestClassifier", "TimeSeriesForestRegressor", "aeon.classification.interval_based", "aeon.regression.interval_based", {"n_estimators": AEON_ESTIMATORS, "min_interval_length": 1, "n_jobs": 1}, {"n_estimators": AEON_ESTIMATORS, "min_interval_length": 1, "n_jobs": 1}, "interval_forest", "interval_forest"),
    ]
