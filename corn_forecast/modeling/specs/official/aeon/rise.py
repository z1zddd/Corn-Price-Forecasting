"""RISE official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import AEON_ESTIMATORS, spec_pair


def build_rise_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_rise", "tsc_interval", "RandomIntervalSpectralEnsembleClassifier", "RandomIntervalSpectralEnsembleRegressor", "aeon.classification.interval_based", "aeon.regression.interval_based", {"n_estimators": max(16, AEON_ESTIMATORS // 2), "min_interval_length": 1, "acf_lag": 1, "acf_min_values": 1, "n_jobs": 1}, {"n_estimators": max(16, AEON_ESTIMATORS // 2), "min_interval_length": 1, "acf_lag": 1, "acf_min_values": 1, "n_jobs": 1}, "rise", "rise"),
    ]
