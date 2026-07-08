"""Summary feature official-pool spec."""

from __future__ import annotations

from ..base import OfficialPoolSpec
from .common import spec_pair


def build_summary_specs() -> list[OfficialPoolSpec]:
    return [
        spec_pair("aeon_summary", "tsc_feature", "SummaryClassifier", "SummaryRegressor", "aeon.classification.feature_based", "aeon.regression.feature_based", {"n_jobs": 1}, {"n_jobs": 1}, "summary_random_forest", "summary_random_forest"),
    ]
