"""Aeon official-pool specs grouped one method per file."""

from corn_forecast.operator.model.families.official.aeon.catch22 import build_catch22_specs
from corn_forecast.operator.model.families.official.aeon.common import aeon_factory, deep_pair, spec_pair
from corn_forecast.operator.model.families.official.aeon.deep_fcn import build_deep_fcn_specs
from corn_forecast.operator.model.families.official.aeon.deep_inceptiontime import build_deep_inceptiontime_specs
from corn_forecast.operator.model.families.official.aeon.deep_mlp import build_deep_mlp_specs
from corn_forecast.operator.model.families.official.aeon.deep_timecnn import build_deep_timecnn_specs
from corn_forecast.operator.model.families.official.aeon.knn_dtw import build_knn_dtw_specs
from corn_forecast.operator.model.families.official.aeon.knn_euclidean import build_knn_euclidean_specs
from corn_forecast.operator.model.families.official.aeon.minirocket import build_minirocket_specs
from corn_forecast.operator.model.families.official.aeon.multirocket import build_multirocket_specs
from corn_forecast.operator.model.families.official.aeon.multirocket_hydra import build_multirocket_hydra_specs
from corn_forecast.operator.model.families.official.aeon.rdst import build_rdst_specs
from corn_forecast.operator.model.families.official.aeon.rise import build_rise_specs
from corn_forecast.operator.model.families.official.aeon.summary import build_summary_specs
from corn_forecast.operator.model.families.official.aeon.time_series_forest import build_time_series_forest_specs
from corn_forecast.operator.model.families.official.base import OfficialPoolSpec


def build_aeon_specs() -> list[OfficialPoolSpec]:
    """Build aeon entries using aeon's official estimator classes."""

    return [
        *build_minirocket_specs(),
        *build_multirocket_specs(),
        *build_multirocket_hydra_specs(),
        *build_time_series_forest_specs(),
        *build_rise_specs(),
        *build_catch22_specs(),
        *build_summary_specs(),
        *build_rdst_specs(),
        *build_knn_dtw_specs(),
        *build_knn_euclidean_specs(),
        *build_deep_mlp_specs(),
        *build_deep_fcn_specs(),
        *build_deep_inceptiontime_specs(),
        *build_deep_timecnn_specs(),
    ]


__all__ = [
    "aeon_factory",
    "build_aeon_specs",
    "build_catch22_specs",
    "build_deep_fcn_specs",
    "build_deep_inceptiontime_specs",
    "build_deep_mlp_specs",
    "build_deep_timecnn_specs",
    "build_knn_dtw_specs",
    "build_knn_euclidean_specs",
    "build_minirocket_specs",
    "build_multirocket_hydra_specs",
    "build_multirocket_specs",
    "build_rdst_specs",
    "build_rise_specs",
    "build_summary_specs",
    "build_time_series_forest_specs",
    "deep_pair",
    "spec_pair",
]
