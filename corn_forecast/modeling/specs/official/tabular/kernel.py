"""Kernel official-pool model specs."""

from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from ..base import OfficialPoolSpec
from .common import make_spec


def build_kernel_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("gaussian_process_rbf", "kernel", "sklearn", lambda s: GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), max_iter_predict=100, random_state=s), lambda s: GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0), alpha=1e-2, normalize_y=True, random_state=s), "laplace_log_marginal", "gp_regression"),
    ]
