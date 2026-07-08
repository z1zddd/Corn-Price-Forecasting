"""Bayes and discriminant official-pool model specs."""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.naive_bayes import GaussianNB

from ..base import OfficialPoolSpec
from .common import make_spec


def build_statistical_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("gaussian_nb", "bayes", "sklearn", lambda s: GaussianNB(), lambda s: BayesianRidge(), "gaussian_nb", "bayesian_ridge"),
        make_spec("lda_shrinkage", "discriminant", "sklearn", lambda s: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), lambda s: Ridge(alpha=1.0), "lda_shrinkage", "ridge_l2"),
    ]
