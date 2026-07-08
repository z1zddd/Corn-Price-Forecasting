"""SVM official-pool model specs."""

from __future__ import annotations

from sklearn.svm import SVC, SVR

from ..base import OfficialPoolSpec
from .common import make_spec


def build_svm_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("svc_rbf", "svm", "sklearn", lambda s: SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=1.0, gamma="scale", kernel="rbf"), "hinge_platt", "epsilon_insensitive"),
        make_spec("svc_poly2", "svm", "sklearn", lambda s: SVC(C=0.7, degree=2, gamma="scale", kernel="poly", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=0.7, degree=2, gamma="scale", kernel="poly"), "hinge_platt", "epsilon_insensitive"),
        make_spec("svc_sigmoid", "svm", "sklearn", lambda s: SVC(C=0.7, gamma="scale", kernel="sigmoid", class_weight="balanced", probability=True, random_state=s), lambda s: SVR(C=0.7, gamma="scale", kernel="sigmoid"), "hinge_platt", "epsilon_insensitive"),
    ]
