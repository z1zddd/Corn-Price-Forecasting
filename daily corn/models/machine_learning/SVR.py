from __future__ import annotations

from typing import Any

from sklearn.svm import SVR

from models.machine_learning._adapter import SklearnPriceRegressor


class SVRPriceRegressor(SklearnPriceRegressor):
    source = "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"

    def _make_model(self, params: dict[str, Any]) -> SVR:
        params.pop("random_state", None)
        return SVR(**params)

