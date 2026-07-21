from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsRegressor

from models.machine_learning._adapter import SklearnPriceRegressor


class KNNPriceRegressor(SklearnPriceRegressor):
    source = "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html"

    def _make_model(self, params: dict[str, Any]) -> KNeighborsRegressor:
        params.pop("random_state", None)
        return KNeighborsRegressor(**params)

