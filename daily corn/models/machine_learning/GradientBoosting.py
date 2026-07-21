from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor

from models.machine_learning._adapter import SklearnPriceRegressor


class GradientBoostingPriceRegressor(SklearnPriceRegressor):
    source = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html"

    def _make_model(self, params: dict[str, Any]) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(**params)

