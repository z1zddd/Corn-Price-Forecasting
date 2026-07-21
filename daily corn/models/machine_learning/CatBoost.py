from __future__ import annotations

import importlib
from typing import Any

from models.machine_learning._adapter import SklearnPriceRegressor


class CatBoostPriceRegressor(SklearnPriceRegressor):
    source = "https://github.com/catboost/catboost"

    def _make_model(self, params: dict[str, Any]) -> Any:
        try:
            catboost = importlib.import_module("catboost")
        except ModuleNotFoundError as error:
            raise ImportError(
                "CatBoostPriceRegressor requires CatBoost; install it with `pip install catboost`."
            ) from error
        return catboost.CatBoostRegressor(**params)
