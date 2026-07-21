from __future__ import annotations

import importlib
from typing import Any

from models.machine_learning._adapter import SklearnPriceRegressor


class LightGBMPriceRegressor(SklearnPriceRegressor):
    source = "https://github.com/microsoft/LightGBM"

    def _make_model(self, params: dict[str, Any]) -> Any:
        try:
            lightgbm = importlib.import_module("lightgbm")
        except ModuleNotFoundError as error:
            raise ImportError(
                "LightGBMPriceRegressor requires LightGBM; install it with `pip install lightgbm`."
            ) from error
        return lightgbm.LGBMRegressor(**params)

