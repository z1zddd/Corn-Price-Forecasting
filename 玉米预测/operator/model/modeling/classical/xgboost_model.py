"""XGBoost wrapper using the unified BaseModel contract."""

from __future__ import annotations

from src.models.classical._sklearn_wrapper import SklearnWindowModel


class XGBoostModel(SklearnWindowModel):
    def __init__(self, task: str = "regression", **params):
        if task == "classification":
            from xgboost import XGBClassifier

            model = XGBClassifier(eval_metric="logloss", **params)
        else:
            from xgboost import XGBRegressor

            model = XGBRegressor(objective="reg:squarederror", **params)
        super().__init__(model=model, task=task)

