"""LightGBM wrapper using the unified BaseModel contract."""

from __future__ import annotations

from src.models.classical._sklearn_wrapper import SklearnWindowModel


class LightGBMModel(SklearnWindowModel):
    def __init__(self, task: str = "regression", **params):
        if task == "classification":
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(**params)
        else:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(**params)
        super().__init__(model=model, task=task)

