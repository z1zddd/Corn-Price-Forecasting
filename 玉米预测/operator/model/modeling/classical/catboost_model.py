"""CatBoost wrapper using the unified BaseModel contract."""

from __future__ import annotations

from src.models.classical._sklearn_wrapper import SklearnWindowModel


class CatBoostModel(SklearnWindowModel):
    def __init__(self, task: str = "regression", **params):
        params.setdefault("verbose", False)
        if task == "classification":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(**params)
        else:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(**params)
        super().__init__(model=model, task=task)

