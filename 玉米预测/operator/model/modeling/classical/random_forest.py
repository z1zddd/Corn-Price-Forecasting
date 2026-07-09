"""Random Forest wrapper using the unified BaseModel contract."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.classical._sklearn_wrapper import SklearnWindowModel


class RandomForestModel(SklearnWindowModel):
    def __init__(self, task: str = "regression", **params):
        if task == "classification":
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)
        super().__init__(model=model, task=task)

