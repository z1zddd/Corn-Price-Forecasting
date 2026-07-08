"""Package-native GBDT official-pool model specs."""

from __future__ import annotations

from ..base import Factory, OfficialPoolSpec
from .common import make_spec


def build_gbdt_specs() -> list[OfficialPoolSpec]:
    return [
        make_spec("lightgbm_gbdt", "gbdt", "lightgbm", lightgbm_classifier("gbdt"), lightgbm_regressor("gbdt"), "binary_logloss", "l2"),
        make_spec("lightgbm_goss", "gbdt", "lightgbm", lightgbm_classifier("goss"), lightgbm_regressor("goss"), "binary_logloss_goss", "l2_goss"),
        make_spec("lightgbm_dart", "gbdt", "lightgbm", lightgbm_classifier("dart"), lightgbm_regressor("dart"), "binary_logloss_dart", "l2_dart"),
        make_spec("xgboost_gbtree", "gbdt", "xgboost", xgb_classifier("gbtree"), xgb_regressor("gbtree"), "logloss", "squared_error"),
        make_spec("xgboost_dart", "gbdt", "xgboost", xgb_classifier("dart"), xgb_regressor("dart"), "logloss_dart", "squared_error_dart"),
        make_spec("catboost_depthwise", "gbdt", "catboost", catboost_classifier("Depthwise"), catboost_regressor("Depthwise"), "logloss_depthwise", "rmse_depthwise"),
    ]


def lightgbm_classifier(boosting_type: str) -> Factory:
    def factory(seed: int):
        from lightgbm import LGBMClassifier

        params = {
            "boosting_type": boosting_type,
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        if boosting_type == "goss":
            params.pop("subsample")
        return LGBMClassifier(**params)

    return factory


def lightgbm_regressor(boosting_type: str) -> Factory:
    def factory(seed: int):
        from lightgbm import LGBMRegressor

        params = {
            "boosting_type": boosting_type,
            "n_estimators": 120,
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        if boosting_type == "goss":
            params.pop("subsample")
        return LGBMRegressor(**params)

    return factory


def xgb_classifier(booster: str) -> Factory:
    def factory(seed: int):
        from xgboost import XGBClassifier

        return XGBClassifier(
            booster=booster,
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    return factory


def xgb_regressor(booster: str) -> Factory:
    def factory(seed: int):
        from xgboost import XGBRegressor

        return XGBRegressor(
            booster=booster,
            n_estimators=100,
            learning_rate=0.035,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    return factory


def catboost_classifier(grow_policy: str) -> Factory:
    def factory(seed: int):
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=100,
            learning_rate=0.035,
            depth=3,
            grow_policy=grow_policy,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return factory


def catboost_regressor(grow_policy: str) -> Factory:
    def factory(seed: int):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=100,
            learning_rate=0.035,
            depth=3,
            grow_policy=grow_policy,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return factory
