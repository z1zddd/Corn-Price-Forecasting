"""Shared helpers for aeon official-pool specs."""

from __future__ import annotations

import importlib
import inspect

from ..base import Factory, OfficialPoolSpec
from ..keras.runtime import configure_tensorflow_runtime


AEON_KERNELS = 384
AEON_ESTIMATORS = 64
AEON_DEEP_EPOCHS = 12
AEON_DEEP_BATCH_SIZE = 16


def spec_pair(
    name: str,
    family: str,
    cls_name: str,
    reg_name: str,
    cls_module: str,
    reg_module: str,
    cls_kwargs: dict[str, object],
    reg_kwargs: dict[str, object],
    classifier_loss: str,
    regressor_loss: str,
    input_kind: str = "aeon_collection",
) -> OfficialPoolSpec:
    return OfficialPoolSpec(
        name=name,
        family=family,
        package="aeon",
        classifier_factory=aeon_factory(cls_module, cls_name, cls_kwargs),
        regressor_factory=aeon_factory(reg_module, reg_name, reg_kwargs),
        classifier_loss=classifier_loss,
        regressor_loss=regressor_loss,
        input_kind=input_kind,
        source_kind="official_aeon",
    )


def deep_pair(
    name: str,
    cls_name: str,
    reg_name: str,
    epochs: int,
    batch_size: int,
    cls_extra: dict[str, object],
    reg_extra: dict[str, object],
) -> OfficialPoolSpec:
    base = {
        "n_epochs": epochs,
        "batch_size": batch_size,
        "verbose": False,
        "save_best_model": False,
        "save_last_model": False,
        "save_init_model": False,
    }
    return spec_pair(
        name,
        "tsc_deep_learning",
        cls_name,
        reg_name,
        "aeon.classification.deep_learning",
        "aeon.regression.deep_learning",
        {**base, **cls_extra},
        {**base, **reg_extra},
        "deep_categorical_crossentropy",
        "deep_mean_squared_error",
    )


def aeon_factory(module_name: str, class_name: str, kwargs: dict[str, object]) -> Factory:
    def factory(seed: int):
        if "deep_learning" in module_name:
            configure_tensorflow_runtime()
        module = importlib.import_module(module_name)
        estimator_cls = getattr(module, class_name)
        params = dict(kwargs)
        try:
            signature = inspect.signature(estimator_cls)
            if "random_state" in signature.parameters and "random_state" not in params:
                params["random_state"] = seed
        except (TypeError, ValueError):
            pass
        return estimator_cls(**params)

    return factory
