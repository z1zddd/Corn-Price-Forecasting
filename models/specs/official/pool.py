"""Pool assembly and named pool expansion for official model specs."""

from __future__ import annotations

from ensembles.deployment import BEST_DEPLOYMENT_MODEL_POOL_NAME, deployment_ensemble_model_configs

from .aeon import build_aeon_specs
from .base import OFFICIAL_57_MODEL_NAMES, OfficialPoolSpec
from .keras import build_keras_sequence_specs
from .tabular import build_tabular_specs


def expand_model_pool(name: str) -> list[dict]:
    """Expand a named model pool into framework model configs."""

    if name == "official_57":
        return [{"name": model_name, "type": "official_pool", "enabled": True} for model_name in OFFICIAL_57_MODEL_NAMES]
    if name == BEST_DEPLOYMENT_MODEL_POOL_NAME:
        return deployment_ensemble_model_configs()
    if name == "official_57_plus_best_deployment":
        return [
            *[{"name": model_name, "type": "official_pool", "enabled": True} for model_name in OFFICIAL_57_MODEL_NAMES],
            *deployment_ensemble_model_configs(),
        ]
    raise ValueError(f"Unknown model pool: {name}")


def build_official_model_pool() -> list[OfficialPoolSpec]:
    """Return the 57-model pool used by the long-lookback corn benchmark."""

    specs = {spec.name: spec for spec in [*build_tabular_specs(), *build_aeon_specs(), *build_keras_sequence_specs()]}
    missing = [name for name in OFFICIAL_57_MODEL_NAMES if name not in specs]
    if missing:
        raise AssertionError(f"Official model pool is missing entries: {missing}")
    return [specs[name] for name in OFFICIAL_57_MODEL_NAMES]
