"""Simplified YAML config loader inspired by lightning-hydra-template."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parents[1]
MODEL_CONFIG_DIR = PROJECT_ROOT / "玉米预测" / "operator" / "model" / "configs"


def load_config(name: str = "experiment", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = resolve_config_path(name)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = deepcopy(cfg)
    for key, value in (overrides or {}).items():
        set_by_dotted_key(cfg, key, value)
    return cfg


def resolve_config_path(name: str) -> Path:
    raw = Path(name)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    if raw.suffix:
        candidates.append(CONFIG_DIR / raw)
    else:
        candidates.append(CONFIG_DIR / f"{name}.yaml")
        candidates.append(CONFIG_DIR / name)

    normalized = str(name).replace("\\", "/")
    if normalized.startswith("model/"):
        model_name = normalized.split("/", 1)[1]
        model_path = MODEL_CONFIG_DIR / model_name
        if model_path.suffix:
            candidates.append(model_path)
        else:
            candidates.append(model_path.with_suffix(".yaml"))
            candidates.append(model_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Config not found: {name}")


def set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    current = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
