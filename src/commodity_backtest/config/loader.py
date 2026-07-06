"""YAML config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from commodity_backtest.config.schema import validate_config


def load_config(path: str | Path, *, validate: bool = False) -> dict[str, Any]:
    """Load a YAML config from disk."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    if validate:
        validate_config(config)
    return config