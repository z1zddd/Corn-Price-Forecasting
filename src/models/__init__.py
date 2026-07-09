"""Compatibility package for model operators.

The actual model layer lives under ``玉米预测/operator/model/modeling``.
Keeping this package path lets existing configs with ``src.models.*`` class
paths continue to work while the repository layout follows the operator /
datasets split.
"""

from __future__ import annotations

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELING_DIR = _PROJECT_ROOT / "玉米预测" / "operator" / "model" / "modeling"
__path__ = [str(_MODELING_DIR)]

