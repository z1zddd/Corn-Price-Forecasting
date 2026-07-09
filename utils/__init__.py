"""Compatibility package for model-layer utilities."""

from __future__ import annotations

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_UTILS_DIR = _PROJECT_ROOT / "玉米预测" / "operator" / "model" / "modeling" / "utils"
__path__ = [str(_UTILS_DIR)]

