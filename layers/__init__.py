"""Compatibility package for model-layer building blocks."""

from __future__ import annotations

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LAYERS_DIR = _PROJECT_ROOT / "玉米预测" / "operator" / "model" / "modeling" / "layers"
__path__ = [str(_LAYERS_DIR)]

