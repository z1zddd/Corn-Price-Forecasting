"""Compatibility facade for the official 57-model pool.

The implementation lives under :mod:`models.specs.official`, split by
model family. Keep this module so older configs/imports continue to work.
"""

from models.specs.official import *  # noqa: F401,F403
