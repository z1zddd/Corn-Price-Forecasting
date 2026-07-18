from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path("/root/dce_corn_rf/daily_corn_xgboost_repo/daily corn")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_experiment as runner  # noqa: E402


_make_fixed_split = runner.make_fixed_split


def make_fixed_split_without_additional_embargo(samples, ratios, embargo):
    return _make_fixed_split(samples, ratios=ratios, embargo=0)


runner.make_fixed_split = make_fixed_split_without_additional_embargo


if __name__ == "__main__":
    runner.main()
