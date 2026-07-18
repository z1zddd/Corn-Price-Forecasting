from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path("/root/dce_corn_rf/daily_corn_xgboost_repo/daily corn")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments import run_experiment as runner  # noqa: E402


def make_fixed_split_without_additional_embargo(samples, ratios, embargo):
    del embargo
    n_samples = len(samples.X)
    train_end = int(np.floor(n_samples * ratios[0]))
    validation_end = int(np.floor(n_samples * (ratios[0] + ratios[1])))
    validation_candidates = np.arange(train_end, validation_end, dtype=int)
    test_idx = np.arange(validation_end, n_samples, dtype=int)
    if validation_candidates.size == 0 or test_idx.size == 0:
        raise ValueError("Zero-embargo split leaves an empty validation or test segment")

    first_validation_anchor = samples.metadata.iloc[validation_candidates[0]][
        "anchor_date"
    ]
    first_test_anchor = samples.metadata.iloc[test_idx[0]]["anchor_date"]
    train_candidates = np.arange(0, train_end, dtype=int)
    train_idx = train_candidates[
        (
            samples.metadata.iloc[train_candidates]["target_date"]
            <= first_validation_anchor
        ).to_numpy()
    ]
    validation_idx = validation_candidates[
        (
            samples.metadata.iloc[validation_candidates]["target_date"]
            <= first_test_anchor
        ).to_numpy()
    ]
    refit_candidates = np.arange(0, validation_end, dtype=int)
    refit_idx = refit_candidates[
        (
            samples.metadata.iloc[refit_candidates]["target_date"]
            <= first_test_anchor
        ).to_numpy()
    ]
    if train_idx.size == 0 or validation_idx.size == 0 or refit_idx.size == 0:
        raise ValueError("Target-date purging leaves an empty split segment")
    return runner.FixedSplit(train_idx, validation_idx, refit_idx, test_idx)


runner.make_fixed_split = make_fixed_split_without_additional_embargo


if __name__ == "__main__":
    runner.main()
