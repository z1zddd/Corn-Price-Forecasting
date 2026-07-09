"""Multi-model comparison utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.metrics import compare_models


def write_comparison(results: dict[str, dict[str, float]], output_dir: str | Path) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison = compare_models(results)
    comparison.to_csv(out / "comparison.csv", index=False)
    return comparison

