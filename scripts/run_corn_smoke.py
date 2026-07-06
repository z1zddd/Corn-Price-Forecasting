"""Run a small corn smoke backtest from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from commodity_backtest.backtest.engine import run_backtest
from commodity_backtest.config.loader import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "corn.yaml", validate=True)
    output_dir = ROOT / "experiments" / "corn_smoke"
    comparison = run_backtest(cfg, output_dir=output_dir)
    print(comparison.to_string(index=False))
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()