"""List or clean generated output directories."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRS = [ROOT / "experiments", ROOT / "outputs", ROOT / ".mplconfig"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List or clean generated backtest outputs")
    parser.add_argument("--yes", action="store_true", help="Actually delete generated output directories")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in OUTPUT_DIRS:
        if not path.exists():
            continue
        if not args.yes:
            print(f"would remove: {path}")
            continue
        resolved = path.resolve()
        if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
            raise ValueError(f"Refusing to remove path outside repository: {resolved}")
        shutil.rmtree(resolved)
        print(f"removed: {resolved}")
    if not args.yes:
        print("Pass --yes to delete these directories.")


if __name__ == "__main__":
    main()
