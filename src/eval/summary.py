"""Markdown summary writers for experiment roots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(
    output_dir: str | Path,
    title: str,
    run_type: str,
    data_config: dict,
    comparison: pd.DataFrame,
    extra_lines: list[str] | None = None,
) -> None:
    output = Path(output_dir)
    lines = [
        f"# {title}",
        "",
        f"Run type: `{run_type}`",
        f"Target mode: `{data_config.get('target_mode')}`",
        f"Horizon: `{data_config.get('horizon')}`",
        f"Sequence length: `{data_config.get('seq_len')}`",
        f"CSV: `{data_config.get('csv_path')}`",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Outputs",
        "",
        "- `comparison.csv`",
        "- `data_manifest.json`",
        "- `experiment_config.json`",
        "- `data_config.json`",
    ]
    if extra_lines:
        lines.extend(["", *extra_lines])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_training_history(model, output_dir: str | Path) -> None:
    history = getattr(model, "history", None)
    if not history:
        return
    out = Path(output_dir)
    frame = pd.DataFrame(history)
    if not frame.empty:
        frame.to_csv(out / "training_history.csv", index=False)
