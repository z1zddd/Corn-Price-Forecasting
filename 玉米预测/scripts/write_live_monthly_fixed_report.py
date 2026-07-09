#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    out_dir = Path(sys.argv[1]).expanduser().resolve()
    summary_path = out_dir / "checkpoint_summary_metrics.csv"
    progress_path = out_dir / "PROGRESS.txt"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    ok = summary[summary["r2_status"].eq("ok")].copy()
    cols = [
        "feature_set",
        "horizon_months",
        "head",
        "model",
        "lookback_months",
        "n_predictions",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "reg_price_r2",
        "r2_status",
    ]
    lines: list[str] = ["# Live Monthly Fixed Rolling Leaderboard", ""]
    if progress_path.exists():
        lines.extend(["```", progress_path.read_text(encoding="utf-8").strip(), "```", ""])

    lines.append("## Best By Feature / Horizon / Head")
    rows = []
    for _, group in ok.groupby(["feature_set", "horizon_months", "head"]):
        rows.append(group.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False).iloc[0])
    if rows:
        lines.append(pd.DataFrame(rows)[cols].sort_values(["feature_set", "horizon_months", "head"]).to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("_No R2-ok rows yet._")

    lines.extend(["", "## Top 30 Overall"])
    if not ok.empty:
        lines.append(ok.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False)[cols].head(30).to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("_No R2-ok rows yet._")

    rf = summary[summary["model"].eq("random_forest_shallow")]
    if not rf.empty:
        lines.extend(["", "## random_forest_shallow"])
        lines.append(rf[cols].sort_values(["feature_set", "lookback_months", "horizon_months", "head"]).to_markdown(index=False, floatfmt=".3f"))

    if (out_dir / "checkpoint_model_errors.csv").exists():
        err = pd.read_csv(out_dir / "checkpoint_model_errors.csv")
        if not err.empty:
            lines.extend(["", "## Error Counts"])
            err_counts = err.groupby(["model", "phase", "error_type"]).size().sort_values(ascending=False).head(30).reset_index(name="count")
            lines.append(err_counts.to_markdown(index=False))

    (out_dir / "LIVE_LEADERBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
