#!/usr/bin/env python3
"""Aggregate per-model rolling benchmark outputs into live leaderboards."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_corn_monthly_spike_official_pool_two_heads import summarize  # noqa: E402


SORT_COLS = ["balanced_accuracy", "auc", "average_precision"]
ERROR_COLUMNS = [
    "source_model_dir",
    "source_run_dir",
    "source_is_checkpoint",
    "feature_set",
    "model",
    "family",
    "package",
    "lookback_months",
    "horizon_months",
    "origin_id",
    "head",
    "error_type",
    "error_message",
]
STATUS_COLUMNS = ["timestamp", "model", "status", "rc", "run_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root output directory containing parts/<model>/<run_id>")
    parser.add_argument("--out", default="", help="Aggregation output directory. Defaults to <root>/live")
    return parser.parse_args()


def collect_csvs(root: Path, final_name: str, checkpoint_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for part_dir in sorted((root / "parts").glob("*")):
        if not part_dir.is_dir():
            continue
        run_dirs = [p for p in part_dir.iterdir() if p.is_dir()]
        if not run_dirs:
            continue
        run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
        source = run_dir / final_name
        is_checkpoint = False
        if not source.exists():
            source = run_dir / checkpoint_name
            is_checkpoint = True
        if not source.exists():
            continue
        try:
            df = pd.read_csv(source)
        except Exception:
            continue
        if df.empty:
            continue
        df.insert(0, "source_is_checkpoint", is_checkpoint)
        df.insert(0, "source_run_dir", str(run_dir))
        df.insert(0, "source_model_dir", part_dir.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def deduplicate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    identity_cols = [
        "feature_set",
        "model",
        "lookback_months",
        "horizon_months",
        "origin_id",
        "anchor_month",
        "target_month",
        "head",
    ]
    if not set(identity_cols).issubset(predictions.columns):
        return predictions
    out = predictions.copy()
    out["_row_order"] = np.arange(len(out))
    out = (
        out.sort_values("_row_order")
        .drop_duplicates(identity_cols, keep="last")
        .drop(columns=["_row_order"])
        .reset_index(drop=True)
    )
    return out


def deduplicate_folds(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return folds
    identity_cols = ["feature_set", "model", "lookback_months", "horizon_months", "origin_id"]
    if not set(identity_cols).issubset(folds.columns):
        return folds
    out = folds.copy()
    out["_row_order"] = np.arange(len(out))
    out = (
        out.sort_values("_row_order")
        .drop_duplicates(identity_cols, keep="last")
        .drop(columns=["_row_order"])
        .reset_index(drop=True)
    )
    return out


def clean_and_summarize(summary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return summary
    try:
        recomputed = summarize(predictions)
    except Exception:
        return summary
    if recomputed.empty:
        return summary
    return recomputed


def read_status(root: Path) -> pd.DataFrame:
    status_path = root / "part_status.tsv"
    if not status_path.exists():
        return pd.DataFrame(columns=STATUS_COLUMNS)
    try:
        return pd.read_csv(status_path, sep="\t", names=STATUS_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=STATUS_COLUMNS)


def latest_status(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty:
        return status
    tmp = status.copy()
    tmp["_order"] = np.arange(len(tmp))
    return tmp.sort_values("_order").groupby("model", as_index=False).tail(1).drop(columns=["_order"])


def add_rank_fields(summary: pd.DataFrame, folds: pd.DataFrame | None = None) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["r2_ok"] = out["r2_status"].astype(str).eq("ok") if "r2_status" in out.columns else False
    for col in SORT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    expected_cols = ["feature_set", "lookback_months", "horizon_months"]
    out["expected_predictions"] = np.nan
    if folds is not None and not folds.empty and set(expected_cols + ["model", "origin_id"]).issubset(folds.columns):
        fold_counts = (
            folds.groupby(expected_cols + ["model"], dropna=False)["origin_id"]
            .nunique()
            .reset_index(name="model_origin_count")
        )
        expected = (
            fold_counts.groupby(expected_cols, dropna=False)["model_origin_count"]
            .max()
            .reset_index(name="expected_predictions")
        )
        out = out.drop(columns=["expected_predictions"]).merge(expected, on=expected_cols, how="left")
    out["completion_ratio"] = np.where(
        out["expected_predictions"].notna() & (out["expected_predictions"] > 0),
        out["n_predictions"] / out["expected_predictions"],
        np.nan,
    )
    complete_mask = out["expected_predictions"].isna() | out["n_predictions"].eq(out["expected_predictions"])
    out["valid_ba"] = out["balanced_accuracy"].notna() & complete_mask
    return out


def sort_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        ["r2_ok", "valid_ba", "balanced_accuracy", "auc", "average_precision", "n_predictions"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )


def topn_by(df: pd.DataFrame, group_cols: list[str], n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        sort_metrics(df)
        .groupby(group_cols, group_keys=False, dropna=False)
        .head(n)
        .reset_index(drop=True)
    )


def write_report(out_dir: Path, summary: pd.DataFrame, status: pd.DataFrame) -> None:
    lines: list[str] = []
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines += ["# Long Lookback 2016-Fixed Live Report", "", f"- Updated: `{now}`"]
    if not status.empty:
        latest = latest_status(status)
        counts = latest["status"].value_counts().to_dict()
        lines.append(f"- Model status: `{counts}`")
        running = latest[latest["status"].eq("running")]["model"].tolist()
        if running:
            lines.append(f"- Running now: `{', '.join(running)}`")
    lines.append(f"- Summary rows aggregated: `{len(summary)}`")
    lines.append("")
    if summary.empty:
        (out_dir / "LIVE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    display_cols = [
        "feature_set",
        "horizon_months",
        "head",
        "family",
        "model",
        "lookback_months",
        "n_predictions",
        "expected_predictions",
        "completion_ratio",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "reg_price_r2",
        "r2_status",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]

    lines += ["## Overall Top 20", ""]
    lines.append(sort_metrics(summary)[display_cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## Top 10 By Horizon And Head", ""]
    lines.append(topn_by(summary, ["horizon_months", "head"], 10)[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## Top 10 By Family, Horizon And Head", ""]
    family_top = topn_by(summary, ["family", "horizon_months", "head"], 10)
    lines.append(family_top[display_cols].head(240).to_markdown(index=False, floatfmt=".4f"))

    abnormal = summary[summary.get("r2_status", pd.Series(index=summary.index, dtype=object)).astype(str).ne("ok")]
    if not abnormal.empty:
        lines += ["", "## R2 Diagnostic Non-OK", ""]
        lines.append(abnormal[display_cols].head(60).to_markdown(index=False, floatfmt=".4f"))

    (out_dir / "LIVE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "live"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_csvs(root, "summary_metrics.csv", "checkpoint_summary_metrics.csv")
    predictions = deduplicate_predictions(collect_csvs(root, "rolling_predictions.csv", "checkpoint_rolling_predictions.csv"))
    folds = deduplicate_folds(collect_csvs(root, "folds.csv", "checkpoint_folds.csv"))
    errors = collect_csvs(root, "model_errors.csv", "checkpoint_model_errors.csv")
    status = read_status(root)
    if errors.empty and len(errors.columns) == 0:
        errors = pd.DataFrame(columns=ERROR_COLUMNS)

    summary = clean_and_summarize(summary, predictions)
    summary = add_rank_fields(summary, folds)
    summary.to_csv(out_dir / "all_summary_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "all_rolling_predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(out_dir / "all_folds.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(out_dir / "all_model_errors.csv", index=False, encoding="utf-8-sig")
    latest_status(status).to_csv(out_dir / "latest_model_status.csv", index=False, encoding="utf-8-sig")

    if not summary.empty:
        sort_metrics(summary).to_csv(out_dir / "leaderboard_overall.csv", index=False, encoding="utf-8-sig")
        topn_by(summary, ["horizon_months", "head"], 10).to_csv(out_dir / "top10_by_horizon_head.csv", index=False, encoding="utf-8-sig")
        topn_by(summary, ["feature_set", "horizon_months", "head"], 10).to_csv(out_dir / "top10_by_feature_horizon_head.csv", index=False, encoding="utf-8-sig")
        topn_by(summary, ["family", "horizon_months", "head"], 10).to_csv(out_dir / "top10_by_family_horizon_head.csv", index=False, encoding="utf-8-sig")
        summary[summary["r2_status"].astype(str).ne("ok")].to_csv(out_dir / "r2_non_ok_rows.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, summary, status)
    print(out_dir / "LIVE_REPORT.md")


if __name__ == "__main__":
    main()
