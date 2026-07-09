#!/usr/bin/env python3
"""Rerun the previous best hard-vote stream set on the new daily-derived data.

This intentionally launches only the streams used by the previous best
aggregate model, instead of expanding each model across all lookbacks and
horizons.  The new daily file has no precomputed news PCA columns, so this
rerun is a no-news data-processing check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OFFICIAL_SCRIPT = SCRIPT_DIR / "run_corn_monthly_spike_official_pool_two_heads.py"


@dataclass(frozen=True)
class Stream:
    model: str
    lookback_months: int
    horizon_months: int
    head: str
    family_note: str

    @property
    def feature_set(self) -> str:
        return "no_news"

    @property
    def stream_id(self) -> str:
        return f"{self.model}_nonews_lb{self.lookback_months}_h{self.horizon_months}_{self.head}"


H1_STREAMS = [
    Stream("mlp_small_relu", 6, 1, "cls", "old_h1_top6_member"),
    Stream("sgd_modified_huber", 6, 1, "cls", "old_h1_top6_member"),
    Stream("lightgbm_dart", 12, 1, "reg", "old_h1_top6_member"),
    Stream("keras_lstm_u16", 12, 1, "cls", "old_h1_top6_member"),
    Stream("keras_gru_u16", 12, 1, "reg", "old_h1_top6_member"),
]

H2_STREAMS = [
    Stream("aeon_knn_euclidean", 6, 2, "reg", "old_h2_top6_member"),
    Stream("knn_5_distance", 6, 2, "reg", "old_h2_top6_member"),
    Stream("aeon_deep_timecnn", 12, 2, "cls", "old_h2_top6_member"),
    Stream("knn_3_uniform", 12, 2, "cls", "old_h2_top6_member"),
    Stream("aeon_rise", 6, 2, "cls", "old_h2_top6_member"),
]

SKIPPED_OLD_STREAMS = [
    {
        "model": "keras_tcn_filters16_k2_d1",
        "feature_set": "no_news",
        "lookback_months": 9,
        "horizon_months": 1,
        "head": "reg",
        "reason": "keras-tcn package unavailable on rerun environment; no pretrained weights are involved.",
    },
    {
        "model": "aeon_deep_timecnn",
        "feature_set": "with_news_precomputed_pca",
        "lookback_months": 12,
        "horizon_months": 2,
        "head": "cls",
        "reason": "new daily source has no precomputed news PCA columns; using no_news only.",
    },
]

OLD_BEST = {
    "top6_h1_hard_vote_strict": {
        "balanced_accuracy": 0.765734,
        "auc": 0.821678,
        "average_precision": 0.742374,
        "accuracy": 0.763889,
        "n_predictions": 72,
    },
    "top6_h2_hard_vote_strict": {
        "balanced_accuracy": 0.712171,
        "auc": 0.794408,
        "average_precision": 0.832263,
        "accuracy": 0.714286,
        "n_predictions": 70,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--official-script", default=str(DEFAULT_OFFICIAL_SCRIPT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date-format", default="%Y-%m")
    parser.add_argument("--test-start", default="2019-12-01")
    parser.add_argument("--min-train", type=int, default=24)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--monthly-cutoff-lag", type=int, default=1)
    parser.add_argument("--deep-epochs", type=int, default=12)
    parser.add_argument("--deep-batch-size", type=int, default=16)
    parser.add_argument("--aeon-kernels", type=int, default=384)
    parser.add_argument("--aeon-estimators", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_stream(args: argparse.Namespace, stream: Stream, out_dir: Path) -> dict:
    stream_out = out_dir / "base_streams" / stream.stream_id
    log_dir = out_dir / "logs"
    stream_out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(stream_out.glob("*/rolling_predictions.csv"))
    if args.resume and existing:
        run_dir = existing[-1].parent
        return {"stream": asdict(stream), "status": "resumed", "run_dir": str(run_dir)}

    cmd = [
        sys.executable,
        args.official_script,
        "--csv",
        args.csv,
        "--out-dir",
        str(stream_out),
        "--date-col",
        "month",
        "--date-format",
        args.date_format,
        "--price-col",
        "dce_corn_close",
        "--label-col",
        "spike",
        "--lookbacks",
        str(stream.lookback_months),
        "--horizons",
        str(stream.horizon_months),
        "--heads",
        stream.head,
        "--feature-sets",
        stream.feature_set,
        "--label-mode",
        "direct_horizon_direction",
        "--min-train",
        str(args.min_train),
        "--val-size",
        str(args.val_size),
        "--test-size",
        "1",
        "--step-size",
        "1",
        "--origin-mode",
        "monthly",
        "--threshold-mode",
        "validation",
        "--test-start",
        args.test_start,
        "--monthly-cutoff-lag",
        str(args.monthly_cutoff_lag),
        "--models",
        stream.model,
        "--seed",
        "42",
        "--save-folds",
        "--checkpoint",
        "--aeon-kernels",
        str(args.aeon_kernels),
        "--aeon-estimators",
        str(args.aeon_estimators),
        "--deep-epochs",
        str(args.deep_epochs),
        "--deep-batch-size",
        str(args.deep_batch_size),
    ]
    log_path = log_dir / f"{stream.stream_id}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    run_dirs = sorted([p.parent for p in stream_out.glob("*/rolling_predictions.csv")], key=lambda p: p.stat().st_mtime)
    run_dir = run_dirs[-1] if run_dirs else None
    return {
        "stream": asdict(stream),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "run_dir": str(run_dir) if run_dir else "",
        "log": str(log_path),
    }


def read_csvs(run_records: list[dict], name: str) -> pd.DataFrame:
    frames = []
    for rec in run_records:
        if rec.get("status") not in {"ok", "resumed"} or not rec.get("run_dir"):
            continue
        path = Path(rec["run_dir"]) / name
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")


def ensemble_metrics(predictions: pd.DataFrame, name: str, streams: list[Stream]) -> tuple[pd.DataFrame, dict]:
    frames = []
    for stream in streams:
        mask = (
            predictions["model"].eq(stream.model)
            & predictions["feature_set"].eq(stream.feature_set)
            & predictions["lookback_months"].eq(stream.lookback_months)
            & predictions["horizon_months"].eq(stream.horizon_months)
            & predictions["head"].eq(stream.head)
        )
        part = predictions.loc[mask].copy()
        if part.empty:
            raise ValueError(f"missing stream: {stream.stream_id}")
        part["stream_id"] = stream.stream_id
        frames.append(part)

    aligned = pd.concat(frames, ignore_index=True)
    index_cols = ["anchor_month", "target_month", "actual_direction", "actual_return"]
    votes = aligned.pivot_table(index=index_cols, columns="stream_id", values="predicted_direction", aggfunc="first")
    probs = aligned.pivot_table(index=index_cols, columns="stream_id", values="predicted_probability", aggfunc="first")
    complete = votes.notna().sum(axis=1).eq(len(streams))
    dropped = int((~complete).sum())
    votes = votes.loc[complete]
    probs = probs.loc[complete]
    base = votes.reset_index()
    vote_score = votes.astype(float).mean(axis=1).to_numpy(float)
    prob_score = probs.astype(float).mean(axis=1).to_numpy(float)
    y_true = base["actual_direction"].to_numpy(int)
    pred = (vote_score > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = pd.DataFrame(
        {
            "ensemble": name,
            "anchor_month": base["anchor_month"],
            "target_month": base["target_month"],
            "actual_direction": y_true,
            "actual_return": base["actual_return"].to_numpy(float),
            "predicted_direction": pred,
            "vote_score": vote_score,
            "mean_probability": prob_score,
        }
    )
    metrics = {
        "ensemble": name,
        "horizon_months": int(streams[0].horizon_months),
        "candidate_count": len(streams),
        "dropped_incomplete_rows": dropped,
        "n_predictions": int(len(out)),
        "auc": safe_auc(y_true, vote_score),
        "average_precision": safe_ap(y_true, vote_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else float("nan"),
        "actual_positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "rule": "hard_vote_strict_gt_0.5",
    }
    return out, metrics


def write_report(out_dir: Path, audit: dict, run_records: list[dict], ensemble_metrics_rows: list[dict]) -> None:
    rows = []
    for row in ensemble_metrics_rows:
        old = OLD_BEST.get(row["ensemble"], {})
        rows.append(
            {
                "ensemble": row["ensemble"],
                "new_BA": row["balanced_accuracy"],
                "old_BA": old.get("balanced_accuracy"),
                "new_AUC": row["auc"],
                "old_AUC": old.get("auc"),
                "new_AP": row["average_precision"],
                "old_AP": old.get("average_precision"),
                "new_ACC": row["accuracy"],
                "old_ACC": old.get("accuracy"),
                "new_n": row["n_predictions"],
                "old_n": old.get("n_predictions"),
            }
        )
    comparison = pd.DataFrame(rows)
    stream_status = pd.DataFrame(run_records)
    lines = [
        "# Daily New Data Best-Mix Rerun",
        "",
        "## Data Audit",
        f"- source rows: {audit.get('source_rows')}",
        f"- monthly rows: {audit.get('monthly_rows')} ({audit.get('month_min')} to {audit.get('month_max')})",
        f"- monthly columns: {audit.get('monthly_columns')}",
        f"- excluded leak columns: {', '.join(audit.get('leak_columns_excluded', []))}",
        f"- target: {audit.get('target_definition')}",
        "",
        "## Scope",
        "- New daily source has no precomputed news PCA columns, so this rerun uses no_news only.",
        "- Previous TCN stream is skipped because keras-tcn was not available; these are package/code dependencies, not downloaded model weights.",
        "- Rolling settings: monthly test rows=1, min_train=24, val_rows=12, monthly_cutoff_lag=1, validation thresholds.",
        "",
        "## Ensemble Comparison",
        comparison.to_markdown(index=False),
        "",
        "## Skipped Old Streams",
        pd.DataFrame(SKIPPED_OLD_STREAMS).to_markdown(index=False),
        "",
        "## Stream Status",
        stream_status.to_markdown(index=False),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    comparison.to_csv(out_dir / "ensemble_vs_old_best_comparison.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "monthly_data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}

    run_records: list[dict] = []
    for stream in [*H1_STREAMS, *H2_STREAMS]:
        print(f"[stream] {stream.stream_id}", flush=True)
        rec = run_stream(args, stream, out_dir)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        run_records.append(rec)
        (out_dir / "stream_run_records.json").write_text(json.dumps(run_records, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions = read_csvs(run_records, "rolling_predictions.csv")
    summaries = read_csvs(run_records, "summary_metrics.csv")
    folds = read_csvs(run_records, "folds.csv")
    predictions.to_csv(out_dir / "all_selected_rolling_predictions.csv", index=False)
    summaries.to_csv(out_dir / "all_selected_summary_metrics.csv", index=False)
    if not folds.empty:
        folds.to_csv(out_dir / "all_selected_folds.csv", index=False)

    ensemble_predictions = []
    ensemble_rows = []
    for name, streams in [
        ("top6_h1_hard_vote_strict", H1_STREAMS),
        ("top6_h2_hard_vote_strict", H2_STREAMS),
    ]:
        pred, metrics = ensemble_metrics(predictions, name, streams)
        ensemble_predictions.append(pred)
        ensemble_rows.append(metrics)
    pd.concat(ensemble_predictions, ignore_index=True).to_csv(out_dir / "ensemble_selected_predictions.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(out_dir / "ensemble_selected_metrics.csv", index=False)
    write_report(out_dir, audit, run_records, ensemble_rows)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
