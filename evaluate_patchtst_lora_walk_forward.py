from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = {"5td": 5, "10td": 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic retrain walk-forward LoRA evaluation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/PatchTST-FM"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=756)
    parser.add_argument("--prediction-length", type=int, default=10)
    parser.add_argument("--retrain-every", type=int, default=126)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--valid-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


def max_drawdown(equity: pd.Series) -> tuple[pd.Series, float]:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown, float(drawdown.min())


def summarize(result: pd.DataFrame, label: str) -> dict[str, object]:
    _, mdd = max_drawdown(result["equity"])
    return {
        "horizon": label,
        "predictions": len(result),
        "start_date": result["date"].min(),
        "end_date": result["date"].max(),
        "direction_win_rate": result["direction_correct"].mean(),
        "mean_strategy_return_per_forecast": result["strategy_return"].mean(),
        "final_equity_forecast_level": result["equity"].iloc[-1],
        "max_drawdown_forecast_level": mdd,
        "pred_up_rate": (result["predicted_return"] > 0).mean(),
    }


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    last_idx = min(df["target_corn_ret_fwd_5td"].last_valid_index(), df["target_corn_ret_fwd_10td"].last_valid_index())
    first_test_idx = df.index[df["date"] >= pd.Timestamp("2024-01-01")][0]
    fold_starts = list(range(first_test_idx, last_idx + 1, args.retrain_every))
    if args.max_folds > 0:
        fold_starts = fold_starts[: args.max_folds]

    all_rows = []
    script_dir = Path(__file__).resolve().parent
    for fold_no, fold_start in enumerate(fold_starts, start=1):
        fold_end = min(fold_start + args.retrain_every - 1, last_idx)
        train_cut = fold_start - args.prediction_length
        subset_start = max(0, train_cut - args.context_length - args.train_samples - args.valid_samples - 5)
        subset_end = fold_end + args.prediction_length
        fold_df = df.iloc[subset_start : subset_end + 1].copy().reset_index(drop=True)
        fold_csv = args.work_dir / f"fold_{fold_no:03d}.csv"
        fold_df.to_csv(fold_csv, index=False)
        train_end_date = df.loc[max(0, train_cut - args.valid_samples), "date"].strftime("%Y-%m-%d")
        valid_end_date = df.loc[train_cut, "date"].strftime("%Y-%m-%d")

        dataset_dir = args.work_dir / f"fold_{fold_no:03d}_dataset"
        train_dir = args.work_dir / f"fold_{fold_no:03d}_train"
        eval_dir = args.work_dir / f"fold_{fold_no:03d}_eval"
        run(
            [
                "python3",
                str(script_dir / "prepare_patchtst_lora_dataset.py"),
                "--input",
                str(fold_csv),
                "--out-dir",
                str(dataset_dir),
                "--context-length",
                str(args.context_length),
                "--prediction-length",
                str(args.prediction_length),
                "--train-end-date",
                train_end_date,
                "--valid-end-date",
                valid_end_date,
            ]
        )
        run(
            [
                "python3",
                str(script_dir / "train_patchtst_lora.py"),
                "--dataset-dir",
                str(dataset_dir),
                "--model-dir",
                str(args.model_dir),
                "--out-dir",
                str(train_dir),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--max-train-samples",
                str(args.train_samples),
                "--max-valid-samples",
                str(args.valid_samples),
            ]
        )
        run(
            [
                "python3",
                str(script_dir / "evaluate_patchtst_lora.py"),
                "--dataset-dir",
                str(dataset_dir),
                "--model-dir",
                str(args.model_dir),
                "--adapter-dir",
                str(train_dir / "adapter"),
                "--out-dir",
                str(eval_dir),
                "--split",
                "test",
            ]
        )
        for label in TARGETS:
            pred_path = eval_dir / f"test_predictions_{label}.csv"
            if pred_path.exists():
                pred = pd.read_csv(pred_path)
                pred["fold"] = fold_no
                all_rows.append(pred)

    if not all_rows:
        raise ValueError("No walk-forward predictions were generated")
    raw = pd.concat(all_rows, ignore_index=True)
    raw = raw.drop_duplicates(subset=["date", "prediction_horizon"], keep="first")
    summaries = []
    for label, horizon in TARGETS.items():
        result = raw[raw["prediction_horizon"] == horizon].copy().reset_index(drop=True)
        result["equity"] = (1 + result["strategy_return"]).cumprod()
        result["drawdown"], _ = max_drawdown(result["equity"])
        result.to_csv(args.out_dir / f"walk_forward_predictions_{label}.csv", index=False)
        summaries.append(summarize(result, label))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "walk_forward_summary.csv", index=False)
    (args.out_dir / "walk_forward_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    (args.out_dir / "walk_forward_report.md").write_text(
        "# LoRA Periodic Walk-forward Evaluation\n\n" + summary.to_string(index=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
