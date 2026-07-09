from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel


TARGETS = {"5td": 5, "10td": 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PatchTST-FM LoRA adapter.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/PatchTST-FM"))
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def max_drawdown(equity: pd.Series) -> tuple[pd.Series, float]:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown, float(drawdown.min())


def summarize(result: pd.DataFrame, label: str) -> dict[str, object]:
    _, mdd = max_drawdown(result["equity"])
    up = result["predicted_return"] > 0
    down = result["predicted_return"] < 0
    return {
        "horizon": label,
        "predictions": len(result),
        "start_date": result["date"].min(),
        "end_date": result["date"].max(),
        "direction_win_rate": result["direction_correct"].mean(),
        "mae": mean_absolute_error(result["actual_return"], result["predicted_return"]),
        "rmse": mean_squared_error(result["actual_return"], result["predicted_return"]) ** 0.5,
        "mean_strategy_return_per_forecast": result["strategy_return"].mean(),
        "median_strategy_return_per_forecast": result["strategy_return"].median(),
        "final_equity_forecast_level": result["equity"].iloc[-1],
        "max_drawdown_forecast_level": mdd,
        "pred_up_rate": up.mean(),
        "up_signal_win_rate": result.loc[up, "direction_correct"].mean() if up.any() else np.nan,
        "down_signal_win_rate": result.loc[down, "direction_correct"].mean() if down.any() else np.nan,
    }


def plot_curves(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for label, result in outputs.items():
        axes[0].plot(pd.to_datetime(result["date"]), result["equity"], label=f"{label} strategy")
        axes[0].plot(pd.to_datetime(result["date"]), result["buy_hold_target_equity"], linestyle="--", alpha=0.6, label=f"{label} long-only target")
        axes[1].plot(pd.to_datetime(result["date"]), result["drawdown"], label=f"{label} drawdown")
    axes[0].set_title("PatchTST-FM LoRA equity")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "equity_drawdown.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(args.dataset_dir / "x.npy", mmap_mode="r")
    samples = pd.read_csv(args.dataset_dir / "samples.csv")
    idx = samples.index[samples["split"] == args.split].to_numpy()
    x_tensor = torch.tensor(np.asarray(x[idx]), dtype=torch.float32)
    sample_meta = samples.iloc[idx].reset_index(drop=True)

    with (args.dataset_dir / "preprocess.pkl").open("rb") as f:
        preprocess = pickle.load(f)
    scaler = preprocess["scaler"]
    close_mean = float(scaler.mean_[0])
    close_scale = float(scaler.scale_[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model.to(device)
    model.eval()

    preds = []
    loader = DataLoader(TensorDataset(x_tensor), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(past_values=xb, prediction_length=10)
            close_scaled = out.prediction_outputs[:, :, 0].detach().cpu().numpy()
            close_unscaled = close_scaled * close_scale + close_mean
            preds.append(close_unscaled)
    pred_close = np.concatenate(preds, axis=0)

    source = pd.read_csv(json.loads((args.dataset_dir / "config.json").read_text())["input"])
    source["date"] = pd.to_datetime(source["date"])
    rows = []
    for n, meta in sample_meta.iterrows():
        anchor_idx = int(meta["anchor_idx"])
        latest_close = float(meta["anchor_close"])
        for label, horizon in TARGETS.items():
            pred_ret = float(pred_close[n, horizon - 1] / latest_close - 1)
            target_col = f"target_corn_ret_fwd_{horizon}td"
            actual_ret = float(source.loc[anchor_idx, target_col])
            position = 1 if pred_ret > 0 else -1
            rows.append(
                {
                    "date": meta["anchor_date"],
                    "close": latest_close,
                    "prediction_horizon": horizon,
                    "predicted_close": float(pred_close[n, horizon - 1]),
                    "predicted_return": pred_ret,
                    "actual_return": actual_ret,
                    "predicted_direction": "UP" if pred_ret > 0 else "DOWN",
                    "actual_direction": "UP" if actual_ret > 0 else "DOWN",
                    "direction_correct": int(np.sign(pred_ret) == np.sign(actual_ret)),
                    "position": position,
                    "strategy_return": position * actual_ret,
                }
            )

    raw = pd.DataFrame(rows)
    outputs = {}
    summaries = []
    for label, horizon in TARGETS.items():
        result = raw[raw["prediction_horizon"] == horizon].copy().reset_index(drop=True)
        result["equity"] = (1 + result["strategy_return"]).cumprod()
        result["buy_hold_target_equity"] = (1 + result["actual_return"]).cumprod()
        result["drawdown"], _ = max_drawdown(result["equity"])
        outputs[label] = result
        summaries.append(summarize(result, label))
        result.to_csv(args.out_dir / f"test_predictions_{label}.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    plot_curves(outputs, args.out_dir)
    report = [
        "# PatchTST-FM LoRA Evaluation",
        "",
        f"Split: {args.split}. Device: {device}.",
        "",
        "## Summary",
        summary.to_string(index=False),
    ]
    (args.out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
