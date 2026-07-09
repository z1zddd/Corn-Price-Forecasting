from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel


TARGETS = {"5td": ("target_corn_ret_fwd_5td", 5), "10td": ("target_corn_ret_fwd_10td", 10)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchTST-FM zero-shot multivariate walk-forward forecast.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/PatchTST-FM"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=756)
    parser.add_argument("--max-points", type=int, default=0, help="0 means all eligible points.")
    parser.add_argument("--step", type=int, default=1)
    return parser.parse_args()


def max_drawdown(equity: pd.Series) -> tuple[pd.Series, float]:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown, float(drawdown.min())


def summarize(result: pd.DataFrame, label: str) -> dict[str, object]:
    equity = result["equity"]
    _, mdd = max_drawdown(equity)
    up = result["predicted_return"] > 0
    down = result["predicted_return"] < 0
    return {
        "horizon": label,
        "predictions": len(result),
        "start_date": result["date"].min().date(),
        "end_date": result["date"].max().date(),
        "direction_win_rate": result["direction_correct"].mean(),
        "mae": mean_absolute_error(result["actual_return"], result["predicted_return"]),
        "rmse": mean_squared_error(result["actual_return"], result["predicted_return"]) ** 0.5,
        "mean_strategy_return_per_forecast": result["strategy_return"].mean(),
        "median_strategy_return_per_forecast": result["strategy_return"].median(),
        "final_equity_forecast_level": equity.iloc[-1],
        "max_drawdown_forecast_level": mdd,
        "pred_up_rate": up.mean(),
        "up_signal_win_rate": result.loc[up, "direction_correct"].mean() if up.any() else np.nan,
        "down_signal_win_rate": result.loc[down, "direction_correct"].mean() if down.any() else np.nan,
    }


def plot_curves(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for label, result in outputs.items():
        axes[0].plot(result["date"], result["equity"], label=f"{label} strategy")
        axes[0].plot(result["date"], result["buy_hold_target_equity"], linestyle="--", alpha=0.6, label=f"{label} long-only target")
        axes[1].plot(result["date"], result["drawdown"], label=f"{label} drawdown")

    axes[0].set_title("PatchTST-FM zero-shot forecast-level equity")
    axes[0].set_ylabel("Equity, compounded per forecast")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "patchtst_zero_shot_equity_drawdown.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["date", "target_corn_ret_fwd_5td", "target_corn_ret_fwd_10td"]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if "dce_corn_close" not in numeric_cols:
        raise ValueError("dce_corn_close must be available as a numeric feature")
    numeric_cols = ["dce_corn_close"] + [c for c in numeric_cols if c != "dce_corn_close" and not df[c].isna().all()]
    close_idx = 0

    last_test_idx = min(df[TARGETS["5td"][0]].last_valid_index(), df[TARGETS["10td"][0]].last_valid_index())
    eligible = list(range(args.context_length - 1, last_test_idx + 1, args.step))
    if args.max_points > 0:
        eligible = eligible[-args.max_points :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()

    all_rows = []
    with torch.no_grad():
        for n, i in enumerate(eligible, start=1):
            hist = df.iloc[i - args.context_length + 1 : i + 1]
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            hist_values = hist[numeric_cols]
            x_np = imputer.fit_transform(hist_values)
            x_np = scaler.fit_transform(x_np)
            x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

            out = model(past_values=x, prediction_length=10, quantile_levels=[0.1, 0.5, 0.9])
            pred_scaled = out.prediction_outputs.detach().cpu().numpy()[0]
            pred_unscaled = scaler.inverse_transform(pred_scaled)
            latest_close = float(df.loc[i, "dce_corn_close"])

            for label, (target, horizon) in TARGETS.items():
                pred_close = float(pred_unscaled[horizon - 1, close_idx])
                pred_ret = pred_close / latest_close - 1
                actual_ret = float(df.loc[i, target])
                position = 1 if pred_ret > 0 else -1
                all_rows.append(
                    {
                        "date": df.loc[i, "date"],
                        "close": latest_close,
                        "prediction_horizon": horizon,
                        "predicted_close": pred_close,
                        "predicted_return": pred_ret,
                        "actual_return": actual_ret,
                        "predicted_direction": "UP" if pred_ret > 0 else "DOWN",
                        "actual_direction": "UP" if actual_ret > 0 else "DOWN",
                        "direction_correct": int(np.sign(pred_ret) == np.sign(actual_ret)),
                        "position": position,
                        "strategy_return": position * actual_ret,
                    }
                )
            if n % 25 == 0:
                print(f"processed {n}/{len(eligible)} through {df.loc[i, 'date'].date()}", flush=True)

    raw = pd.DataFrame(all_rows)
    outputs = {}
    summaries = []
    for label, (_, horizon) in TARGETS.items():
        result = raw[raw["prediction_horizon"] == horizon].copy().reset_index(drop=True)
        result["equity"] = (1 + result["strategy_return"]).cumprod()
        result["buy_hold_target_equity"] = (1 + result["actual_return"]).cumprod()
        result["drawdown"], _ = max_drawdown(result["equity"])
        outputs[label] = result
        summaries.append(summarize(result, label))
        result.to_csv(args.out_dir / f"patchtst_zero_shot_predictions_{label}.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "patchtst_zero_shot_summary.csv", index=False)
    plot_curves(outputs, args.out_dir)

    config = {
        "model_dir": str(args.model_dir),
        "context_length": args.context_length,
        "step": args.step,
        "max_points": args.max_points,
        "feature_count": len(numeric_cols),
        "features": numeric_cols,
        "device": str(device),
    }
    (args.out_dir / "patchtst_zero_shot_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report = [
        "# PatchTST-FM Zero-shot Multivariate Forecast",
        "",
        f"Data range: {df['date'].min().date()} to {df['date'].max().date()}, rows: {len(df):,}.",
        f"Context length: {args.context_length}. Step: {args.step}. Device: {device}.",
        f"Feature count: {len(numeric_cols)}.",
        "",
        "## Summary",
        summary.to_string(index=False),
        "",
        "Outputs:",
        "- patchtst_zero_shot_predictions_5td.csv",
        "- patchtst_zero_shot_predictions_10td.csv",
        "- patchtst_zero_shot_summary.csv",
        "- patchtst_zero_shot_equity_drawdown.png",
        "- patchtst_zero_shot_config.json",
    ]
    (args.out_dir / "patchtst_zero_shot_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
