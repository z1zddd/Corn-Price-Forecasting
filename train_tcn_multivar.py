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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ["target_corn_ret_fwd_5td", "target_corn_ret_fwd_10td"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN on multivariate corn futures windows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=756)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    return parser.parse_args()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int, levels: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = n_features
        for level in range(levels):
            layers.append(TemporalBlock(in_ch, channels, kernel_size=3, dilation=2**level, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: batch, time, features. Conv1d wants batch, features, time.
        x = x.transpose(1, 2)
        return self.head(self.tcn(x))


def split_name(anchor_date: pd.Timestamp) -> str:
    if anchor_date < pd.Timestamp("2023-01-01"):
        return "train"
    if anchor_date < pd.Timestamp("2024-01-01"):
        return "valid"
    return "test"


def build_dataset(df: pd.DataFrame, context_length: int):
    feature_cols = [c for c in df.columns if c not in ["date", *TARGETS]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all()]
    numeric_cols = ["dce_corn_close"] + [c for c in numeric_cols if c != "dce_corn_close"]

    anchors = []
    last_idx = min(df[TARGETS[0]].last_valid_index(), df[TARGETS[1]].last_valid_index())
    for i in range(context_length - 1, last_idx + 1):
        anchors.append({"anchor_idx": i, "date": df.loc[i, "date"], "split": split_name(df.loc[i, "date"])})
    meta = pd.DataFrame(anchors)

    train_anchor = meta.loc[meta["split"] == "train", "anchor_idx"].to_numpy()
    train_rows = sorted(set(row for anchor in train_anchor for row in range(anchor - context_length + 1, anchor + 1)))
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(df.loc[train_rows, numeric_cols])
    scaler.fit(imputer.transform(df.loc[train_rows, numeric_cols]))
    values = scaler.transform(imputer.transform(df[numeric_cols])).astype("float32")

    x = np.empty((len(meta), context_length, len(numeric_cols)), dtype="float32")
    y = np.empty((len(meta), 2), dtype="float32")
    for n, row in meta.iterrows():
        i = int(row["anchor_idx"])
        x[n] = values[i - context_length + 1 : i + 1]
        y[n] = df.loc[i, TARGETS].to_numpy(dtype="float32")
    return x, y, meta, numeric_cols


def max_drawdown(equity: pd.Series):
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown, float(drawdown.min())


def eval_predictions(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> pd.DataFrame:
    summaries = []
    for idx, label in enumerate(["5td", "10td"]):
        actual = y_true[:, idx]
        pred = y_pred[:, idx]
        position = np.where(pred > 0, 1, -1)
        result = pd.DataFrame(
            {
                "date": meta["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                "predicted_return": pred,
                "actual_return": actual,
                "predicted_direction": np.where(pred > 0, "UP", "DOWN"),
                "actual_direction": np.where(actual > 0, "UP", "DOWN"),
                "direction_correct": (np.sign(pred) == np.sign(actual)).astype(int),
                "position": position,
                "strategy_return": position * actual,
            }
        )
        result["equity"] = (1 + result["strategy_return"]).cumprod()
        result["drawdown"], mdd = max_drawdown(result["equity"])
        result.to_csv(out_dir / f"tcn_predictions_{label}.csv", index=False)
        up = pred > 0
        down = pred < 0
        summaries.append(
            {
                "horizon": label,
                "predictions": len(result),
                "start_date": result["date"].iloc[0],
                "end_date": result["date"].iloc[-1],
                "direction_win_rate": result["direction_correct"].mean(),
                "mae": mean_absolute_error(actual, pred),
                "rmse": mean_squared_error(actual, pred) ** 0.5,
                "mean_strategy_return_per_forecast": result["strategy_return"].mean(),
                "median_strategy_return_per_forecast": result["strategy_return"].median(),
                "final_equity_forecast_level": result["equity"].iloc[-1],
                "max_drawdown_forecast_level": mdd,
                "pred_up_rate": up.mean(),
                "up_signal_win_rate": result.loc[up, "direction_correct"].mean() if up.any() else np.nan,
                "down_signal_win_rate": result.loc[down, "direction_correct"].mean() if down.any() else np.nan,
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print("building window dataset", flush=True)

    x, y, meta, feature_cols = build_dataset(df, args.context_length)
    idx_train = meta.index[meta["split"] == "train"].to_numpy()
    idx_valid = meta.index[meta["split"] == "valid"].to_numpy()
    idx_test = meta.index[meta["split"] == "test"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dataset x={x.shape} y={y.shape} device={device}", flush=True)
    model = TCNRegressor(x.shape[2], args.channels, args.levels, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x[idx_train]), torch.tensor(y[idx_train])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_x = torch.tensor(x[idx_valid], device=device)
    valid_y = torch.tensor(y[idx_valid], device=device)
    print("starting training", flush=True)

    best_valid = float("inf")
    best_epoch = 0
    stale = 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y).detach().cpu())
        train_loss = float(np.mean(losses))
        logs.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}", flush=True)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), args.out_dir / "best_tcn.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    pd.DataFrame(logs).to_csv(args.out_dir / "train_log.csv", index=False)
    model.load_state_dict(torch.load(args.out_dir / "best_tcn.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        pred_test = model(torch.tensor(x[idx_test], device=device)).detach().cpu().numpy()

    summary = eval_predictions(meta.iloc[idx_test].reset_index(drop=True), y[idx_test], pred_test, args.out_dir)
    summary.to_csv(args.out_dir / "tcn_summary.csv", index=False)
    (args.out_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "epochs_requested": args.epochs,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid,
                "channels": args.channels,
                "levels": args.levels,
                "dropout": args.dropout,
                "feature_count": len(feature_cols),
                "device": str(device),
                "split_counts": meta["split"].value_counts().to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = ["# TCN Multivariate Forecast", "", "## Summary", summary.to_string(index=False)]
    (args.out_dir / "tcn_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
