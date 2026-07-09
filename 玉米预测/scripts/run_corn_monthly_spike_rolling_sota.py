#!/usr/bin/env python3
"""Rolling-origin spike classification for monthly corn prices.

The experiment predicts future monthly direction labels (`spike`) from the
previous 1-3 months and evaluates next 1-2 month horizons.  Each rolling origin
fits scalers, validation thresholds, classifiers, and diagnostic price
regressors on past data only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FUTURE_COL_KEYWORDS = ("next", "future", "target", "fwd", "lead")
DEFAULT_EXCLUDE_COLS = {
    "first_trade_date",
    "last_trade_date",
    "spike",
    "dce_corn_close_next_month",
    "dce_corn_close_next_month_ret",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    kind: str


@dataclass
class FoldRecord:
    lookback_months: int
    horizon_months: int
    origin_id: int
    cutoff_month: str
    train_rows: int
    val_rows: int
    test_rows: int
    model: str
    model_family: str
    threshold: float
    feature_scaler_rows: int
    target_scaler_rows: int


class TrainOnlyStandardizer:
    """Feature scaler fitted only on the rolling training fold."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TrainOnlyStandardizer":
        if x.ndim != 3:
            raise ValueError(f"Expected [N, T, F] array, got {x.shape}.")
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.mean_ = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        self.scale_ = np.where((scale < 1e-12) | ~np.isfinite(scale), 1.0, scale)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Feature scaler is not fitted.")
        z = (x.astype(np.float64) - self.mean_) / self.scale_
        z = np.where(np.isfinite(z), z, 0.0)
        return z.astype(np.float32)


class TargetStandardizer:
    """One-dimensional target scaler with explicit inverse transform."""

    def __init__(self) -> None:
        self.mean_: float = 0.0
        self.scale_: float = 1.0

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mean_ = float(np.nanmean(y))
        scale = float(np.nanstd(y))
        self.scale_ = scale if np.isfinite(scale) and scale >= 1e-12 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float64).reshape(-1) - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64).reshape(-1) * self.scale_ + self.mean_).astype(np.float64)


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-4, 1.0 - 1e-4))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.probability, dtype=np.float32)


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype=np.float32)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        bottleneck = min(max(out_channels, 4), 32)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(bottleneck, out_channels, kernel_size=k, padding="same", bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in (3, 5, 9)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        merged = out_channels * 4
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, merged, kernel_size=1, bias=False), nn.BatchNorm1d(merged))
            if in_channels != merged
            else nn.Identity()
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self.bottleneck(x)
        out = torch.cat([branch(xb) for branch in self.branches] + [self.pool_branch(x)], dim=1)
        return self.activation(out + self.shortcut(x))


class InceptionTimeTiny(nn.Module):
    """Compact InceptionTime-style network for very short monthly windows."""

    def __init__(self, n_features: int, hidden: int = 12, blocks: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = n_features
        for _ in range(blocks):
            layers.append(InceptionBlock(channels, hidden, dropout))
            channels = hidden * 4
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z = self.encoder(x.transpose(1, 2)).mean(dim=-1)
        return self.head(z).squeeze(-1)


class ModernTCNTiny(nn.Module):
    """ModernTCN-inspired depthwise + grouped pointwise convolution."""

    def __init__(self, n_features: int, hidden: int = 48, blocks: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(n_features, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Conv1d(hidden, hidden, kernel_size=1, groups=4 if hidden % 4 == 0 else 1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(blocks)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            z = z + block(z)
        pooled = torch.cat([z[:, :, -1], z.mean(dim=-1)], dim=-1)
        return self.head(pooled).squeeze(-1)


class TimeMixerTiny(nn.Module):
    """TimeMixer-inspired multiscale decomposition and feature mixing."""

    def __init__(self, n_features: int, seq_len: int, hidden: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.input_norm = nn.LayerNorm(n_features)
        # Raw, trend-smoothed, seasonal residual, and coarse last-observation summaries.
        in_dim = seq_len * n_features * 3 + n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        trend = torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        seasonal = x - trend
        coarse = x.mean(dim=1)
        mixed = torch.cat([x.flatten(1), trend.flatten(1), seasonal.flatten(1), coarse], dim=-1)
        return self.net(mixed).squeeze(-1)


class FITSTiny(nn.Module):
    """FITS-inspired lightweight frequency-domain classifier/regressor."""

    def __init__(self, n_features: int, seq_len: int, hidden: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        freq_bins = seq_len // 2 + 1
        in_dim = n_features * freq_bins * 2 + n_features
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        freq_features = torch.cat([spectrum.real.flatten(1), spectrum.imag.flatten(1), x[:, -1, :]], dim=-1)
        return self.net(freq_features).squeeze(-1)


class ITransformerTiny(nn.Module):
    """iTransformer-inspired encoder with variables as tokens."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 48,
        n_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, n_features, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 1))
        nn.init.normal_(self.variable_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.value_embedding(x.transpose(1, 2)) + self.variable_embedding
        z = self.encoder(z)
        pooled = torch.cat([z[:, -1, :], z.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)


class PatchTSTTiny(nn.Module):
    """PatchTST-inspired channel-independent patch Transformer."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 48,
        n_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_len = 2 if seq_len >= 2 else 1
        self.patch_count = max(1, seq_len - self.patch_len + 1)
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 1))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B*F, patches, patch_len]
        patches = x.transpose(1, 2).unfold(dimension=-1, size=self.patch_len, step=1)
        patches = patches.reshape(-1, self.patch_count, self.patch_len)
        z = self.patch_embedding(patches) + self.position[:, : patches.shape[1], :]
        z = self.encoder(z).reshape(x.shape[0], self.n_features, self.patch_count, -1)
        pooled = torch.cat([z[:, :, -1, :].mean(dim=1), z.mean(dim=(1, 2))], dim=-1)
        return self.head(pooled).squeeze(-1)


class TimeXerTiny(nn.Module):
    """TimeXer-inspired endogenous/exogenous cross-attention for exogenous features."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 48,
        n_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.endogenous_embedding = nn.Linear(seq_len, d_model)
        self.exogenous_embedding = nn.Linear(seq_len, d_model)
        self.exogenous_projection = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        endogenous = self.endogenous_embedding(x[:, :, -1].unsqueeze(1))
        if self.n_features > 1:
            exogenous = self.exogenous_embedding(x[:, :, :-1].transpose(1, 2))
            exogenous = self.encoder(self.exogenous_projection(exogenous))
            cross, _ = self.cross_attention(endogenous, exogenous, exogenous, need_weights=False)
            pooled_token = (endogenous + cross).squeeze(1)
            exo_summary = exogenous.mean(dim=1)
        else:
            pooled_token = endogenous.squeeze(1)
            exo_summary = pooled_token
        pooled = torch.cat([pooled_token, exo_summary], dim=-1)
        return self.head(pooled).squeeze(-1)


class TorchSequenceModel:
    def __init__(
        self,
        network_factory: Callable[[int, int], nn.Module],
        task: str,
        seed: int,
        epochs: int,
        patience: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        device: str,
    ) -> None:
        self.network_factory = network_factory
        self.task = task
        self.seed = seed
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.model: nn.Module | None = None
        self.history: list[dict[str, float]] = []

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "TorchSequenceModel":
        set_seed(self.seed)
        self.model = self.network_factory(x_train.shape[-1], x_train.shape[1]).to(self.device)
        train_y = np.asarray(y_train, dtype=np.float32).reshape(-1)
        val_y = np.asarray(y_val, dtype=np.float32).reshape(-1)
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(train_y, dtype=torch.float32)),
            batch_size=min(self.batch_size, max(1, len(x_train))),
            shuffle=True,
            drop_last=False,
        )
        val_x_t = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        val_y_t = torch.as_tensor(val_y, dtype=torch.float32, device=self.device)
        if self.task == "classification":
            positives = float(np.sum(train_y == 1.0))
            negatives = float(np.sum(train_y == 0.0))
            pos_weight = torch.as_tensor([max(negatives / positives, 1e-3) if positives > 0 else 1.0], device=self.device)
            loss_fn: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        best_loss = float("inf")
        stale = 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            losses: list[float] = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(val_x_t), val_y_t).detach().cpu())
            self.history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_loss": val_loss})
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def _raw_predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Torch model is not fitted.")
        self.model.eval()
        out: list[np.ndarray] = []
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for start in range(0, len(x_t), self.batch_size):
                out.append(self.model(x_t[start : start + self.batch_size]).detach().cpu().numpy())
        return np.concatenate(out).reshape(-1).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._raw_predict(x)
        return sigmoid(logits).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._raw_predict(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Monthly feature CSV.")
    parser.add_argument("--out-dir", default="outputs/corn_monthly_spike_rolling_sota")
    parser.add_argument("--date-col", default="month")
    parser.add_argument("--date-format", default="%y-%b")
    parser.add_argument("--price-col", default="dce_corn_close")
    parser.add_argument("--label-col", default="spike")
    parser.add_argument("--lookbacks", default="1,2,3", help="Comma-separated lookback months.")
    parser.add_argument("--horizons", default="1,2", help="Comma-separated horizon months.")
    parser.add_argument(
        "--label-mode",
        choices=("existing_spike", "direct_horizon_direction"),
        default="existing_spike",
        help="existing_spike uses the supplied label at target month; direct_horizon_direction creates price[t+h] > price[t].",
    )
    parser.add_argument("--min-train", type=int, default=48)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=3)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--max-origins", type=int, default=0, help="0 means all origins; useful for smoke runs.")
    parser.add_argument(
        "--models",
        default="modern_tcn,timemixer,fits,lightgbm,xgboost,catboost,extra_trees,patchtst,itransformer,timexer",
        help="Comma-separated model names. Optional extras: inceptiontime,tabpfn.",
    )
    parser.add_argument("--include-pca", action="store_true", help="Include pca_* columns. Default excludes them to avoid upstream PCA leakage.")
    parser.add_argument("--extra-exclude", default="", help="Additional comma-separated feature columns to exclude.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-folds", action="store_true", help="Save per-origin fold metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_monthly_csv(csv_path, args.date_col, args.date_format)
    feature_cols, excluded_cols = select_feature_columns(
        df=df,
        date_col=args.date_col,
        price_col=args.price_col,
        label_col=args.label_col,
        include_pca=args.include_pca,
        extra_exclude=parse_csv_list(args.extra_exclude),
    )
    feature_cols = move_price_last(feature_cols, args.price_col)
    model_specs = resolve_model_specs(parse_csv_list(args.models))
    lookbacks = [int(x) for x in parse_csv_list(args.lookbacks)]
    horizons = [int(x) for x in parse_csv_list(args.horizons)]

    manifest = {
        "run_id": run_id,
        "csv": str(csv_path),
        "rows": int(len(df)),
        "date_min": str(df[args.date_col].min().date()),
        "date_max": str(df[args.date_col].max().date()),
        "price_col": args.price_col,
        "label_col": args.label_col,
        "label_mode": args.label_mode,
        "lookbacks": lookbacks,
        "horizons": horizons,
        "min_train": args.min_train,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "models": [asdict(spec) for spec in model_specs],
        "sota_references": sota_references(),
        "device": device,
        "include_pca": bool(args.include_pca),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "excluded_cols": excluded_cols,
        "leakage_controls": [
            "future/target/next/lead columns excluded from features",
            "rolling training labels restricted to target_idx <= cutoff anchor index",
            "validation threshold selected on past validation fold only",
            "feature standardizer fitted on train fold only",
            "price target standardizer fitted on train fold only and inverse-transformed before R2",
            "pca_* columns excluded by default because upstream full-sample PCA cannot be audited from this CSV",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_predictions: list[dict[str, object]] = []
    all_folds: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for lookback in lookbacks:
        for horizon in horizons:
            samples = make_samples(
                df=df,
                feature_cols=feature_cols,
                date_col=args.date_col,
                price_col=args.price_col,
                label_col=args.label_col,
                lookback=lookback,
                horizon=horizon,
                label_mode=args.label_mode,
            )
            origins = make_rolling_origins(
                samples=samples,
                min_train=args.min_train,
                val_size=args.val_size,
                test_size=args.test_size,
                step_size=args.step_size,
                max_origins=args.max_origins,
            )
            if not origins:
                print(f"[skip] lookback={lookback} horizon={horizon}: no rolling origins", flush=True)
                continue
            for model_spec in model_specs:
                print(
                    f"[run] lookback={lookback} horizon={horizon} model={model_spec.name} origins={len(origins)}",
                    flush=True,
                )
                combo_predictions, combo_folds = run_combo(
                    samples=samples,
                    origins=origins,
                    model_spec=model_spec,
                    seed=args.seed,
                    device=device,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
                all_predictions.extend(combo_predictions)
                all_folds.extend(combo_folds)
                summary_rows.append(summarize_predictions(pd.DataFrame(combo_predictions)))

    predictions_df = pd.DataFrame(all_predictions)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["horizon_months", "lookback_months", "balanced_accuracy", "auc", "average_precision"],
        ascending=[True, True, False, False, False],
    )
    predictions_df.to_csv(out_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    if args.save_folds:
        pd.DataFrame(all_folds).to_csv(out_dir / "folds.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, summary_df, manifest)
    print("\n=== Summary ===")
    cols = [
        "model",
        "lookback_months",
        "horizon_months",
        "n_predictions",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "r2_price",
        "r2_status",
    ]
    print(summary_df[cols].to_string(index=False))
    print(f"\nSaved: {out_dir}")


def load_monthly_csv(csv_path: Path, date_col: str, date_format: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format=date_format, errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        bad = df.loc[df[date_col].isna(), date_col].head(5).tolist()
        raise ValueError(f"Could not parse date values: {bad}")
    return df.sort_values(date_col).reset_index(drop=True)


def select_feature_columns(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    label_col: str,
    include_pca: bool,
    extra_exclude: list[str],
) -> tuple[list[str], list[str]]:
    excluded = set(DEFAULT_EXCLUDE_COLS) | {date_col, label_col} | set(extra_exclude)
    selected: list[str] = []
    excluded_cols: list[str] = []
    for col in df.columns:
        lower = col.lower()
        should_exclude = (
            col in excluded
            or any(keyword in lower for keyword in FUTURE_COL_KEYWORDS)
            or (not include_pca and lower.startswith("pca_"))
            or not pd.api.types.is_numeric_dtype(df[col])
        )
        if should_exclude:
            excluded_cols.append(col)
            continue
        selected.append(col)
    if price_col not in selected:
        raise ValueError(f"price_col={price_col!r} must be an eligible numeric feature.")
    if not selected:
        raise ValueError("No numeric feature columns selected.")
    return selected, excluded_cols


def move_price_last(feature_cols: list[str], price_col: str) -> list[str]:
    return [col for col in feature_cols if col != price_col] + [price_col]


def make_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str,
    price_col: str,
    label_col: str,
    lookback: int,
    horizon: int,
    label_mode: str,
) -> dict[str, np.ndarray | pd.DataFrame]:
    features = df[feature_cols].to_numpy(dtype=np.float32)
    prices = df[price_col].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy(dtype=np.int64)
    dates = pd.to_datetime(df[date_col])
    x_rows: list[np.ndarray] = []
    y_cls: list[int] = []
    y_price: list[float] = []
    meta_rows: list[dict[str, object]] = []
    for anchor_idx in range(lookback - 1, len(df) - horizon):
        start = anchor_idx - lookback + 1
        target_idx = anchor_idx + horizon
        window = features[start : anchor_idx + 1]
        if label_mode == "existing_spike":
            label = int(labels[target_idx])
            target_return = prices[target_idx] / prices[target_idx - 1] - 1.0 if target_idx > 0 else np.nan
        elif label_mode == "direct_horizon_direction":
            target_return = prices[target_idx] / prices[anchor_idx] - 1.0
            label = int(target_return > 0.0)
        else:
            raise ValueError(f"Unknown label_mode={label_mode}")
        if not np.isfinite(window).any() or not np.isfinite(prices[target_idx]):
            continue
        x_rows.append(window)
        y_cls.append(label)
        y_price.append(float(prices[target_idx]))
        meta_rows.append(
            {
                "sample_id": len(x_rows) - 1,
                "anchor_idx": anchor_idx,
                "target_idx": target_idx,
                "anchor_month": str(dates.iloc[anchor_idx].date()),
                "target_month": str(dates.iloc[target_idx].date()),
                "anchor_price": float(prices[anchor_idx]),
                "target_price": float(prices[target_idx]),
                "target_return": float(target_return),
            }
        )
    if not x_rows:
        raise ValueError("No valid samples generated.")
    return {
        "X": np.stack(x_rows).astype(np.float32),
        "y_cls": np.asarray(y_cls, dtype=np.int64),
        "y_price": np.asarray(y_price, dtype=np.float64),
        "meta": pd.DataFrame(meta_rows),
        "lookback": np.asarray([lookback], dtype=np.int64),
        "horizon": np.asarray([horizon], dtype=np.int64),
    }


def make_rolling_origins(
    samples: dict[str, np.ndarray | pd.DataFrame],
    min_train: int,
    val_size: int,
    test_size: int,
    step_size: int,
    max_origins: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    meta = samples["meta"]
    assert isinstance(meta, pd.DataFrame)
    origins: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    n = len(meta)
    cursor = 0
    origin_id = 0
    while cursor < n:
        cutoff_anchor_idx = int(meta.iloc[cursor]["anchor_idx"])
        trainval_idx = meta.index[meta["target_idx"] <= cutoff_anchor_idx].to_numpy(dtype=int)
        if len(trainval_idx) >= min_train + val_size:
            future_mask = meta["anchor_idx"] > cutoff_anchor_idx
            candidate_test = meta.index[future_mask].to_numpy(dtype=int)[:test_size]
            if len(candidate_test) > 0:
                train_idx = trainval_idx[: -val_size]
                val_idx = trainval_idx[-val_size:]
                origins.append((train_idx, val_idx, candidate_test, origin_id))
                origin_id += 1
                if max_origins and len(origins) >= max_origins:
                    break
                cursor = int(candidate_test[-1]) + step_size
                continue
        cursor += 1
    return origins


def run_combo(
    samples: dict[str, np.ndarray | pd.DataFrame],
    origins: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    model_spec: ModelSpec,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    x = samples["X"]
    y_cls = samples["y_cls"]
    y_price = samples["y_price"]
    meta = samples["meta"]
    lookback = int(np.asarray(samples["lookback"])[0])
    horizon = int(np.asarray(samples["horizon"])[0])
    assert isinstance(x, np.ndarray)
    assert isinstance(y_cls, np.ndarray)
    assert isinstance(y_price, np.ndarray)
    assert isinstance(meta, pd.DataFrame)

    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for train_idx, val_idx, test_idx, origin_id in origins:
        fold_seed = seed + 1000 * lookback + 100 * horizon + 17 * origin_id + stable_name_offset(model_spec.name)
        scaler = TrainOnlyStandardizer().fit(x[train_idx])
        x_train = scaler.transform(x[train_idx])
        x_val = scaler.transform(x[val_idx])
        x_test = scaler.transform(x[test_idx])

        classifier = fit_classifier(
            model_spec=model_spec,
            x_train=x_train,
            y_train=y_cls[train_idx],
            x_val=x_val,
            y_val=y_cls[val_idx],
            seed=fold_seed,
            device=device,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
        )
        val_prob = classifier.predict_proba(x_val)
        threshold = select_threshold(y_cls[val_idx], val_prob)
        test_prob = classifier.predict_proba(x_test)
        test_pred = (test_prob >= threshold).astype(int)

        y_scaler = TargetStandardizer().fit(y_price[train_idx])
        regressor = fit_regressor(
            model_spec=model_spec,
            x_train=x_train,
            y_train_scaled=y_scaler.transform(y_price[train_idx]),
            x_val=x_val,
            y_val_scaled=y_scaler.transform(y_price[val_idx]),
            seed=fold_seed + 5,
            device=device,
            epochs=max(10, epochs),
            patience=patience,
            batch_size=batch_size,
        )
        pred_price_scaled = regressor.predict(x_test)
        pred_price = y_scaler.inverse_transform(pred_price_scaled)

        cutoff_row = meta.iloc[val_idx[-1]]
        fold = FoldRecord(
            lookback_months=lookback,
            horizon_months=horizon,
            origin_id=origin_id,
            cutoff_month=str(cutoff_row["target_month"]),
            train_rows=len(train_idx),
            val_rows=len(val_idx),
            test_rows=len(test_idx),
            model=model_spec.name,
            model_family=model_spec.family,
            threshold=float(threshold),
            feature_scaler_rows=len(train_idx),
            target_scaler_rows=len(train_idx),
        )
        folds.append(asdict(fold))

        for row_offset, sample_idx in enumerate(test_idx):
            row = meta.iloc[int(sample_idx)]
            predictions.append(
                {
                    "model": model_spec.name,
                    "model_family": model_spec.family,
                    "lookback_months": lookback,
                    "horizon_months": horizon,
                    "origin_id": origin_id,
                    "threshold": float(threshold),
                    "anchor_month": row["anchor_month"],
                    "target_month": row["target_month"],
                    "anchor_idx": int(row["anchor_idx"]),
                    "target_idx": int(row["target_idx"]),
                    "anchor_price": float(row["anchor_price"]),
                    "actual_price": float(row["target_price"]),
                    "predicted_price": float(pred_price[row_offset]),
                    "actual_return": float(row["target_return"]),
                    "actual_spike": int(y_cls[int(sample_idx)]),
                    "predicted_spike": int(test_pred[row_offset]),
                    "predicted_probability": float(test_prob[row_offset]),
                }
            )
    return predictions, folds


def fit_classifier(
    model_spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
):
    if len(np.unique(y_train)) < 2:
        return ConstantProbabilityModel(float(np.mean(y_train)))
    if model_spec.kind == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(flatten(x_train), y_train.astype(int))
        return SklearnClassifierAdapter(model)
    if model_spec.kind == "xgboost":
        from xgboost import XGBClassifier

        neg = max(1, int(np.sum(y_train == 0)))
        pos = max(1, int(np.sum(y_train == 1)))
        model = XGBClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(flatten(x_train), y_train.astype(int))
        return SklearnClassifierAdapter(model)
    if model_spec.kind == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=180,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(flatten(x_train), y_train.astype(int))
        return SklearnClassifierAdapter(model)
    if model_spec.kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(flatten(x_train), y_train.astype(int))
        return SklearnClassifierAdapter(model)
    if model_spec.kind == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("tabpfn is optional; install `tabpfn` or remove it from --models.") from exc

        model = TabPFNClassifier(device=device if device == "cuda" else "cpu")
        model.fit(flatten(x_train), y_train.astype(int))
        return SklearnClassifierAdapter(model)
    if model_spec.kind == "inceptiontime":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: InceptionTimeTiny(n_features=n_features),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    if model_spec.kind == "modern_tcn":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: ModernTCNTiny(n_features=n_features),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    if model_spec.kind == "timemixer":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: TimeMixerTiny(n_features=n_features, seq_len=seq_len),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    if model_spec.kind == "fits":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: FITSTiny(n_features=n_features, seq_len=seq_len),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    if model_spec.kind == "patchtst":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: PatchTSTTiny(n_features=n_features, seq_len=seq_len),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    if model_spec.kind == "itransformer":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: ITransformerTiny(n_features=n_features, seq_len=seq_len),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    if model_spec.kind == "timexer":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: TimeXerTiny(n_features=n_features, seq_len=seq_len),
            task="classification",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train, x_val, y_val)
    raise ValueError(f"Unsupported classifier kind: {model_spec.kind}")


def fit_regressor(
    model_spec: ModelSpec,
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    x_val: np.ndarray,
    y_val_scaled: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
):
    if np.nanstd(y_train_scaled) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y_train_scaled)))
    if model_spec.kind == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(flatten(x_train), y_train_scaled)
        return SklearnRegressorAdapter(model)
    if model_spec.kind == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(flatten(x_train), y_train_scaled)
        return SklearnRegressorAdapter(model)
    if model_spec.kind == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=180,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        model.fit(flatten(x_train), y_train_scaled)
        return SklearnRegressorAdapter(model)
    if model_spec.kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(flatten(x_train), y_train_scaled)
        return SklearnRegressorAdapter(model)
    if model_spec.kind == "tabpfn":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(flatten(x_train), y_train_scaled)
        return SklearnRegressorAdapter(model)
    if model_spec.kind == "inceptiontime":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: InceptionTimeTiny(n_features=n_features),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    if model_spec.kind == "modern_tcn":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: ModernTCNTiny(n_features=n_features),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    if model_spec.kind == "timemixer":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: TimeMixerTiny(n_features=n_features, seq_len=seq_len),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    if model_spec.kind == "fits":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: FITSTiny(n_features=n_features, seq_len=seq_len),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    if model_spec.kind == "patchtst":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: PatchTSTTiny(n_features=n_features, seq_len=seq_len),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    if model_spec.kind == "itransformer":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: ITransformerTiny(n_features=n_features, seq_len=seq_len),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    if model_spec.kind == "timexer":
        return TorchSequenceModel(
            network_factory=lambda n_features, seq_len: TimeXerTiny(n_features=n_features, seq_len=seq_len),
            task="regression",
            seed=seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=5e-4,
            weight_decay=1e-4,
            device=device,
        ).fit(x_train, y_train_scaled, x_val, y_val_scaled)
    raise ValueError(f"Unsupported regressor kind: {model_spec.kind}")


class SklearnClassifierAdapter:
    def __init__(self, model) -> None:
        self.model = model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(flatten(x))
        return proba[:, 1].astype(np.float32)


class SklearnRegressorAdapter:
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(flatten(x)).astype(np.float32)


def select_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    y_val = np.asarray(y_val, dtype=int).reshape(-1)
    prob_val = np.asarray(prob_val, dtype=float).reshape(-1)
    if len(np.unique(y_val)) < 2:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.2, 0.8, 61):
        pred = (prob_val >= threshold).astype(int)
        score = balanced_accuracy_score(y_val, pred)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def summarize_predictions(df: pd.DataFrame) -> dict[str, object]:
    y = df["actual_spike"].to_numpy(dtype=int)
    pred = df["predicted_spike"].to_numpy(dtype=int)
    prob = df["predicted_probability"].to_numpy(dtype=float)
    actual_price = df["actual_price"].to_numpy(dtype=float)
    predicted_price = df["predicted_price"].to_numpy(dtype=float)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    r2 = safe_r2(actual_price, predicted_price)
    row = {
        "model": str(df["model"].iloc[0]),
        "model_family": str(df["model_family"].iloc[0]),
        "lookback_months": int(df["lookback_months"].iloc[0]),
        "horizon_months": int(df["horizon_months"].iloc[0]),
        "n_predictions": int(len(df)),
        "class_0_count": int((y == 0).sum()),
        "class_1_count": int((y == 1).sum()),
        "predicted_positive_rate": float(np.mean(pred == 1)),
        "actual_positive_rate": float(np.mean(y == 1)),
        "auc": safe_auc(y, prob),
        "average_precision": safe_ap(y, prob),
        "balanced_accuracy": safe_balanced_accuracy(y, pred),
        "accuracy": float(accuracy_score(y, pred)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "r2_price": r2,
        "r2_status": r2_status(r2),
        "price_rmse": float(math.sqrt(mean_squared_error(actual_price, predicted_price))),
        "price_mae": float(mean_absolute_error(actual_price, predicted_price)),
        "probability_std": float(np.nanstd(prob)),
    }
    return row


def write_report(out_dir: Path, summary_df: pd.DataFrame, manifest: dict[str, object]) -> None:
    lines = [
        "# Corn Monthly Spike Rolling SOTA Backtest",
        "",
        f"- CSV: `{manifest['csv']}`",
        f"- Date range: `{manifest['date_min']}` to `{manifest['date_max']}`",
        f"- Label mode: `{manifest['label_mode']}`",
        f"- Features: `{manifest['feature_count']}` numeric columns; include_pca=`{manifest['include_pca']}`",
        f"- Rolling protocol: min_train=`{manifest['min_train']}`, val_size=`{manifest['val_size']}`, test_size=`{manifest['test_size']}`, step_size=`{manifest['step_size']}`",
        "",
        "## Leakage Controls",
        "",
    ]
    for item in manifest["leakage_controls"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Model Source Notes", ""])
    references = manifest.get("sota_references", {})
    if isinstance(references, dict):
        for name, reference in references.items():
            lines.append(f"- `{name}`: {reference}")
    lines.extend(
        [
            "",
            "## R2 Diagnostic Rule",
            "",
            "- `r2_price < -0.1`: abnormal",
            "- `-0.1 <= r2_price < 0`: likely abnormal",
            "- `r2_price >= 0`: ok",
            "",
            "## Summary Metrics",
            "",
        ]
    )
    display_cols = [
        "model",
        "lookback_months",
        "horizon_months",
        "n_predictions",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "r2_price",
        "r2_status",
    ]
    lines.append(summary_df[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "Outputs:",
            "",
            "- `summary_metrics.csv`",
            "- `rolling_predictions.csv`",
            "- `manifest.json`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_model_specs(names: list[str]) -> list[ModelSpec]:
    specs = {
        "modern_tcn": ModelSpec("modern_tcn", "non_transformer_short_sequence_dl_sota", "modern_tcn"),
        "timemixer": ModelSpec("timemixer", "non_transformer_short_sequence_dl_sota", "timemixer"),
        "fits": ModelSpec("fits", "non_transformer_short_sequence_dl_sota", "fits"),
        "inceptiontime": ModelSpec("inceptiontime", "non_transformer_short_sequence_dl_sota", "inceptiontime"),
        "lightgbm": ModelSpec("lightgbm", "ml_sota_gradient_boosted_trees", "lightgbm"),
        "xgboost": ModelSpec("xgboost", "ml_sota_gradient_boosted_trees", "xgboost"),
        "catboost": ModelSpec("catboost", "ml_sota_gradient_boosted_trees", "catboost"),
        "extra_trees": ModelSpec("extra_trees", "ml_sota_tree_ensemble", "extra_trees"),
        "tabpfn": ModelSpec("tabpfn", "ml_tabular_foundation_model_optional", "tabpfn"),
        "patchtst": ModelSpec("patchtst", "new_wave_transformer_short_sequence_sota", "patchtst"),
        "itransformer": ModelSpec("itransformer", "new_wave_transformer_short_sequence_sota", "itransformer"),
        "timexer": ModelSpec("timexer", "new_wave_transformer_short_sequence_sota", "timexer"),
    }
    resolved: list[ModelSpec] = []
    for name in names:
        if name not in specs:
            raise ValueError(f"Unknown model {name!r}; choose from {sorted(specs)}")
        resolved.append(specs[name])
    return resolved


def sota_references() -> dict[str, str]:
    return {
        "modern_tcn": "ModernTCN, ICLR 2024: pure convolution time-series model using depthwise/grouped pointwise convolution. https://openreview.net/forum?id=vpJMJerXHU",
        "timemixer": "TimeMixer, ICLR 2024: fully MLP-based decomposable multiscale mixing for long- and short-term forecasting. https://openreview.net/forum?id=7oLshfEIC2",
        "fits": "FITS, ICLR 2024: lightweight frequency interpolation for time-series modeling. https://openreview.net/forum?id=bWcnvZ3qMb",
        "lightgbm": "LightGBM, NeurIPS 2017: efficient gradient boosted decision trees; still a strong tabular baseline. https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
        "xgboost": "XGBoost-style histogram GBDT: strong tabular ML baseline.",
        "catboost": "CatBoost, NeurIPS 2018: ordered boosting GBDT; strong tabular baseline. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
        "extra_trees": "ExtraTrees: variance-reduced randomized tree ensemble baseline.",
        "tabpfn": "TabPFN v2, Nature 2025: tabular foundation model for small data; optional dependency. https://www.nature.com/articles/s41586-024-08328-6",
        "patchtst": "PatchTST, ICLR 2023: channel-independent patch Transformer for time series. https://openreview.net/forum?id=Jbdc0vTOcol",
        "itransformer": "iTransformer, ICLR 2024: inverted Transformer with variates as tokens. https://openreview.net/forum?id=JePfAI8fah",
        "timexer": "TimeXer, NeurIPS 2024: Transformer design for exogenous-variable forecasting. https://openreview.net/forum?id=INAeUQ04lT",
    }


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def flatten(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], -1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, prob: np.ndarray) -> float:
    return float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def safe_balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.nanstd(y_true) > 1e-12 else float("nan")


def r2_status(r2: float) -> str:
    if not np.isfinite(r2):
        return "undefined"
    if r2 < -0.1:
        return "abnormal"
    if r2 < 0:
        return "likely_abnormal"
    return "ok"


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_name_offset(name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name)) % 997


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
