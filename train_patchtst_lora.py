from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchTST-FM LoRA adapter on corn windows.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/root/PatchTST-FM"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = np.load(dataset_dir / "x.npy", mmap_mode="r")
    y = np.load(dataset_dir / "y_close_scaled.npy", mmap_mode="r")
    import pandas as pd

    samples = pd.read_csv(dataset_dir / "samples.csv")
    idx = samples.index[samples["split"] == split].to_numpy()
    if max_samples > 0:
        idx = idx[-max_samples:]
    return torch.tensor(np.asarray(x[idx]), dtype=torch.float32), torch.tensor(np.asarray(y[idx]), dtype=torch.float32)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(past_values=xb, prediction_length=yb.shape[1], quantile_levels=[0.5])
            pred_close = out.quantile_outputs[:, 0, :, 0]
            loss = torch.nn.functional.mse_loss(pred_close, yb)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_split(args.dataset_dir, "train", args.max_train_samples)
    valid_x, valid_y = load_split(args.dataset_dir, "valid", args.max_valid_samples)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(str(args.model_dir), trust_remote_code=True, local_files_only=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["qkv", "proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "device": str(device),
    }
    (args.out_dir / "param_info.json").write_text(json.dumps(param_info, indent=2), encoding="utf-8")
    print(json.dumps(param_info, indent=2), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_valid = float("inf")
    best_epoch = 0
    stale = 0
    log_path = args.out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            losses = []
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(past_values=xb, prediction_length=yb.shape[1], quantile_levels=[0.5])
                pred_close = out.quantile_outputs[:, 0, :, 0]
                loss = torch.nn.functional.mse_loss(pred_close, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(losses))
            valid_loss = evaluate(model, valid_loader, device)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            f.flush()
            print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}", flush=True)

            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                stale = 0
                model.save_pretrained(args.out_dir / "adapter")
            else:
                stale += 1
                if stale >= args.patience:
                    break

    result = {"best_epoch": best_epoch, "best_valid_loss": best_valid, **param_info}
    (args.out_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
