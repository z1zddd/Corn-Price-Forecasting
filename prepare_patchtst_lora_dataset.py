from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multivariate PatchTST LoRA window dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=756)
    parser.add_argument("--prediction-length", type=int, default=10)
    parser.add_argument("--smoke-samples", type=int, default=0)
    parser.add_argument("--train-end-date", default="2022-12-31")
    parser.add_argument("--valid-end-date", default="2023-12-31")
    return parser.parse_args()


def split_name(anchor_date: pd.Timestamp, train_end_date: pd.Timestamp, valid_end_date: pd.Timestamp) -> str:
    if anchor_date <= train_end_date:
        return "train"
    if anchor_date <= valid_end_date:
        return "valid"
    return "test"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ["date", "target_corn_ret_fwd_5td", "target_corn_ret_fwd_10td"]]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not df[c].isna().all()]
    numeric_cols = ["dce_corn_close"] + [c for c in numeric_cols if c != "dce_corn_close"]
    close_idx = 0

    anchors = []
    train_end_date = pd.Timestamp(args.train_end_date)
    valid_end_date = pd.Timestamp(args.valid_end_date)
    for i in range(args.context_length - 1, len(df) - args.prediction_length):
        anchors.append(
            {
                "anchor_idx": i,
                "anchor_date": df.loc[i, "date"],
                "split": split_name(df.loc[i, "date"], train_end_date, valid_end_date),
            }
        )
    meta = pd.DataFrame(anchors)
    if args.smoke_samples > 0:
        train_idx = meta.index[meta["split"] == "train"][: args.smoke_samples]
        keep = set(train_idx.tolist())
        meta = meta.loc[sorted(keep)].reset_index(drop=True)

    train_anchor_rows = meta.loc[meta["split"] == "train", "anchor_idx"].to_numpy()
    if len(train_anchor_rows) == 0:
        raise ValueError("No training windows available")

    train_source_rows = sorted(
        set(
            row
            for anchor in train_anchor_rows
            for row in range(anchor - args.context_length + 1, anchor + args.prediction_length + 1)
        )
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = df.loc[train_source_rows, numeric_cols]
    imputer.fit(train_values)
    scaler.fit(imputer.transform(train_values))

    values = scaler.transform(imputer.transform(df[numeric_cols])).astype("float32")
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()

    x = np.empty((len(meta), args.context_length, len(numeric_cols)), dtype="float32")
    y = np.empty((len(meta), args.prediction_length), dtype="float32")
    anchor_close = np.empty(len(meta), dtype="float32")
    anchor_dates = []
    splits = []

    for n, row in meta.iterrows():
        i = int(row["anchor_idx"])
        x[n] = values[i - args.context_length + 1 : i + 1]
        y[n] = values[i + 1 : i + 1 + args.prediction_length, close_idx]
        anchor_close[n] = float(df.loc[i, "dce_corn_close"])
        anchor_dates.append(dates[i])
        splits.append(row["split"])

    np.save(args.out_dir / "x.npy", x)
    np.save(args.out_dir / "y_close_scaled.npy", y)
    pd.DataFrame(
        {
            "sample_idx": range(len(meta)),
            "anchor_idx": meta["anchor_idx"],
            "anchor_date": anchor_dates,
            "split": splits,
            "anchor_close": anchor_close,
        }
    ).to_csv(args.out_dir / "samples.csv", index=False)
    with (args.out_dir / "preprocess.pkl").open("wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f)
    (args.out_dir / "feature_columns.json").write_text(json.dumps(numeric_cols, indent=2), encoding="utf-8")
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
                "train_end_date": args.train_end_date,
                "valid_end_date": args.valid_end_date,
                "feature_count": len(numeric_cols),
                "close_idx": close_idx,
                "samples": len(meta),
                "split_counts": pd.Series(splits).value_counts().to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((args.out_dir / "config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
