from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def find_project_root(script_path: Path) -> Path:
    for parent in script_path.parents:
        if (parent / "玉米预测" / "datasets").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {script_path}")


ROOT = find_project_root(Path(__file__).resolve())
SOURCE = Path("/Users/keyizhan/Downloads/玉米价格月度_混合特征无缺失值双头LSTM版.csv")
TARGET = ROOT / "玉米预测/datasets/processed/corn_monthly_dual_stream_spike_github.csv"
BACKUP = ROOT / "玉米预测/datasets/processed/corn_monthly_dual_stream_spike_github_annual_spike_backup.csv"
AUDIT = ROOT / "玉米预测/datasets/processed/corn_monthly_spike_label_audit.csv"
AUDIT_JSON = ROOT / "玉米预测/datasets/processed/corn_monthly_spike_label_audit.json"


def run_lengths(frame: pd.DataFrame, col: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    values = frame[col].tolist()
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            out.append(
                {
                    "start_month": str(frame.loc[start, "month"]),
                    "end_month": str(frame.loc[idx - 1, "month"]),
                    "value": None if pd.isna(values[start]) else int(values[start]),
                    "length": idx - start,
                }
            )
            start = idx
    return out


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)

    df = pd.read_csv(SOURCE)
    df["parsed_month"] = pd.to_datetime(df["month"].astype(str), format="%y-%b")
    df = df.sort_values("parsed_month").reset_index(drop=True)

    old_spike = df["spike"].copy() if "spike" in df.columns else pd.Series([pd.NA] * len(df))
    close = df["dce_corn_close"].astype(float)
    df["dce_corn_close_next_month"] = close.shift(-1)
    df["dce_corn_close_next_month_ret"] = df["dce_corn_close_next_month"] / close - 1.0
    direction = (df["dce_corn_close_next_month_ret"] > 0).astype("float")
    direction[df["dce_corn_close_next_month_ret"].isna()] = pd.NA
    df["dce_corn_close_next_month_direction"] = direction

    labeled_ret = df["dce_corn_close_next_month_ret"].dropna()
    abs_threshold = float(labeled_ret.abs().median())
    monthly_spike = (df["dce_corn_close_next_month_ret"].abs() >= abs_threshold).astype("float")
    monthly_spike[df["dce_corn_close_next_month_ret"].isna()] = pd.NA
    df["spike"] = monthly_spike

    audit = pd.DataFrame(
        {
            "month": df["month"],
            "dce_corn_close": close,
            "dce_corn_close_next_month": df["dce_corn_close_next_month"],
            "dce_corn_close_next_month_ret": df["dce_corn_close_next_month_ret"],
            "old_annual_like_spike": old_spike,
            "new_monthly_spike_abs_ret_ge_median": df["spike"],
            "new_monthly_direction_ret_gt_0": df["dce_corn_close_next_month_direction"],
        }
    )
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    df = df.drop(columns=["parsed_month"])
    df.to_csv(TARGET, index=False, encoding="utf-8-sig")

    audit_payload = {
        "source": str(SOURCE),
        "target": str(TARGET),
        "backup": str(BACKUP),
        "rule": "spike = abs(dce_corn_close_next_month_ret) >= median(abs(dce_corn_close_next_month_ret)) over labeled monthly rows",
        "abs_threshold": abs_threshold,
        "rows": int(len(df)),
        "labeled_spike_rows": int(df["spike"].notna().sum()),
        "old_spike_counts": old_spike.value_counts(dropna=False).sort_index().to_dict(),
        "new_spike_counts": df["spike"].value_counts(dropna=False).sort_index().to_dict(),
        "new_direction_counts": df["dce_corn_close_next_month_direction"].value_counts(dropna=False).sort_index().to_dict(),
        "old_spike_runs": run_lengths(pd.DataFrame({"month": df["month"], "spike": old_spike}), "spike"),
        "new_spike_runs_first_30": run_lengths(pd.DataFrame({"month": df["month"], "spike": df["spike"]}), "spike")[:30],
    }
    AUDIT_JSON.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
