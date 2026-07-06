"""Command line interface."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from commodity_backtest.backtest.engine import run_backtest
from commodity_backtest.config.loader import load_config
from commodity_backtest.data.diagnosis import diagnose_frame
from commodity_backtest.data.loader import load_commodity_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""

    parser = argparse.ArgumentParser(prog="commodity-backtest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose")
    diagnose_source = diagnose.add_mutually_exclusive_group(required=True)
    diagnose_source.add_argument("--csv")
    diagnose_source.add_argument("--config")
    diagnose.add_argument("--date-col", default="date")

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir", default="experiments/manual_run")

    run_lookbacks = subparsers.add_parser("run-lookbacks")
    run_lookbacks.add_argument("--config", required=True)
    run_lookbacks.add_argument("--output-dir", default="experiments/lookback_sweep")

    auto_window = subparsers.add_parser("auto-window")
    auto_window_source = auto_window.add_mutually_exclusive_group(required=True)
    auto_window_source.add_argument("--csv")
    auto_window_source.add_argument("--config")
    auto_window.add_argument("--date-col", default="date")

    build_config = subparsers.add_parser("build-config")
    build_config.add_argument("--base-config", required=True)
    build_config.add_argument("--output", required=True)
    build_config.add_argument("--commodity-name", required=True)
    build_config.add_argument("--exchange")
    build_config.add_argument("--frequency")
    build_config.add_argument("--csv", required=True)
    build_config.add_argument("--date-col", required=True)
    build_config.add_argument("--price-col", required=True)
    build_config.add_argument("--feature-cols", default=None)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--experiment", required=True)

    interpret = subparsers.add_parser("interpret")
    interpret.add_argument("--experiment", required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "diagnose":
        if args.config:
            cfg = load_config(Path(args.config), validate=True)
            data_cfg = cfg["data"]
            df, encoding = load_commodity_csv(
                data_cfg["csv_path"],
                date_col=data_cfg["date_col"],
                encodings=data_cfg.get("encoding", ["utf-8", "gbk", "gb18030"]),
            )
        else:
            df, encoding = load_commodity_csv(args.csv, date_col=args.date_col)
        report = diagnose_frame(df)
        report["encoding"] = encoding
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "run":
        cfg = load_config(Path(args.config), validate=True)
        comparison = run_backtest(cfg, output_dir=args.output_dir)
        print(comparison.to_string(index=False))
        print(f"comparison.csv written to {Path(args.output_dir) / 'comparison.csv'}")
        return
    if args.command == "run-lookbacks":
        cfg = load_config(Path(args.config), validate=True)
        output_root = Path(args.output_dir)
        rows: list[dict] = []
        for lookback in cfg["lookback"].get("candidates", [cfg["lookback"]["default"]]):
            run_cfg = copy.deepcopy(cfg)
            run_cfg["lookback"]["default"] = int(lookback)
            run_dir = output_root / f"lookback_{lookback}"
            comparison = run_backtest(run_cfg, output_dir=run_dir)
            best = comparison.iloc[0].to_dict()
            best["lookback"] = int(lookback)
            rows.append(best)
        output_root.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(rows).sort_values(["DirAcc", "ProfitFactor", "Sharpe"], ascending=[False, False, False])
        summary.to_csv(output_root / "lookback_comparison.csv", index=False)
        print(summary.to_string(index=False))
        print(f"lookback_comparison.csv written to {output_root / 'lookback_comparison.csv'}")
        return
    if args.command == "auto-window":
        if args.config:
            cfg = load_config(Path(args.config), validate=False)
            data_cfg = cfg["data"]
            df, encoding = load_commodity_csv(
                data_cfg["csv_path"],
                date_col=data_cfg["date_col"],
                encodings=data_cfg.get("encoding", ["utf-8", "gbk", "gb18030"]),
            )
        else:
            df, encoding = load_commodity_csv(args.csv, date_col=args.date_col)
        payload = {
            "rows": int(len(df)),
            "encoding": encoding,
            "recommendation": recommend_window_settings(int(len(df))),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.command == "build-config":
        cfg = load_config(Path(args.base_config), validate=False)
        cfg.setdefault("commodity", {})
        cfg["commodity"]["name"] = args.commodity_name
        if args.exchange is not None:
            cfg["commodity"]["exchange"] = args.exchange
        if args.frequency is not None:
            cfg["commodity"]["frequency"] = args.frequency
        cfg.setdefault("data", {})
        cfg["data"]["csv_path"] = args.csv
        cfg["data"]["date_col"] = args.date_col
        cfg["data"]["price_col"] = args.price_col
        if args.feature_cols is not None:
            cfg["data"]["feature_cols"] = parse_feature_cols(args.feature_cols)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"config written to {output_path}")
        return
    if args.command == "compare":
        path = Path(args.experiment) / "comparison.csv"
        comparison = pd.read_csv(path)
        print(comparison.to_string(index=False))
        return
    if args.command == "interpret":
        path = Path(args.experiment) / "agent_verdict.json"
        print(path.read_text(encoding="utf-8"))
        return
    raise ValueError(f"Unknown command: {args.command}")


def recommend_window_settings(rows: int) -> dict:
    """Recommend conservative lookback and chronological window settings."""

    rows = max(0, int(rows))
    if rows < 24:
        min_train = max(4, rows // 2)
        lookback = max(1, min(3, min_train - 1))
        mode = "expanding"
        window_size = None
        max_train = None
        candidates = sorted({lookback, max(1, lookback - 1)})
    elif rows < 72:
        min_train = 12
        lookback = 3
        mode = "expanding"
        window_size = None
        max_train = None
        candidates = [3, 6, 9]
    elif rows < 144:
        min_train = 24
        lookback = 6
        mode = "expanding_with_cap"
        window_size = None
        max_train = 60
        candidates = [3, 6, 12]
    else:
        min_train = 48
        lookback = 12
        mode = "rolling"
        window_size = 72
        max_train = None
        candidates = [6, 12, 24]

    candidates = [candidate for candidate in candidates if candidate < min_train]
    return {
        "lookback": {"default": lookback, "candidates": candidates},
        "train_window": {
            "mode": mode,
            "min_train_periods": min_train,
            "stride_periods": 1,
            "window_size_periods": window_size,
            "max_train_periods": max_train,
        },
    }


def parse_feature_cols(value: str):
    """Parse CLI feature column input."""

    if value == "auto_numeric":
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()