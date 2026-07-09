#!/usr/bin/env python3
"""Official rolling benchmark for monthly corn direction, two heads only.

Feature sets:
- no_news: excludes pca_* columns.
- with_news_precomputed_pca: includes the existing pca_* columns as supplied.

Heads:
- cls: direct up/down classifier trained on price[t+h] > price[t].
- reg: return regressor, inverse-transformed to future price and thresholded.

The base tabular estimators are official scikit-learn/LightGBM/XGBoost/CatBoost
classes from run_corn_monthly_spike_official_50_three_heads.py. The time-series
estimators are official aeon classes. No paper architecture is reimplemented by
hand in this script.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
import sys
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_corn_monthly_spike_official_50_three_heads import (  # noqa: E402
    ConstantClassifier,
    ConstantRegressor,
    TargetStandardizer,
    TrainOnlyStandardizer,
    build_model_zoo,
    flatten,
    load_monthly_csv,
    make_rolling_origins,
    make_samples,
    move_price_last,
    parse_csv_list,
    predict_probability,
    r2_status,
    safe_ap,
    safe_auc,
    safe_balanced_accuracy,
    safe_r2,
    select_feature_columns,
    select_score_threshold,
    set_seed,
    sigmoid,
    stable_name_offset,
)


Factory = Callable[[int], object]


@dataclass(frozen=True)
class OfficialSpec:
    name: str
    family: str
    package: str
    classifier_factory: Factory | None
    regressor_factory: Factory | None
    classifier_loss: str
    regressor_loss: str
    input_kind: str
    source_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default="outputs/corn_monthly_spike_official_pool_two_heads")
    parser.add_argument("--date-col", default="month")
    parser.add_argument("--date-format", default="%y-%b")
    parser.add_argument("--price-col", default="dce_corn_close")
    parser.add_argument("--label-col", default="spike")
    parser.add_argument("--lookbacks", default="1,2,3")
    parser.add_argument("--horizons", default="1,2")
    parser.add_argument("--heads", default="cls,reg")
    parser.add_argument("--feature-sets", default="no_news,with_news_precomputed_pca")
    parser.add_argument("--label-mode", choices=("direct_horizon_direction", "existing_spike_at_target"), default="direct_horizon_direction")
    parser.add_argument("--min-train", type=int, default=48)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=3)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--origin-mode", choices=("blocked", "monthly"), default="blocked")
    parser.add_argument("--threshold-mode", choices=("validation", "fixed"), default="validation")
    parser.add_argument("--test-start", default="")
    parser.add_argument("--monthly-cutoff-lag", type=int, default=0)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--origin-id-start", type=int, default=0)
    parser.add_argument("--origin-id-stop", type=int, default=0)
    parser.add_argument("--extra-exclude", default="")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-folds", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest checkpointed run under --out-dir and skip fully completed feature/lookback/horizon/model combos.",
    )
    parser.add_argument("--aeon-kernels", type=int, default=384)
    parser.add_argument("--aeon-estimators", type=int, default=64)
    parser.add_argument("--deep-epochs", type=int, default=12)
    parser.add_argument("--deep-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    csv_path = Path(args.csv).expanduser().resolve()
    base_out_dir = Path(args.out_dir).expanduser().resolve()
    resume_from = find_latest_checkpoint_run(base_out_dir) if args.resume_latest else None
    if resume_from is not None:
        out_dir = resume_from
        run_id = out_dir.name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "PROGRESS.txt"

    df = load_monthly_csv(csv_path, args.date_col, args.date_format)
    lookbacks = [int(x) for x in parse_csv_list(args.lookbacks)]
    horizons = [int(x) for x in parse_csv_list(args.horizons)]
    requested_heads = parse_csv_list(args.heads)
    validate_heads(requested_heads)
    requested_feature_sets = parse_csv_list(args.feature_sets)
    test_start = pd.to_datetime(args.test_start).date() if args.test_start else None
    model_specs = select_models(
        requested=args.models,
        seed=args.seed,
        aeon_kernels=args.aeon_kernels,
        aeon_estimators=args.aeon_estimators,
        deep_epochs=args.deep_epochs,
        deep_batch_size=args.deep_batch_size,
    )

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
        "requested_heads": requested_heads,
        "feature_sets": requested_feature_sets,
        "min_train": args.min_train,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "monthly_origin_test_rows": 1 if args.origin_mode == "monthly" else args.test_size,
        "origin_mode": args.origin_mode,
        "origin_id_start": args.origin_id_start,
        "origin_id_stop": args.origin_id_stop,
        "threshold_mode": args.threshold_mode,
        "test_start": str(test_start) if test_start is not None else "",
        "monthly_cutoff_lag": args.monthly_cutoff_lag,
        "checkpoint": bool(args.checkpoint),
        "resume_latest": bool(args.resume_latest),
        "resume_from_checkpoint": str(resume_from) if resume_from is not None else "",
        "heads": {
            "cls": "direct classifier probability, validation threshold or fixed probability >= 0.5",
            "reg": "return regressor, train-fold target scaling, inverse return to price, validation threshold or fixed return >= 0",
        },
        "models": [model_to_manifest(m) for m in model_specs],
        "feature_set_details": {},
        "leakage_controls": [
            "future/target/next/lead columns excluded from features",
            "rolling training rows require target_idx <= cutoff anchor index",
            "feature scaler fitted on train fold only",
            "return target scaler fitted on train fold only and inverse-transformed before price/R2 diagnostics",
            "head thresholds selected on validation fold only when threshold_mode=validation; fixed at cls>=0.5 and reg_return>=0 when threshold_mode=fixed",
            "monthly origin mode tests every eligible month once; fixed thresholds train on all rows known by the test anchor month, validation thresholds use the most recent val_size known rows for threshold selection",
            "test_start filters monthly test anchor months; monthly_cutoff_lag shifts known target cutoff earlier to keep the initial 2017 test trained on labels no later than 2016",
            "no sum/fusion head is used",
        ],
        "pca_news_note": (
            "with_news_precomputed_pca includes the pca_* columns already present in the CSV. "
            "The rolling benchmark scales them using train folds only, but cannot prove the PCA "
            "projection itself was fitted without future rows unless raw news features are supplied."
        ),
        "official_sources": official_sources(),
        "runtime_versions": collect_versions(),
        "resource_caps": {
            "aeon_kernels": args.aeon_kernels,
            "aeon_estimators": args.aeon_estimators,
            "deep_epochs": args.deep_epochs,
            "deep_batch_size": args.deep_batch_size,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_predictions: list[dict[str, object]] = []
    all_folds: list[dict[str, object]] = []
    all_errors: list[dict[str, object]] = []
    completed_combo_keys: set[tuple[str, int, int, str]] = set()
    completed_origin_keys: set[tuple[str, int, int, str, int]] = set()
    if resume_from is not None:
        all_predictions, all_folds, all_errors, completed_combo_keys, completed_origin_keys = load_completed_checkpoint_state(
            out_dir=out_dir,
            requested_heads=requested_heads,
        )
        print(
            f"[resume] loaded {len(all_predictions)} predictions, {len(all_folds)} folds, "
            f"{len(all_errors)} errors from {out_dir}; completed combos={len(completed_combo_keys)}, "
            f"completed origins={len(completed_origin_keys)}",
            flush=True,
        )

    for feature_set in requested_feature_sets:
        include_pca = feature_set_to_include_pca(feature_set)
        feature_cols, excluded_cols = select_feature_columns(
            df=df,
            date_col=args.date_col,
            price_col=args.price_col,
            label_col=args.label_col,
            include_pca=include_pca,
            extra_exclude=parse_csv_list(args.extra_exclude),
        )
        feature_cols = move_price_last(feature_cols, args.price_col)
        manifest["feature_set_details"][feature_set] = {
            "include_pca_columns": include_pca,
            "feature_count": len(feature_cols),
            "pca_feature_count": int(sum(c.lower().startswith("pca_") for c in feature_cols)),
            "feature_cols": feature_cols,
            "excluded_cols": excluded_cols,
        }

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
                if args.origin_mode == "monthly":
                    origins = make_monthly_origins(
                        samples,
                        args.min_train,
                        args.val_size,
                        args.max_origins,
                        args.threshold_mode,
                        test_start,
                        args.monthly_cutoff_lag,
                    )
                else:
                    origins = make_rolling_origins(
                        samples,
                        args.min_train,
                        args.val_size,
                        args.test_size,
                        args.step_size,
                        args.max_origins,
                    )
                if args.origin_id_start or args.origin_id_stop:
                    origin_start = max(0, int(args.origin_id_start))
                    origin_stop = int(args.origin_id_stop) if args.origin_id_stop else None
                    origins = [
                        origin
                        for origin in origins
                        if int(origin[3]) >= origin_start
                        and (origin_stop is None or int(origin[3]) < origin_stop)
                    ]
                if not origins:
                    print(f"[skip] feature_set={feature_set} lookback={lookback} horizon={horizon}: no origins", flush=True)
                    continue

                for spec in model_specs:
                    combo_key = (feature_set, int(lookback), int(horizon), spec.name)
                    if combo_key in completed_combo_keys:
                        print(
                            f"[resume-skip] feature_set={feature_set} lookback={lookback} "
                            f"horizon={horizon} model={spec.name}",
                            flush=True,
                        )
                        continue
                    origins_to_run = [
                        origin
                        for origin in origins
                        if (feature_set, int(lookback), int(horizon), spec.name, int(origin[3])) not in completed_origin_keys
                    ]
                    if not origins_to_run:
                        print(
                            f"[resume-skip-origins] feature_set={feature_set} lookback={lookback} "
                            f"horizon={horizon} model={spec.name} origins={len(origins)}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[run] feature_set={feature_set} lookback={lookback} horizon={horizon} "
                        f"model={spec.name} origins={len(origins_to_run)}/{len(origins)}",
                        flush=True,
                    )

                    def checkpoint_after_origin(origin_id: int, partial_preds, partial_folds, partial_errors) -> None:
                        progress_path.write_text(
                            "\n".join(
                                [
                                    f"status=running",
                                    f"last_feature_set={feature_set}",
                                    f"last_lookback={lookback}",
                                    f"last_horizon={horizon}",
                                    f"last_model={spec.name}",
                                    f"last_origin_id={origin_id}",
                                    f"origins={len(origins)}",
                                    f"remaining_origins={len(origins_to_run)}",
                                    f"predictions={len(all_predictions) + len(partial_preds)}",
                                    f"folds={len(all_folds) + len(partial_folds)}",
                                    f"errors={len(all_errors) + len(partial_errors)}",
                                    f"updated_at={datetime.now().isoformat(timespec='seconds')}",
                                ]
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        if args.checkpoint:
                            write_checkpoint(
                                out_dir,
                                all_predictions + list(partial_preds),
                                all_folds + list(partial_folds),
                                all_errors + list(partial_errors),
                                args.save_folds,
                            )

                    preds, folds, errors = run_model_combo(
                        samples,
                        origins_to_run,
                        spec,
                        args.seed,
                        feature_set,
                        args.threshold_mode,
                        requested_heads,
                        after_origin=checkpoint_after_origin,
                    )
                    all_predictions.extend(preds)
                    all_folds.extend(folds)
                    all_errors.extend(errors)
                    for fold in folds:
                        try:
                            completed_origin_keys.add(
                                (
                                    str(fold["feature_set"]),
                                    int(fold["lookback_months"]),
                                    int(fold["horizon_months"]),
                                    str(fold["model"]),
                                    int(fold["origin_id"]),
                                )
                            )
                        except Exception:
                            pass
                    progress_path.write_text(
                        "\n".join(
                            [
                                f"status=running",
                                f"last_feature_set={feature_set}",
                                f"last_lookback={lookback}",
                                f"last_horizon={horizon}",
                                f"last_model={spec.name}",
                                f"origins={len(origins)}",
                                f"predictions={len(all_predictions)}",
                                f"errors={len(all_errors)}",
                                f"updated_at={datetime.now().isoformat(timespec='seconds')}",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if args.checkpoint:
                        write_checkpoint(out_dir, all_predictions, all_folds, all_errors, args.save_folds)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_errors:
        pd.DataFrame(all_errors).to_csv(out_dir / "model_errors.csv", index=False, encoding="utf-8-sig")

    predictions_df = pd.DataFrame(all_predictions)
    if predictions_df.empty:
        raise RuntimeError("No predictions were produced. See model_errors.csv if present.")
    predictions_df.to_csv(out_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")

    summary_df = summarize(predictions_df)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    model_summary_df = summarize_model_level(summary_df)
    model_summary_df.to_csv(out_dir / "model_level_summary.csv", index=False, encoding="utf-8-sig")
    if args.save_folds:
        pd.DataFrame(all_folds).to_csv(out_dir / "folds.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, summary_df, model_summary_df, manifest, len(all_errors))

    print("\n=== Best By Feature Set / Horizon / Head ===")
    display_cols = [
        "feature_set",
        "horizon_months",
        "head",
        "model",
        "lookback_months",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "reg_price_r2",
        "r2_status",
    ]
    best_rows = []
    for (_, _, _), group in summary_df.groupby(["feature_set", "horizon_months", "head"]):
        best_rows.append(group.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False).iloc[0])
    print(pd.DataFrame(best_rows)[display_cols].to_string(index=False))
    if all_errors:
        print(f"\nModel errors recorded: {len(all_errors)} -> {out_dir / 'model_errors.csv'}")
    progress_path.write_text(
        "\n".join(
            [
                "status=complete",
                f"predictions={len(all_predictions)}",
                f"errors={len(all_errors)}",
                f"updated_at={datetime.now().isoformat(timespec='seconds')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {out_dir}")


def validate_heads(heads: list[str]) -> None:
    allowed = {"cls", "reg"}
    if not heads:
        raise ValueError("--heads must include at least one of cls,reg")
    unknown = sorted(set(heads) - allowed)
    if unknown:
        raise ValueError(f"Unknown head(s): {unknown}. Use cls, reg, or cls,reg.")


def feature_set_to_include_pca(feature_set: str) -> bool:
    mapping = {
        "no_news": False,
        "with_news_precomputed_pca": True,
    }
    if feature_set not in mapping:
        raise ValueError(f"Unknown feature set {feature_set!r}. Use no_news or with_news_precomputed_pca.")
    return mapping[feature_set]


def make_monthly_origins(
    samples,
    min_train: int,
    val_size: int,
    max_origins: int,
    threshold_mode: str,
    test_start,
    monthly_cutoff_lag: int,
):
    """One walk-forward test row per eligible anchor month.

    For a test anchor month A, only rows whose target month is already known by
    A are allowed. In validation threshold mode, the latest val_size known rows
    are held out for threshold selection and earlier rows are used for fitting.
    """
    meta = samples["meta"]
    origins = []
    origin_id = 0
    for sample_idx, row in meta.iterrows():
        anchor_idx = int(row["anchor_idx"])
        anchor_month = pd.to_datetime(row["anchor_month"]).date()
        if test_start is not None and anchor_month < test_start:
            continue
        cutoff_idx = anchor_idx - max(0, int(monthly_cutoff_lag))
        if cutoff_idx < 0:
            continue
        known_idx = meta.index[meta["target_idx"] <= cutoff_idx].to_numpy(dtype=int)
        if threshold_mode == "validation":
            if len(known_idx) < min_train + val_size:
                continue
            train_idx = known_idx[:-val_size]
            val_idx = known_idx[-val_size:]
        else:
            if len(known_idx) < min_train:
                continue
            train_idx = known_idx
            val_idx = np.asarray([], dtype=int)
        if len(train_idx) < min_train:
            continue
        origins.append((train_idx, val_idx, np.asarray([sample_idx], dtype=int), origin_id))
        origin_id += 1
        if max_origins and len(origins) >= max_origins:
            break
    return origins


def write_checkpoint(
    out_dir: Path,
    predictions: list[dict[str, object]],
    folds: list[dict[str, object]],
    errors: list[dict[str, object]],
    save_folds: bool,
) -> None:
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(out_dir / "checkpoint_rolling_predictions.csv", index=False, encoding="utf-8-sig")
        summarize(predictions_df).to_csv(out_dir / "checkpoint_summary_metrics.csv", index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "checkpoint_model_errors.csv", index=False, encoding="utf-8-sig")
    if save_folds and folds:
        pd.DataFrame(folds).to_csv(out_dir / "checkpoint_folds.csv", index=False, encoding="utf-8-sig")


def find_latest_checkpoint_run(base_out_dir: Path) -> Path | None:
    if not base_out_dir.exists():
        return None
    candidates = [
        path
        for path in base_out_dir.iterdir()
        if path.is_dir() and (path / "checkpoint_rolling_predictions.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "checkpoint_rolling_predictions.csv").stat().st_mtime)


def load_completed_checkpoint_state(
    out_dir: Path,
    requested_heads: list[str],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    set[tuple[str, int, int, str]],
    set[tuple[str, int, int, str, int]],
]:
    group_cols = ["feature_set", "lookback_months", "horizon_months", "model"]

    def read_checkpoint_csv(name: str) -> pd.DataFrame:
        path = out_dir / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"[resume] unable to read {path}: {exc}", flush=True)
            return pd.DataFrame()

    predictions = read_checkpoint_csv("checkpoint_rolling_predictions.csv")
    folds = read_checkpoint_csv("checkpoint_folds.csv")
    errors = read_checkpoint_csv("checkpoint_model_errors.csv")

    completed_origin_keys: set[tuple[str, int, int, str, int]] = set()
    origin_cols = set(group_cols + ["origin_id"])
    pred_origin_cols = origin_cols | {"head"}
    requested_head_set = {str(head) for head in requested_heads}
    if not predictions.empty and pred_origin_cols.issubset(predictions.columns):
        # A fold row can exist even when both heads failed and produced no
        # prediction. Only treat an origin as complete when all requested heads
        # have predictions, so transient deep-learning failures are retryable.
        for key_values, group in predictions.groupby(group_cols + ["origin_id"], dropna=False):
            try:
                present_heads = {str(head) for head in group["head"].dropna().unique()}
                if not requested_head_set.issubset(present_heads):
                    continue
                feature_set, lookback, horizon, model, origin_id = key_values
                completed_origin_keys.add(
                    (
                        str(feature_set),
                        int(lookback),
                        int(horizon),
                        str(model),
                        int(origin_id),
                    )
                )
            except Exception:
                continue

    # A partial checkpoint can already contain both requested heads after one
    # origin, so whole-combo completion cannot be inferred from summary rows.
    # Resume by origin/fold identity instead.
    completed_keys: set[tuple[str, int, int, str]] = set()

    return (
        predictions.to_dict("records"),
        folds.to_dict("records"),
        errors.to_dict("records"),
        completed_keys,
        completed_origin_keys,
    )


def run_model_combo(
    samples,
    origins,
    spec: OfficialSpec,
    seed: int,
    feature_set: str,
    threshold_mode: str,
    requested_heads: list[str],
    after_origin=None,
):
    x = samples["X"]
    y = samples["y"]
    returns = samples["returns"]
    meta = samples["meta"]
    lookback = int(samples["lookback"])
    horizon = int(samples["horizon"])
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for train_idx, val_idx, test_idx, origin_id in origins:
        fold_seed = seed + stable_name_offset(spec.name) + 101 * lookback + 17 * horizon + origin_id
        scaler = TrainOnlyStandardizer().fit(x[train_idx])
        x_train_3d = scaler.transform(x[train_idx])
        x_val_3d = scaler.transform(x[val_idx]) if len(val_idx) else None
        x_test_3d = scaler.transform(x[test_idx])
        x_train = format_input(x_train_3d, spec.input_kind)
        x_val = format_input(x_val_3d, spec.input_kind) if x_val_3d is not None else None
        x_test = format_input(x_test_3d, spec.input_kind)

        cls_threshold = float("nan")
        reg_threshold = float("nan")
        cls_test_prob = None
        cls_pred = None
        reg_test_return = None
        reg_test_prob = None
        reg_pred = None

        if "cls" in requested_heads and spec.classifier_factory is not None:
            try:
                clf = fit_classifier(spec, x_train, y[train_idx], fold_seed)
                cls_test_prob = predict_probability(clf, x_test)
                if threshold_mode == "fixed":
                    cls_threshold = 0.5
                else:
                    if x_val is None:
                        raise RuntimeError("Validation threshold mode requires non-empty validation indices.")
                    cls_val_prob = predict_probability(clf, x_val)
                    cls_threshold = select_score_threshold(y[val_idx], cls_val_prob)
                cls_pred = cls_test_prob >= cls_threshold
            except Exception as exc:  # keep the full run alive when aeon rejects a tiny window
                errors.append(error_row(spec, feature_set, lookback, horizon, origin_id, "cls", exc))

        if "reg" in requested_heads and spec.regressor_factory is not None:
            y_scaler = TargetStandardizer().fit(returns[train_idx])
            try:
                reg = fit_regressor(spec, x_train, y_scaler.transform(returns[train_idx]), fold_seed)
                reg_test_return = y_scaler.inverse_transform(np.asarray(reg.predict(x_test), dtype=float))
                if threshold_mode == "fixed":
                    reg_threshold = 0.0
                else:
                    if x_val is None:
                        raise RuntimeError("Validation threshold mode requires non-empty validation indices.")
                    reg_val_return = y_scaler.inverse_transform(np.asarray(reg.predict(x_val), dtype=float))
                    reg_threshold = select_score_threshold(y[val_idx], reg_val_return)
                return_scale = max(float(np.nanstd(returns[train_idx])), 1e-4)
                reg_test_prob = sigmoid(reg_test_return / return_scale)
                reg_pred = reg_test_return >= reg_threshold
            except Exception as exc:
                errors.append(error_row(spec, feature_set, lookback, horizon, origin_id, "reg", exc))

        actual_price = meta.iloc[test_idx]["actual_price"].to_numpy(float)
        anchor_price = meta.iloc[test_idx]["anchor_price"].to_numpy(float)
        if reg_test_return is not None:
            predicted_price = anchor_price * (1.0 + reg_test_return)
            reg_price_r2 = safe_r2(actual_price, predicted_price)
        else:
            predicted_price = np.full(len(test_idx), np.nan, dtype=float)
            reg_price_r2 = float("nan")
        naive_price_r2 = safe_r2(actual_price, anchor_price)
        train_target_months = pd.to_datetime(meta.iloc[train_idx]["target_month"])
        val_target_months = pd.to_datetime(meta.iloc[val_idx]["target_month"]) if len(val_idx) else pd.Series(dtype="datetime64[ns]")
        known_cutoff_month = val_target_months.max() if len(val_idx) else train_target_months.max()

        folds.append(
            {
                "feature_set": feature_set,
                "model": spec.name,
                "family": spec.family,
                "package": spec.package,
                "input_kind": spec.input_kind,
                "lookback_months": lookback,
                "horizon_months": horizon,
                "origin_id": origin_id,
                "cutoff_month": str(known_cutoff_month.date()),
                "train_min_target_month": str(train_target_months.min().date()),
                "train_max_target_month": str(train_target_months.max().date()),
                "val_min_target_month": str(val_target_months.min().date()) if len(val_idx) else "",
                "val_max_target_month": str(val_target_months.max().date()) if len(val_idx) else "",
                "test_anchor_month": str(meta.iloc[test_idx[0]]["anchor_month"]),
                "test_target_month": str(meta.iloc[test_idx[0]]["target_month"]),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "test_rows": len(test_idx),
                "threshold_mode": threshold_mode,
                "cls_threshold": cls_threshold,
                "reg_threshold": reg_threshold,
                "reg_price_r2_fold": float(reg_price_r2),
                "naive_price_r2_fold": float(naive_price_r2),
            }
        )

        head_payloads = []
        if cls_test_prob is not None and cls_pred is not None:
            head_payloads.append(("cls", cls_test_prob, cls_pred, cls_threshold))
        if reg_test_prob is not None and reg_pred is not None:
            head_payloads.append(("reg", reg_test_prob, reg_pred, reg_threshold))
        if not head_payloads:
            cleanup_deep_backend(spec)
            if after_origin is not None:
                after_origin(int(origin_id), predictions, folds, errors)
            continue

        for offset, sample_idx in enumerate(test_idx):
            base = {
                "feature_set": feature_set,
                "model": spec.name,
                "family": spec.family,
                "package": spec.package,
                "source_kind": spec.source_kind,
                "input_kind": spec.input_kind,
                "classifier_loss": spec.classifier_loss,
                "regressor_loss": spec.regressor_loss,
                "lookback_months": lookback,
                "horizon_months": horizon,
                "origin_id": origin_id,
                "anchor_month": meta.iloc[sample_idx]["anchor_month"],
                "target_month": meta.iloc[sample_idx]["target_month"],
                "anchor_idx": int(meta.iloc[sample_idx]["anchor_idx"]),
                "target_idx": int(meta.iloc[sample_idx]["target_idx"]),
                "anchor_price": float(anchor_price[offset]),
                "actual_price": float(actual_price[offset]),
                "predicted_price_reg": finite_or_nan(predicted_price[offset]),
                "actual_return": float(meta.iloc[sample_idx]["actual_return"]),
                "predicted_return_reg": finite_or_nan(reg_test_return[offset]) if reg_test_return is not None else float("nan"),
                "actual_direction": int(y[sample_idx]),
                "target_existing_spike": int(meta.iloc[sample_idx]["target_existing_spike"]),
                "reg_price_r2_fold": float(reg_price_r2),
                "naive_price_r2_fold": float(naive_price_r2),
            }
            for head, prob, pred, threshold in head_payloads:
                row = dict(base)
                row.update(
                    {
                        "head": head,
                        "threshold": float(threshold),
                        "predicted_direction": int(pred[offset]),
                        "predicted_probability": float(prob[offset]),
                    }
                )
                predictions.append(row)

        cleanup_deep_backend(spec)
        if after_origin is not None:
            after_origin(int(origin_id), predictions, folds, errors)

    return predictions, folds, errors


def cleanup_deep_backend(spec: OfficialSpec) -> None:
    if spec.input_kind == "keras_sequence" or spec.family == "tsc_deep_learning":
        try:
            from tensorflow import keras

            keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()


def configure_tensorflow_runtime() -> None:
    try:
        import tensorflow as tf

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        pass
    except Exception:
        pass


def format_input(x: np.ndarray, input_kind: str) -> np.ndarray:
    if input_kind == "tabular_flat":
        return flatten(x)
    if input_kind == "keras_sequence":
        return np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if input_kind == "aeon_collection":
        return np.ascontiguousarray(np.transpose(x, (0, 2, 1)).astype(np.float32))
    if input_kind == "aeon_collection_pad10":
        collection = np.ascontiguousarray(np.transpose(x, (0, 2, 1)).astype(np.float32))
        if collection.shape[-1] < 10:
            pad_width = [(0, 0), (0, 0), (0, 10 - collection.shape[-1])]
            collection = np.pad(collection, pad_width=pad_width, mode="constant", constant_values=0.0)
        return collection
    if input_kind == "aeon_collection_pad10_float64":
        collection = np.ascontiguousarray(np.transpose(x, (0, 2, 1)).astype(np.float64))
        if collection.shape[-1] < 10:
            pad_width = [(0, 0), (0, 0), (0, 10 - collection.shape[-1])]
            collection = np.pad(collection, pad_width=pad_width, mode="constant", constant_values=0.0)
        return collection
    raise ValueError(f"Unknown input_kind={input_kind!r}")


class KerasSequenceClassifier:
    def __init__(self, architecture: str, params: dict[str, object], epochs: int, batch_size: int, seed: int) -> None:
        self.architecture = architecture
        self.params = dict(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model_ = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        tf, keras = import_keras()
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = build_keras_sequence_model(
            architecture=self.architecture,
            input_shape=x.shape[1:],
            params=self.params,
            task="classification",
            seed=self.seed,
        )
        class_weight = None
        unique, counts = np.unique(y.astype(int), return_counts=True)
        if len(unique) == 2 and counts.min() > 0:
            total = float(counts.sum())
            class_weight = {int(cls): total / (2.0 * float(count)) for cls, count in zip(unique, counts)}
        self.model_.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(1, len(y))),
            verbose=0,
            shuffle=False,
            class_weight=class_weight,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Keras classifier is not fitted.")
        p = np.asarray(self.model_.predict(np.asarray(x, dtype=np.float32), verbose=0), dtype=float).reshape(-1)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class KerasSequenceRegressor:
    def __init__(self, architecture: str, params: dict[str, object], epochs: int, batch_size: int, seed: int) -> None:
        self.architecture = architecture
        self.params = dict(params)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.model_ = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        tf, keras = import_keras()
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = build_keras_sequence_model(
            architecture=self.architecture,
            input_shape=x.shape[1:],
            params=self.params,
            task="regression",
            seed=self.seed,
        )
        self.model_.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=min(self.batch_size, max(1, len(y))),
            verbose=0,
            shuffle=False,
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Keras regressor is not fitted.")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=np.float32), verbose=0), dtype=float).reshape(-1)


def import_keras():
    configure_tensorflow_runtime()
    import tensorflow as tf
    from tensorflow import keras

    configure_tensorflow_runtime()
    return tf, keras


def build_keras_sequence_model(
    architecture: str,
    input_shape: tuple[int, ...],
    params: dict[str, object],
    task: str,
    seed: int,
):
    tf, keras = import_keras()
    inputs = keras.Input(shape=input_shape)
    x = inputs
    architecture = architecture.lower()
    if architecture in {"lstm", "gru", "bilstm", "bigru"}:
        layer_name = "GRU" if "gru" in architecture else "LSTM"
        layer_cls = getattr(keras.layers, layer_name)
        bidirectional = architecture.startswith("bi") or bool(params.get("bidirectional", False))
        units = as_int_list(params.get("units", 32))
        dropout = float(params.get("dropout", 0.0))
        recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
        for idx, unit in enumerate(units):
            return_sequences = idx < len(units) - 1
            recurrent = layer_cls(
                int(unit),
                activation=str(params.get("activation", "tanh")),
                recurrent_activation=str(params.get("recurrent_activation", "sigmoid")),
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=return_sequences,
            )
            x = keras.layers.Bidirectional(recurrent)(x) if bidirectional else recurrent(x)
        if bool(params.get("layer_norm", False)):
            x = keras.layers.LayerNormalization()(x)
    elif architecture == "tcn":
        from tcn import TCN

        x = TCN(
            nb_filters=int(params.get("nb_filters", 16)),
            kernel_size=int(params.get("kernel_size", 2)),
            nb_stacks=int(params.get("nb_stacks", 1)),
            dilations=tuple(int(v) for v in params.get("dilations", (1,))),
            padding=str(params.get("padding", "causal")),
            use_skip_connections=bool(params.get("use_skip_connections", True)),
            dropout_rate=float(params.get("dropout_rate", 0.0)),
            return_sequences=False,
            activation=str(params.get("activation", "relu")),
            use_batch_norm=bool(params.get("use_batch_norm", False)),
            use_layer_norm=bool(params.get("use_layer_norm", False)),
            name=f"tcn_{seed % 100000}",
        )(x)
    else:
        raise ValueError(f"Unsupported Keras sequence architecture {architecture!r}")

    dense_units = as_int_list(params.get("dense_units", []))
    for unit in dense_units:
        x = keras.layers.Dense(int(unit), activation=str(params.get("dense_activation", "relu")))(x)
        if float(params.get("dense_dropout", 0.0)) > 0:
            x = keras.layers.Dropout(float(params.get("dense_dropout", 0.0)))(x)
    if task == "classification":
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)
        loss = str(params.get("loss", "binary_crossentropy"))
        metrics = ["accuracy"]
    elif task == "regression":
        outputs = keras.layers.Dense(1, activation="linear")(x)
        loss = str(params.get("loss", "mse"))
        metrics = ["mse"]
    else:
        raise ValueError(f"Unknown task={task!r}")
    model = keras.Model(inputs=inputs, outputs=outputs)
    learning_rate = float(params.get("learning_rate", 1e-3))
    optimizer_name = str(params.get("optimizer", "adam")).lower()
    if optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=float(params.get("weight_decay", 1e-4)))
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def as_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def fit_classifier(spec: OfficialSpec, x: np.ndarray, y: np.ndarray, seed: int):
    if len(np.unique(y)) < 2:
        return ConstantClassifier(float(np.mean(y)))
    if spec.classifier_factory is None:
        raise RuntimeError(f"{spec.name} does not define a classifier.")
    model = spec.classifier_factory(seed)
    model.fit(x, y.astype(int))
    return model


def fit_regressor(spec: OfficialSpec, x: np.ndarray, y_scaled: np.ndarray, seed: int):
    if np.nanstd(y_scaled) < 1e-12:
        return ConstantRegressor(float(np.nanmean(y_scaled)))
    if spec.regressor_factory is None:
        raise RuntimeError(f"{spec.name} does not define a regressor.")
    model = spec.regressor_factory(seed)
    model.fit(x, y_scaled)
    return model


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "feature_set",
        "model",
        "family",
        "package",
        "source_kind",
        "input_kind",
        "lookback_months",
        "horizon_months",
        "head",
        "classifier_loss",
        "regressor_loss",
    ]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        y = group["actual_direction"].to_numpy(dtype=int)
        pred = group["predicted_direction"].to_numpy(dtype=int)
        prob = group["predicted_probability"].to_numpy(dtype=float)
        actual_price = group["actual_price"].to_numpy(dtype=float)
        pred_price = group["predicted_price_reg"].to_numpy(dtype=float)
        anchor_price = group["anchor_price"].to_numpy(dtype=float)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        reg_price_r2 = safe_r2(actual_price, pred_price) if np.isfinite(pred_price).any() else float("nan")
        naive_r2 = safe_r2(actual_price, anchor_price)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_predictions": int(len(group)),
                "class_0_count": int((y == 0).sum()),
                "class_1_count": int((y == 1).sum()),
                "auc": safe_auc(y, prob),
                "average_precision": safe_ap(y, prob),
                "balanced_accuracy": safe_balanced_accuracy(y, pred),
                "accuracy": float(accuracy_score(y, pred)),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "reg_price_r2": reg_price_r2,
                "naive_price_r2": naive_r2,
                "r2_status": r2_status(reg_price_r2),
                "price_rmse": float(math.sqrt(mean_squared_error(actual_price, pred_price))) if np.isfinite(pred_price).all() else float("nan"),
                "price_mae": float(mean_absolute_error(actual_price, pred_price)) if np.isfinite(pred_price).all() else float("nan"),
                "predicted_positive_rate": float(np.mean(pred == 1)),
                "actual_positive_rate": float(np.mean(y == 1)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["feature_set", "horizon_months", "head", "balanced_accuracy", "auc", "average_precision"],
        ascending=[True, True, True, False, False, False],
    )


def summarize_model_level(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranked = summary.copy()
    ranked["_selection_score"] = ranked["balanced_accuracy"].fillna(float("-inf"))
    idx = ranked.groupby(["feature_set", "model", "head"])["_selection_score"].idxmax()
    best = ranked.loc[idx].drop(columns=["_selection_score"]).copy()
    return best.sort_values(
        ["feature_set", "head", "balanced_accuracy", "auc", "average_precision"],
        ascending=[True, True, False, False, False],
        na_position="last",
    )


def write_report(out_dir: Path, summary: pd.DataFrame, model_summary: pd.DataFrame, manifest: dict[str, object], error_count: int) -> None:
    lines = [
        "# Corn Monthly Official Model Pool Two-Head Rolling Benchmark",
        "",
        f"- CSV: `{manifest['csv']}`",
        f"- Date range: `{manifest['date_min']}` to `{manifest['date_max']}`",
        f"- Label mode: `{manifest['label_mode']}`",
        f"- Models requested: `{len(manifest['models'])}` official/standard estimators",
        "- Heads: `cls`, `reg`",
        f"- Model errors recorded: `{error_count}`",
        "",
        "## Feature Sets",
        "",
    ]
    for feature_set, detail in manifest["feature_set_details"].items():
        lines.append(
            f"- `{feature_set}`: features `{detail['feature_count']}`, "
            f"pca features `{detail['pca_feature_count']}`"
        )
    lines.extend(["", "## Leakage Controls", ""])
    for item in manifest["leakage_controls"]:
        lines.append(f"- {item}")
    lines.extend(["", "## PCA News Note", "", manifest["pca_news_note"], "", "## Official Sources", ""])
    for name, url in manifest["official_sources"].items():
        lines.append(f"- {name}: {url}")

    lines.extend(["", "## Best By Feature Set, Horizon And Head", ""])
    best_rows = []
    for (_, _, _), group in summary.groupby(["feature_set", "horizon_months", "head"]):
        best_rows.append(group.sort_values(["balanced_accuracy", "auc", "average_precision"], ascending=False).iloc[0])
    display_cols = [
        "feature_set",
        "horizon_months",
        "head",
        "model",
        "lookback_months",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "reg_price_r2",
        "naive_price_r2",
        "r2_status",
    ]
    lines.append(pd.DataFrame(best_rows)[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Best Per Model/Head", ""])
    lines.append(model_summary[display_cols].head(120).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "Outputs:",
            "",
            "- `summary_metrics.csv`",
            "- `model_level_summary.csv`",
            "- `rolling_predictions.csv`",
            "- `manifest.json`",
            "- `model_errors.csv` if any model/fold/head failed",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_official_model_pool(
    seed: int,
    aeon_kernels: int,
    aeon_estimators: int,
    deep_epochs: int,
    deep_batch_size: int,
) -> list[OfficialSpec]:
    specs: list[OfficialSpec] = []
    for base in build_model_zoo(seed):
        classifier_factory = base.classifier_factory
        if base.name == "ada_boost_tree":
            classifier_factory = adaboost_tree_classifier_factory()
        specs.append(
            OfficialSpec(
                name=base.name,
                family=base.family,
                package=base.package,
                classifier_factory=classifier_factory,
                regressor_factory=base.regressor_factory,
                classifier_loss=base.classifier_loss,
                regressor_loss=base.regressor_loss,
                input_kind="tabular_flat",
                source_kind="official_tabular_package",
            )
        )

    # Paired official aeon time-series classifiers/regressors.
    paired_aeon = [
        spec_pair(
            "aeon_rocket",
            "tsc_convolution",
            "RocketClassifier",
            "RocketRegressor",
            "aeon.classification.convolution_based",
            "aeon.regression.convolution_based",
            {"n_kernels": aeon_kernels, "n_jobs": 1},
            {"n_kernels": aeon_kernels, "n_jobs": 1},
            "rocket_ridge",
            "rocket_ridge",
        ),
        spec_pair(
            "aeon_minirocket",
            "tsc_convolution",
            "MiniRocketClassifier",
            "MiniRocketRegressor",
            "aeon.classification.convolution_based",
            "aeon.regression.convolution_based",
            {"n_kernels": aeon_kernels, "n_jobs": 1},
            {"n_kernels": aeon_kernels, "n_jobs": 1},
            "minirocket_ridge",
            "minirocket_ridge",
            input_kind="aeon_collection_pad10",
        ),
        spec_pair(
            "aeon_multirocket",
            "tsc_convolution",
            "MultiRocketClassifier",
            "MultiRocketRegressor",
            "aeon.classification.convolution_based",
            "aeon.regression.convolution_based",
            {"n_kernels": aeon_kernels, "n_features_per_kernel": 4, "n_jobs": 1},
            {"n_kernels": aeon_kernels, "n_features_per_kernel": 4, "n_jobs": 1},
            "multirocket_ridge",
            "multirocket_ridge",
            input_kind="aeon_collection_pad10",
        ),
        spec_pair(
            "aeon_hydra",
            "tsc_convolution",
            "HydraClassifier",
            "HydraRegressor",
            "aeon.classification.convolution_based",
            "aeon.regression.convolution_based",
            {"n_kernels": 8, "n_groups": 16, "n_jobs": 1},
            {"n_kernels": 8, "n_groups": 16, "n_jobs": 1},
            "hydra_ridge",
            "hydra_ridge",
        ),
        spec_pair(
            "aeon_multirocket_hydra",
            "tsc_convolution",
            "MultiRocketHydraClassifier",
            "MultiRocketHydraRegressor",
            "aeon.classification.convolution_based",
            "aeon.regression.convolution_based",
            {"n_kernels": 8, "n_groups": 16, "n_jobs": 1},
            {"n_kernels": 8, "n_groups": 16, "n_jobs": 1},
            "multirocket_hydra_ridge",
            "multirocket_hydra_ridge",
            input_kind="aeon_collection_pad10",
        ),
        spec_pair(
            "aeon_tsf",
            "tsc_interval",
            "TimeSeriesForestClassifier",
            "TimeSeriesForestRegressor",
            "aeon.classification.interval_based",
            "aeon.regression.interval_based",
            {"n_estimators": aeon_estimators, "min_interval_length": 1, "n_jobs": 1},
            {"n_estimators": aeon_estimators, "min_interval_length": 1, "n_jobs": 1},
            "interval_forest",
            "interval_forest",
        ),
        spec_pair(
            "aeon_rise",
            "tsc_interval",
            "RandomIntervalSpectralEnsembleClassifier",
            "RandomIntervalSpectralEnsembleRegressor",
            "aeon.classification.interval_based",
            "aeon.regression.interval_based",
            {"n_estimators": max(16, aeon_estimators // 2), "min_interval_length": 1, "acf_lag": 1, "acf_min_values": 1, "n_jobs": 1},
            {"n_estimators": max(16, aeon_estimators // 2), "min_interval_length": 1, "acf_lag": 1, "acf_min_values": 1, "n_jobs": 1},
            "rise",
            "rise",
        ),
        spec_pair(
            "aeon_cif",
            "tsc_interval",
            "CanonicalIntervalForestClassifier",
            "CanonicalIntervalForestRegressor",
            "aeon.classification.interval_based",
            "aeon.regression.interval_based",
            {"n_estimators": aeon_estimators, "min_interval_length": 1, "att_subsample_size": 4, "n_jobs": 1},
            {"n_estimators": aeon_estimators, "min_interval_length": 1, "att_subsample_size": 4, "n_jobs": 1},
            "canonical_interval_forest",
            "canonical_interval_forest",
        ),
        spec_pair(
            "aeon_drcif",
            "tsc_interval",
            "DrCIFClassifier",
            "DrCIFRegressor",
            "aeon.classification.interval_based",
            "aeon.regression.interval_based",
            {"n_estimators": aeon_estimators, "min_interval_length": 1, "att_subsample_size": 4, "n_jobs": 1},
            {"n_estimators": aeon_estimators, "min_interval_length": 1, "att_subsample_size": 4, "n_jobs": 1},
            "drcif",
            "drcif",
        ),
        spec_pair(
            "aeon_quant",
            "tsc_interval",
            "QUANTClassifier",
            "QUANTRegressor",
            "aeon.classification.interval_based",
            "aeon.regression.interval_based",
            {"interval_depth": 3},
            {"interval_depth": 3},
            "quant",
            "quant",
        ),
        spec_pair(
            "aeon_catch22",
            "tsc_feature",
            "Catch22Classifier",
            "Catch22Regressor",
            "aeon.classification.feature_based",
            "aeon.regression.feature_based",
            {"catch24": True, "replace_nans": True, "n_jobs": 1},
            {"catch24": True, "replace_nans": True, "n_jobs": 1},
            "catch22_random_forest",
            "catch22_random_forest",
        ),
        spec_pair(
            "aeon_summary",
            "tsc_feature",
            "SummaryClassifier",
            "SummaryRegressor",
            "aeon.classification.feature_based",
            "aeon.regression.feature_based",
            {"n_jobs": 1},
            {"n_jobs": 1},
            "summary_random_forest",
            "summary_random_forest",
        ),
        spec_pair(
            "aeon_freshprince",
            "tsc_feature",
            "FreshPRINCEClassifier",
            "FreshPRINCERegressor",
            "aeon.classification.feature_based",
            "aeon.regression.feature_based",
            {"n_estimators": max(16, aeon_estimators // 2), "default_fc_parameters": "minimal", "n_jobs": 1},
            {"n_estimators": max(16, aeon_estimators // 2), "default_fc_parameters": "minimal", "n_jobs": 1},
            "tsfresh_random_forest",
            "tsfresh_random_forest",
        ),
        spec_pair(
            "aeon_rdst",
            "tsc_shapelet",
            "RDSTClassifier",
            "RDSTRegressor",
            "aeon.classification.shapelet_based",
            "aeon.regression.shapelet_based",
            {"max_shapelets": 256, "shapelet_lengths": np.array([2], dtype=np.int64), "n_jobs": 1},
            {"max_shapelets": 256, "shapelet_lengths": np.array([2], dtype=np.int64), "n_jobs": 1},
            "random_dilated_shapelet",
            "random_dilated_shapelet",
            input_kind="aeon_collection_pad10_float64",
        ),
        spec_pair(
            "aeon_knn_dtw",
            "tsc_distance",
            "KNeighborsTimeSeriesClassifier",
            "KNeighborsTimeSeriesRegressor",
            "aeon.classification.distance_based",
            "aeon.regression.distance_based",
            {"n_neighbors": 3, "distance": "dtw", "n_jobs": 1},
            {"n_neighbors": 3, "distance": "dtw", "n_jobs": 1},
            "dtw_vote",
            "dtw_mean",
        ),
        spec_pair(
            "aeon_knn_euclidean",
            "tsc_distance",
            "KNeighborsTimeSeriesClassifier",
            "KNeighborsTimeSeriesRegressor",
            "aeon.classification.distance_based",
            "aeon.regression.distance_based",
            {"n_neighbors": 5, "distance": "euclidean", "n_jobs": 1},
            {"n_neighbors": 5, "distance": "euclidean", "n_jobs": 1},
            "euclidean_vote",
            "euclidean_mean",
        ),
        spec_pair(
            "aeon_rist",
            "tsc_hybrid",
            "RISTClassifier",
            "RISTRegressor",
            "aeon.classification.hybrid",
            "aeon.regression.hybrid",
            {"n_intervals": 4, "n_shapelets": 64, "n_jobs": 1},
            {"n_intervals": 4, "n_shapelets": 64, "n_jobs": 1},
            "random_interval_shapelet",
            "random_interval_shapelet",
        ),
        deep_pair(
            "aeon_deep_mlp",
            "MLPClassifier",
            "MLPRegressor",
            deep_epochs,
            deep_batch_size,
            {"n_layers": 2, "n_units": 64},
            {"n_layers": 2, "n_units": 64},
        ),
        deep_pair(
            "aeon_deep_fcn",
            "FCNClassifier",
            "FCNRegressor",
            deep_epochs,
            deep_batch_size,
            {"n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]},
            {"n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]},
        ),
        deep_pair(
            "aeon_deep_resnet",
            "ResNetClassifier",
            "ResNetRegressor",
            deep_epochs,
            deep_batch_size,
            {"n_residual_blocks": 2, "n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]},
            {"n_residual_blocks": 2, "n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]},
        ),
        deep_pair(
            "aeon_deep_inceptiontime",
            "InceptionTimeClassifier",
            "InceptionTimeRegressor",
            deep_epochs,
            deep_batch_size,
            {"n_classifiers": 1, "n_filters": 16, "kernel_size": 3, "depth": 3, "bottleneck_size": 8},
            {"n_regressors": 1, "n_filters": 16, "kernel_size": 3, "depth": 3, "bottleneck_size": 8},
        ),
        deep_pair(
            "aeon_deep_timecnn",
            "TimeCNNClassifier",
            "TimeCNNRegressor",
            deep_epochs,
            deep_batch_size,
            {"kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16]},
            {"kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16]},
        ),
        deep_pair(
            "aeon_deep_lite_time",
            "LITETimeClassifier",
            "LITETimeRegressor",
            deep_epochs,
            deep_batch_size,
            {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1},
            {"n_regressors": 1, "n_filters": 16, "kernel_size": 1},
        ),
    ]
    specs.extend(paired_aeon)

    # Official classification-only aeon estimators with no official paired regressor.
    specs.extend(
        [
            cls_only("aeon_arsenal", "tsc_convolution", "Arsenal", "aeon.classification.convolution_based", {"n_kernels": aeon_kernels, "n_estimators": 8, "n_jobs": 1}, "arsenal"),
            cls_only("aeon_boss_ensemble", "tsc_dictionary", "BOSSEnsemble", "aeon.classification.dictionary_based", {"min_window": 1, "max_win_len_prop": 1, "max_ensemble_size": 20, "n_jobs": 1}, "boss"),
            cls_only("aeon_contractable_boss", "tsc_dictionary", "ContractableBOSS", "aeon.classification.dictionary_based", {"n_parameter_samples": 32, "min_window": 1, "max_ensemble_size": 20, "n_jobs": 1}, "contractable_boss"),
            cls_only("aeon_weasel", "tsc_dictionary", "WEASEL", "aeon.classification.dictionary_based", {"window_inc": 1, "support_probabilities": True, "n_jobs": 1}, "weasel"),
            cls_only("aeon_muse", "tsc_dictionary", "MUSE", "aeon.classification.dictionary_based", {"window_inc": 1, "support_probabilities": True, "n_jobs": 1}, "muse", input_kind="aeon_collection_pad10"),
            cls_only("aeon_weasel_v2", "tsc_dictionary", "WEASEL_V2", "aeon.classification.dictionary_based", {"min_window": 1, "n_jobs": 1}, "weasel_v2"),
            cls_only("aeon_tde", "tsc_dictionary", "TemporalDictionaryEnsemble", "aeon.classification.dictionary_based", {"n_parameter_samples": 32, "min_window": 1, "max_ensemble_size": 20, "n_jobs": 1}, "tde"),
            cls_only("aeon_mrseql", "tsc_dictionary", "MrSEQLClassifier", "aeon.classification.dictionary_based", {}, "mrseql"),
            cls_only("aeon_mrsqm", "tsc_dictionary", "MrSQMClassifier", "aeon.classification.dictionary_based", {}, "mrsqm"),
            cls_only("aeon_shapelet_transform", "tsc_shapelet", "ShapeletTransformClassifier", "aeon.classification.shapelet_based", {"n_shapelet_samples": 256, "max_shapelets": 64, "max_shapelet_length": 1, "batch_size": 64, "n_jobs": 1}, "shapelet_transform"),
            cls_only("aeon_proximity_forest", "tsc_distance", "ProximityForest", "aeon.classification.distance_based", {"n_trees": 8, "n_jobs": 1}, "proximity_forest"),
            cls_only(
                "aeon_hivecotev2",
                "tsc_hybrid",
                "HIVECOTEV2",
                "aeon.classification.hybrid",
                {
                    "time_limit_in_minutes": 0.15,
                    "n_jobs": 1,
                    "stc_params": {"n_shapelet_samples": 128, "max_shapelets": 32, "max_shapelet_length": 1},
                    "drcif_params": {"n_estimators": 16, "min_interval_length": 1, "att_subsample_size": 4},
                    "arsenal_params": {"n_kernels": aeon_kernels, "n_estimators": 4},
                    "tde_params": {"n_parameter_samples": 16, "min_window": 1, "max_ensemble_size": 8},
                },
                "hivecotev2",
            ),
        ]
    )
    return specs


def build_deep_sequence_model_pool(deep_epochs: int, deep_batch_size: int) -> list[OfficialSpec]:
    """Official deep sequence estimators and official-layer Keras variants."""
    specs: list[OfficialSpec] = []

    recurrent_variants = [
        ("u16", {"units": 16}),
        ("u32", {"units": 32}),
        ("u64", {"units": 64}),
        ("stack2_u32", {"units": [32, 16]}),
        ("drop20_u32", {"units": 32, "dropout": 0.2}),
        ("dense16_u32", {"units": 32, "dense_units": [16]}),
        ("adamw_u32", {"units": 32, "optimizer": "adamw", "weight_decay": 1e-4}),
        ("lr3e4_u32", {"units": 32, "learning_rate": 3e-4}),
    ]
    for prefix, architecture in [("keras_lstm", "lstm"), ("keras_gru", "gru"), ("keras_bilstm", "bilstm")]:
        for suffix, params in recurrent_variants:
            specs.append(keras_sequence_pair(f"{prefix}_{suffix}", architecture, params, deep_epochs, deep_batch_size))

    for suffix, params in [
        ("filters8_k2_d1", {"nb_filters": 8, "kernel_size": 2, "dilations": (1,)}),
        ("filters16_k2_d1", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,)}),
        ("filters32_k2_d1", {"nb_filters": 32, "kernel_size": 2, "dilations": (1,)}),
        ("filters16_k3_d1", {"nb_filters": 16, "kernel_size": 3, "dilations": (1,)}),
        ("filters16_k2_d12", {"nb_filters": 16, "kernel_size": 2, "dilations": (1, 2)}),
        ("filters16_stack2", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,), "nb_stacks": 2}),
        ("filters16_drop10", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,), "dropout_rate": 0.1}),
        ("filters16_layernorm", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,), "use_layer_norm": True}),
        ("filters16_batchnorm", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,), "use_batch_norm": True}),
        ("filters16_noskip", {"nb_filters": 16, "kernel_size": 2, "dilations": (1,), "use_skip_connections": False}),
    ]:
        specs.append(keras_sequence_pair(f"keras_tcn_{suffix}", "tcn", params, deep_epochs, deep_batch_size, package="keras-tcn"))

    for suffix, params in [
        ("tiny_f8_k1", {"n_layers": 3, "n_filters": [8, 8, 8], "kernel_size": [1, 1, 1]}),
        ("small_f16_k1", {"n_layers": 3, "n_filters": [16, 16, 16], "kernel_size": [1, 1, 1]}),
        ("small_f16_k3", {"n_layers": 3, "n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]}),
        ("wide_f32_k1", {"n_layers": 3, "n_filters": [32, 32, 32], "kernel_size": [1, 1, 1]}),
        ("two_layer_f16", {"n_layers": 2, "n_filters": [16, 16], "kernel_size": [1, 1]}),
        ("dilated_f16", {"n_layers": 3, "n_filters": [16, 16, 16], "kernel_size": [1, 1, 1], "dilation_rate": [1, 2, 1]}),
    ]:
        specs.append(
            deep_pair(
                f"aeon_deep_fcn_{suffix}",
                "FCNClassifier",
                "FCNRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                params,
            )
        )

    for suffix, params in [
        ("block1_f16", {"n_residual_blocks": 1, "n_filters": [16, 16, 16], "kernel_size": [1, 1, 1]}),
        ("block2_f16", {"n_residual_blocks": 2, "n_filters": [16, 16, 16], "kernel_size": [1, 1, 1]}),
        ("block3_f16", {"n_residual_blocks": 3, "n_filters": [16, 16, 16], "kernel_size": [1, 1, 1]}),
        ("block2_f32", {"n_residual_blocks": 2, "n_filters": [32, 32, 32], "kernel_size": [1, 1, 1]}),
        ("block2_k3", {"n_residual_blocks": 2, "n_filters": [16, 16, 16], "kernel_size": [3, 3, 3]}),
        ("block2_dilated", {"n_residual_blocks": 2, "n_filters": [16, 16, 16], "kernel_size": [1, 1, 1], "dilation_rate": [1, 2, 1]}),
    ]:
        specs.append(
            deep_pair(
                f"aeon_deep_resnet_{suffix}",
                "ResNetClassifier",
                "ResNetRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                params,
            )
        )

    for suffix, params in [
        ("tiny_depth2_f8", {"n_classifiers": 1, "n_filters": 8, "kernel_size": 1, "depth": 2, "bottleneck_size": 4}),
        ("small_depth3_f16", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1, "depth": 3, "bottleneck_size": 8}),
        ("small_depth4_f16", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1, "depth": 4, "bottleneck_size": 8}),
        ("wide_depth3_f32", {"n_classifiers": 1, "n_filters": 32, "kernel_size": 1, "depth": 3, "bottleneck_size": 8}),
        ("no_residual", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1, "depth": 3, "bottleneck_size": 8, "use_residual": False}),
        ("no_bottleneck", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1, "depth": 3, "use_bottleneck": False}),
        ("pool_off", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1, "depth": 3, "bottleneck_size": 8, "use_max_pooling": False}),
        ("ensemble2", {"n_classifiers": 2, "n_filters": 8, "kernel_size": 1, "depth": 2, "bottleneck_size": 4}),
    ]:
        reg_params = dict(params)
        if "n_classifiers" in reg_params:
            reg_params["n_regressors"] = reg_params.pop("n_classifiers")
        specs.append(
            deep_pair(
                f"aeon_deep_inceptiontime_{suffix}",
                "InceptionTimeClassifier",
                "InceptionTimeRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                reg_params,
            )
        )

    for suffix, params in [
        ("f8_k1", {"n_layers": 2, "kernel_size": 1, "avg_pool_size": 1, "n_filters": [8, 8], "padding": "same"}),
        ("f16_k1", {"n_layers": 2, "kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16], "padding": "same"}),
        ("f16_k3", {"n_layers": 2, "kernel_size": 3, "avg_pool_size": 1, "n_filters": [16, 16], "padding": "same"}),
        ("three_layer", {"n_layers": 3, "kernel_size": 1, "avg_pool_size": 1, "n_filters": [16, 16, 16], "padding": "same"}),
    ]:
        specs.append(
            deep_pair(
                f"aeon_deep_timecnn_{suffix}",
                "TimeCNNClassifier",
                "TimeCNNRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                params,
            )
        )

    for suffix, params in [
        ("f8_k1", {"n_classifiers": 1, "n_filters": 8, "kernel_size": 1}),
        ("f16_k1", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 1}),
        ("f16_k3", {"n_classifiers": 1, "n_filters": 16, "kernel_size": 3}),
        ("mv_f16", {"n_classifiers": 1, "use_litemv": True, "n_filters": 16, "kernel_size": 1}),
    ]:
        reg_params = dict(params)
        if "n_classifiers" in reg_params:
            reg_params["n_regressors"] = reg_params.pop("n_classifiers")
        specs.append(
            deep_pair(
                f"aeon_deep_litetime_{suffix}",
                "LITETimeClassifier",
                "LITETimeRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                reg_params,
            )
        )

    for suffix, params in [
        ("f8_pool1", {"n_layers": 2, "n_filters": 8, "kernel_size": 1, "pool_size": 1, "hidden_fc_units": 32}),
        ("f16_pool1", {"n_layers": 2, "n_filters": 16, "kernel_size": 1, "pool_size": 1, "hidden_fc_units": 32}),
        ("f16_fc64", {"n_layers": 2, "n_filters": 16, "kernel_size": 1, "pool_size": 1, "hidden_fc_units": 64}),
        ("f16_elu3", {"n_layers": 3, "n_filters": 16, "kernel_size": 1, "pool_size": 1, "hidden_fc_units": 32}),
    ]:
        specs.append(
            deep_pair(
                f"aeon_deep_disjointcnn_{suffix}",
                "DisjointCNNClassifier",
                "DisjointCNNRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                params,
            )
        )

    for suffix, params in [
        ("depth2_f8", {"n_filters": 8, "kernel_size": 1, "depth": 2, "bottleneck_size": 4}),
        ("depth3_f16", {"n_filters": 16, "kernel_size": 1, "depth": 3, "bottleneck_size": 8}),
        ("no_residual", {"n_filters": 16, "kernel_size": 1, "depth": 3, "bottleneck_size": 8, "use_residual": False}),
    ]:
        specs.append(
            deep_pair(
                f"aeon_deep_individual_inception_{suffix}",
                "IndividualInceptionClassifier",
                "IndividualInceptionRegressor",
                deep_epochs,
                deep_batch_size,
                params,
                params,
            )
        )

    return specs


def adaboost_tree_classifier_factory() -> Factory:
    def factory(seed: int):
        import inspect

        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        params = {
            "estimator": DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=seed),
            "n_estimators": 80,
            "learning_rate": 0.05,
            "random_state": seed,
        }
        if "algorithm" in inspect.signature(AdaBoostClassifier).parameters:
            params["algorithm"] = "SAMME"
        return AdaBoostClassifier(**params)

    return factory


def spec_pair(
    name: str,
    family: str,
    cls_name: str,
    reg_name: str,
    cls_module: str,
    reg_module: str,
    cls_kwargs: dict[str, object],
    reg_kwargs: dict[str, object],
    classifier_loss: str,
    regressor_loss: str,
    input_kind: str = "aeon_collection",
) -> OfficialSpec:
    return OfficialSpec(
        name=name,
        family=family,
        package="aeon",
        classifier_factory=aeon_factory(cls_module, cls_name, cls_kwargs),
        regressor_factory=aeon_factory(reg_module, reg_name, reg_kwargs),
        classifier_loss=classifier_loss,
        regressor_loss=regressor_loss,
        input_kind=input_kind,
        source_kind="official_aeon",
    )


def deep_pair(
    name: str,
    cls_name: str,
    reg_name: str,
    epochs: int,
    batch_size: int,
    cls_extra: dict[str, object],
    reg_extra: dict[str, object],
) -> OfficialSpec:
    base = {
        "n_epochs": epochs,
        "batch_size": batch_size,
        "verbose": False,
        "save_best_model": False,
        "save_last_model": False,
        "save_init_model": False,
    }
    cls_kwargs = {**base, **cls_extra}
    reg_kwargs = {**base, **reg_extra}
    return spec_pair(
        name=name,
        family="tsc_deep_learning",
        cls_name=cls_name,
        reg_name=reg_name,
        cls_module="aeon.classification.deep_learning",
        reg_module="aeon.regression.deep_learning",
        cls_kwargs=cls_kwargs,
        reg_kwargs=reg_kwargs,
        classifier_loss="deep_categorical_crossentropy",
        regressor_loss="deep_mean_squared_error",
    )


def keras_sequence_pair(
    name: str,
    architecture: str,
    params: dict[str, object],
    epochs: int,
    batch_size: int,
    package: str = "tensorflow.keras",
) -> OfficialSpec:
    cls_params = dict(params)
    reg_params = dict(params)

    def cls_factory(seed: int):
        return KerasSequenceClassifier(architecture, cls_params, epochs, batch_size, seed)

    def reg_factory(seed: int):
        return KerasSequenceRegressor(architecture, reg_params, epochs, batch_size, seed)

    return OfficialSpec(
        name=name,
        family="deep_sequence",
        package=package,
        classifier_factory=cls_factory,
        regressor_factory=reg_factory,
        classifier_loss="binary_crossentropy",
        regressor_loss="mean_squared_error",
        input_kind="keras_sequence",
        source_kind="official_keras_layers",
    )


def cls_only(
    name: str,
    family: str,
    class_name: str,
    module: str,
    kwargs: dict[str, object],
    loss: str,
    input_kind: str = "aeon_collection",
) -> OfficialSpec:
    return OfficialSpec(
        name=name,
        family=family,
        package="aeon",
        classifier_factory=aeon_factory(module, class_name, kwargs),
        regressor_factory=None,
        classifier_loss=loss,
        regressor_loss="not_applicable_no_official_paired_regressor",
        input_kind=input_kind,
        source_kind="official_aeon_classifier_only",
    )


def aeon_factory(module_name: str, class_name: str, kwargs: dict[str, object]) -> Factory:
    def factory(seed: int):
        if "deep_learning" in module_name:
            configure_tensorflow_runtime()
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        params = dict(kwargs)
        try:
            import inspect

            signature = inspect.signature(cls)
            if "random_state" in signature.parameters and "random_state" not in params:
                params["random_state"] = seed
        except (TypeError, ValueError):
            pass
        return cls(**params)

    return factory


def select_models(
    requested: str,
    seed: int,
    aeon_kernels: int,
    aeon_estimators: int,
    deep_epochs: int,
    deep_batch_size: int,
) -> list[OfficialSpec]:
    zoo = build_official_model_pool(seed, aeon_kernels, aeon_estimators, deep_epochs, deep_batch_size)
    deep_sequence_zoo = build_deep_sequence_model_pool(deep_epochs, deep_batch_size)
    if requested == "all":
        return zoo
    if requested == "deep_sequence_50plus":
        return deep_sequence_zoo
    wanted = parse_csv_list(requested)
    by_name = {m.name: m for m in [*zoo, *deep_sequence_zoo]}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"Unknown model(s): {missing}")
    return [by_name[name] for name in wanted]


def model_to_manifest(spec: OfficialSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "family": spec.family,
        "package": spec.package,
        "source_kind": spec.source_kind,
        "input_kind": spec.input_kind,
        "has_classifier": spec.classifier_factory is not None,
        "has_regressor": spec.regressor_factory is not None,
        "classifier_loss": spec.classifier_loss,
        "regressor_loss": spec.regressor_loss,
    }


def collect_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "aeon", "tensorflow", "keras", "tcn"]:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"unavailable: {type(exc).__name__}"
    return versions


def official_sources() -> dict[str, str]:
    return {
        "scikit-learn supervised learning": "https://scikit-learn.org/stable/supervised_learning.html",
        "LightGBM Python API": "https://lightgbm.readthedocs.io/en/stable/Python-API.html",
        "XGBoost Python API": "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
        "CatBoost Python reference": "https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier",
        "aeon classification API": "https://www.aeon-toolkit.org/en/stable/api_reference/classification.html",
        "aeon regression API": "https://www.aeon-toolkit.org/en/stable/api_reference/regression.html",
        "aeon deep learning classifiers": "https://www.aeon-toolkit.org/en/stable/api_reference/classification.html#deep-learning",
        "aeon deep learning regressors": "https://www.aeon-toolkit.org/en/stable/api_reference/regression.html#deep-learning",
        "aeon data format": "https://www.aeon-toolkit.org/en/stable/api_reference/data_format.html",
        "TensorFlow official install": "https://www.tensorflow.org/install/pip",
        "Keras LSTM layer": "https://keras.io/api/layers/recurrent_layers/lstm/",
        "Keras GRU layer": "https://keras.io/api/layers/recurrent_layers/gru/",
        "Keras Bidirectional layer": "https://keras.io/api/layers/recurrent_layers/bidirectional/",
        "keras-tcn TCN layer": "https://github.com/philipperemy/keras-tcn",
    }


def error_row(spec: OfficialSpec, feature_set: str, lookback: int, horizon: int, origin_id: int, phase: str, exc: Exception) -> dict[str, object]:
    return {
        "feature_set": feature_set,
        "model": spec.name,
        "family": spec.family,
        "package": spec.package,
        "input_kind": spec.input_kind,
        "lookback_months": lookback,
        "horizon_months": horizon,
        "origin_id": origin_id,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]),
    }


def finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


if __name__ == "__main__":
    main()
