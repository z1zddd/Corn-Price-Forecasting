from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.machine_learning.RandomForest import RandomForestPriceRegressor
from scripts.data_processing.data_pipeline import (
    FixedSplit,
    FoldPreprocessor,
    SupervisedSamples,
    assert_no_temporal_leakage,
    build_supervised_samples,
    iter_expanding_origins,
    load_daily_data,
    make_fixed_split,
)
from scripts.evaluation.evaluate import evaluate_predictions


class ExperimentContractError(RuntimeError):
    """Raised when a formal run cannot satisfy the complete backtest contract."""


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def make_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


FIXED_STRATEGY_RATIOS = {
    "chronological_811": (0.8, 0.1, 0.1),
    "chronological_712": (0.7, 0.1, 0.2),
}
EXPANDING_STRATEGY = "expanding_rolling_backtest"


def resolve_strategy_plan(config: dict[str, Any]) -> dict[str, Any]:
    configured = list(config["split"]["strategies"])
    supported = {*FIXED_STRATEGY_RATIOS, EXPANDING_STRATEGY}
    unsupported = sorted(set(configured).difference(supported))
    if not configured:
        raise ValueError("At least one split strategy must be configured")
    if unsupported:
        raise ValueError(f"Unsupported split strategies: {unsupported}")

    fixed = [strategy for strategy in configured if strategy in FIXED_STRATEGY_RATIOS]
    tuning = list(fixed)
    run_expanding = EXPANDING_STRATEGY in configured
    if run_expanding and "chronological_712" not in tuning:
        tuning.append("chronological_712")
    return {
        "configured": configured,
        "fixed": fixed,
        "tuning": tuning,
        "run_expanding": run_expanding,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def _higher_is_worse(metric_name: str) -> bool:
    return any(
        token in metric_name
        for token in (
            "rmse",
            "mae",
            "mape",
            "smape",
            "mase",
            "max_drawdown",
            "turnover",
            "trade_count",
        )
    ) and "improvement" not in metric_name


def build_seed_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model", "feature_set", "horizon", "lookback", "split_strategy"]
    excluded = set(group_columns + ["seed", "n_predictions"])
    metric_columns = [
        column
        for column in metrics.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys, strict=True))
        row["seed_count"] = int(group["seed"].nunique())
        row["n_predictions_per_seed"] = int(group["n_predictions"].min())
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_worst"] = (
                float(values.max() if _higher_is_worse(metric) else values.min())
                if not values.empty
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_formal_coverage(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
    expected_counts: dict[tuple[int, int, str], int],
) -> pd.DataFrame:
    key_columns = ["horizon", "lookback", "split_strategy", "seed"]
    expected = {
        (int(horizon), int(lookback), strategy, int(seed))
        for horizon in config["target"]["horizons"]
        for lookback in config["lookback"]["candidates"]
        for strategy in config["split"]["strategies"]
        for seed in config["formal"]["seeds"]
    }
    actual_metrics = {
        tuple(row)
        for row in metrics[key_columns].itertuples(index=False, name=None)
    }
    actual_predictions = {
        tuple(row)
        for row in predictions[key_columns].drop_duplicates().itertuples(index=False, name=None)
    }
    missing_metrics = expected.difference(actual_metrics)
    unexpected_metrics = actual_metrics.difference(expected)
    missing_predictions = expected.difference(actual_predictions)
    if missing_metrics or unexpected_metrics or missing_predictions:
        raise ExperimentContractError(
            "Formal coverage mismatch: "
            f"missing_metrics={sorted(missing_metrics)}, "
            f"unexpected_metrics={sorted(unexpected_metrics)}, "
            f"missing_predictions={sorted(missing_predictions)}"
        )
    if metrics.duplicated(key_columns).any():
        raise ExperimentContractError("Duplicate formal metric combinations detected")

    coverage_rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        horizon, lookback, strategy, seed = key
        group = predictions[
            (predictions["horizon"] == horizon)
            & (predictions["lookback"] == lookback)
            & (predictions["split_strategy"] == strategy)
            & (predictions["seed"] == seed)
        ]
        expected_count = int(expected_counts[(horizon, lookback, strategy)])
        metric_count = int(
            metrics.loc[
                (metrics["horizon"] == horizon)
                & (metrics["lookback"] == lookback)
                & (metrics["split_strategy"] == strategy)
                & (metrics["seed"] == seed),
                "n_predictions",
            ].iloc[0]
        )
        valid = (
            len(group) == expected_count
            and metric_count == expected_count
            and group["anchor_date"].nunique() == expected_count
            and np.isfinite(group["predicted_dce_corn_close"].to_numpy(dtype=float)).all()
        )
        coverage_rows.append(
            {
                "horizon": horizon,
                "lookback": lookback,
                "split_strategy": strategy,
                "seed": seed,
                "expected_predictions": expected_count,
                "actual_predictions": len(group),
                "unique_anchor_dates": int(group["anchor_date"].nunique()),
                "status": "PASS" if valid else "FAIL",
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    if (coverage["status"] != "PASS").any():
        raise ExperimentContractError("One or more formal prediction groups failed coverage")
    return coverage


def raise_formal_failure(
    result_root: Path,
    failures: list[dict[str, Any]],
    horizon: int,
    lookback: int,
    split_strategy: str,
    seed: int,
    error: Exception,
) -> None:
    failure = {
        "horizon": horizon,
        "lookback": lookback,
        "split_strategy": split_strategy,
        "seed": seed,
        "error": repr(error),
    }
    failures.append(failure)
    _atomic_write_csv(
        result_root / "model_failures.csv",
        pd.DataFrame(
            failures,
            columns=["horizon", "lookback", "split_strategy", "seed", "error"],
        ),
    )
    raise ExperimentContractError(
        "Formal setting failed: "
        f"horizon={horizon}, lookback={lookback}, "
        f"split={split_strategy}, seed={seed}"
    ) from error


def _mase_scale(y_train: Iterable[float]) -> float:
    values = np.asarray(list(y_train), dtype=float)
    if values.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(values))))


def _columns_hash(columns: Iterable[str]) -> str:
    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_params(config: dict[str, Any], candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    params = dict(config["model"]["fixed_params"])
    params.update(candidate)
    params["random_state"] = int(seed)
    return params


def _prepare_fold(
    samples: SupervisedSamples,
    train_idx: np.ndarray,
    evaluation_idx: np.ndarray,
    max_missing_rate: float,
) -> tuple[FoldPreprocessor, np.ndarray, np.ndarray]:
    preprocessor = FoldPreprocessor(max_missing_rate=max_missing_rate)
    X_train = preprocessor.fit_transform(samples.X.iloc[train_idx])
    X_evaluation = preprocessor.transform(samples.X.iloc[evaluation_idx])
    return preprocessor, X_train, X_evaluation


def tune_parameters(
    samples: SupervisedSamples,
    split: FixedSplit,
    config: dict[str, Any],
    split_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_missing_rate = float(config["feature_set"]["max_training_missing_rate"])
    preprocessor, X_train, X_validation = _prepare_fold(
        samples, split.train_idx, split.validation_idx, max_missing_rate
    )
    y_train = samples.y.iloc[split.train_idx].to_numpy(dtype=float)
    y_validation = samples.y.iloc[split.validation_idx].to_numpy(dtype=float)
    results: list[dict[str, Any]] = []
    for candidate in make_parameter_grid(config["tuning"]["grid"]):
        params = _model_params(config, candidate, int(config["tuning"]["seed"]))
        model = RandomForestPriceRegressor(**params).fit(X_train, y_train)
        predicted = model.predict(X_validation)
        rmse = float(np.sqrt(np.mean((y_validation - predicted) ** 2)))
        results.append(
            {
                "split_tuning_skeleton": split_name,
                "validation_rmse": rmse,
                "selected_feature_count": len(preprocessor.selected_columns),
                **candidate,
            }
        )
    best = min(results, key=lambda item: item["validation_rmse"])
    candidate_names = set(config["tuning"]["grid"])
    return {key: best[key] for key in candidate_names}, results


def _prediction_frame(
    samples: SupervisedSamples,
    indices: np.ndarray,
    predicted: np.ndarray,
    mase_scale: float,
) -> pd.DataFrame:
    frame = samples.metadata.iloc[indices].reset_index(drop=True).copy()
    frame["actual_dce_corn_close"] = samples.y.iloc[indices].to_numpy(dtype=float)
    frame["predicted_dce_corn_close"] = np.asarray(predicted, dtype=float)
    frame["mase_scale"] = mase_scale
    return frame


def run_fixed_setting(
    samples: SupervisedSamples,
    split: FixedSplit,
    config: dict[str, Any],
    best_params: dict[str, Any],
    seed: int,
    checkpoint_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    max_missing_rate = float(config["feature_set"]["max_training_missing_rate"])
    preprocessor, X_train, X_test = _prepare_fold(
        samples, split.refit_idx, split.test_idx, max_missing_rate
    )
    first_test_anchor = samples.metadata.iloc[split.test_idx[0]]["anchor_date"]
    assert_no_temporal_leakage(samples.metadata.iloc[split.refit_idx], first_test_anchor)
    params = _model_params(config, best_params, seed)
    model = RandomForestPriceRegressor(**params).fit(
        X_train, samples.y.iloc[split.refit_idx].to_numpy(dtype=float)
    )
    predicted = model.predict(X_test)
    scale = _mase_scale(samples.y.iloc[split.refit_idx])
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "params": params,
                "selected_columns": preprocessor.selected_columns,
            },
            checkpoint_path,
        )
    audit = {
        "train_start_date": samples.metadata.iloc[split.refit_idx[0]]["anchor_date"],
        "train_end_date": samples.metadata.iloc[split.refit_idx[-1]]["anchor_date"],
        "train_max_target_date": samples.metadata.iloc[split.refit_idx]["target_date"].max(),
        "validation_start_date": samples.metadata.iloc[split.validation_idx[0]]["anchor_date"],
        "validation_end_date": samples.metadata.iloc[split.validation_idx[-1]]["anchor_date"],
        "test_start_date": samples.metadata.iloc[split.test_idx[0]]["anchor_date"],
        "test_end_date": samples.metadata.iloc[split.test_idx[-1]]["anchor_date"],
        "prediction_anchor_date": samples.metadata.iloc[split.test_idx[0]]["anchor_date"],
        "prediction_target_date": samples.metadata.iloc[split.test_idx[0]]["target_date"],
        "n_train": len(split.refit_idx),
        "n_predictions": len(split.test_idx),
        "selected_feature_count": len(preprocessor.selected_columns),
        "selected_columns_hash": _columns_hash(preprocessor.selected_columns),
        "selected_columns": preprocessor.selected_columns,
    }
    return _prediction_frame(samples, split.test_idx, predicted, scale), params, audit


def run_expanding_setting(
    samples: SupervisedSamples,
    test_idx: np.ndarray,
    config: dict[str, Any],
    best_params: dict[str, Any],
    seed: int,
    checkpoint_path: Path | None,
    limit_origins: int | None = None,
    progress_dir: Path | None = None,
    flush_interval: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[str]]:
    if flush_interval < 1:
        raise ValueError("flush_interval must be at least 1")

    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    last_bundle: dict[str, Any] | None = None
    params = _model_params(config, best_params, seed)
    max_missing_rate = float(config["feature_set"]["max_training_missing_rate"])
    completed_positions: set[int] = set()
    progress_predictions_path: Path | None = None
    progress_audits_path: Path | None = None
    progress_state_path: Path | None = None

    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)
        progress_predictions_path = progress_dir / "predictions.csv"
        progress_audits_path = progress_dir / "fold_audit.csv"
        progress_state_path = progress_dir / "progress_state.json"
        if progress_predictions_path.exists():
            existing_predictions = pd.read_csv(progress_predictions_path)
            if not existing_predictions.empty:
                rows.append(existing_predictions)
                completed_positions.update(
                    existing_predictions["sample_position"].astype(int).tolist()
                )
        if progress_audits_path.exists():
            audits.extend(pd.read_csv(progress_audits_path).to_dict(orient="records"))
        if checkpoint_path is not None and checkpoint_path.exists():
            last_bundle = joblib.load(checkpoint_path)

    expected_positions = {int(position) for position in test_idx}
    processed_since_flush = 0
    processed_this_call = 0

    def flush_progress(status: str) -> None:
        nonlocal processed_since_flush
        if progress_dir is None:
            return
        predictions_frame = (
            pd.concat(rows, ignore_index=True)
            .drop_duplicates("sample_position", keep="last")
            .sort_values("sample_position")
            .reset_index(drop=True)
        )
        audits_frame = (
            pd.DataFrame(audits)
            .drop_duplicates("sample_position", keep="last")
            .sort_values("sample_position")
            .reset_index(drop=True)
        )
        _atomic_write_csv(progress_predictions_path, predictions_frame)
        _atomic_write_csv(progress_audits_path, audits_frame)
        if checkpoint_path is not None and last_bundle is not None:
            _atomic_joblib(checkpoint_path, last_bundle)
        _atomic_write_json(
            progress_state_path,
            {
                "status": status,
                "completed_origins": len(completed_positions),
                "total_origins": len(expected_positions),
                "updated_at": datetime.now(),
            },
        )
        processed_since_flush = 0

    for origin in iter_expanding_origins(samples, test_idx):
        if origin.prediction_idx in completed_positions:
            continue
        if limit_origins is not None and processed_this_call >= limit_origins:
            break
        prediction_idx = np.asarray([origin.prediction_idx], dtype=int)
        preprocessor, X_train, X_prediction = _prepare_fold(
            samples, origin.train_idx, prediction_idx, max_missing_rate
        )
        y_train = samples.y.iloc[origin.train_idx].to_numpy(dtype=float)
        model = RandomForestPriceRegressor(**params).fit(X_train, y_train)
        predicted = model.predict(X_prediction)
        scale = _mase_scale(y_train)
        prediction_frame = _prediction_frame(samples, prediction_idx, predicted, scale)
        prediction_frame["sample_position"] = origin.prediction_idx
        rows.append(prediction_frame)
        train_meta = samples.metadata.iloc[origin.train_idx]
        audits.append(
            {
                "train_start_date": train_meta.iloc[0]["anchor_date"],
                "train_end_date": train_meta.iloc[-1]["anchor_date"],
                "train_max_target_date": train_meta["target_date"].max(),
                "test_start_date": samples.metadata.iloc[test_idx[0]]["anchor_date"],
                "test_end_date": samples.metadata.iloc[test_idx[-1]]["anchor_date"],
                "prediction_anchor_date": origin.prediction_anchor_date,
                "prediction_target_date": samples.metadata.iloc[origin.prediction_idx]["target_date"],
                "sample_position": origin.prediction_idx,
                "n_train": len(origin.train_idx),
                "n_predictions": 1,
                "selected_feature_count": len(preprocessor.selected_columns),
                "selected_columns_hash": _columns_hash(preprocessor.selected_columns),
            }
        )
        last_bundle = {
            "model": model,
            "preprocessor": preprocessor,
            "params": params,
            "selected_columns": preprocessor.selected_columns,
        }
        completed_positions.add(origin.prediction_idx)
        processed_this_call += 1
        processed_since_flush += 1
        if processed_since_flush >= flush_interval:
            flush_progress("RUNNING")
    if not rows or last_bundle is None:
        raise ValueError("No expanding predictions were produced")
    status = (
        "COMPLETED"
        if expected_positions.issubset(completed_positions)
        else "RUNNING"
    )
    if progress_dir is not None:
        flush_progress(status)
    elif checkpoint_path is not None:
        _atomic_joblib(checkpoint_path, last_bundle)
    combined_predictions = (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates("sample_position", keep="last")
        .sort_values("sample_position")
        .reset_index(drop=True)
    )
    combined_audits = (
        pd.DataFrame(audits)
        .drop_duplicates("sample_position", keep="last")
        .sort_values("sample_position")
        .to_dict(orient="records")
    )
    return (
        combined_predictions,
        params,
        combined_audits,
        list(last_bundle["selected_columns"]),
    )


def _data_audit(frame: pd.DataFrame, data_path: Path) -> dict[str, Any]:
    missing = frame.isna().sum()
    return {
        "path": str(data_path),
        "sha256": _sha256(data_path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "date_min": frame["date"].min(),
        "date_max": frame["date"].max(),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "target_missing": int(frame["dce_corn_close"].isna().sum()),
        "rows_with_any_missing": int(frame.isna().any(axis=1).sum()),
        "missing_by_column": {key: int(value) for key, value in missing.items() if value},
    }


def run_preflight(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    strategy_plan = resolve_strategy_plan(config)
    data_path = project_root / config["data"]["csv_path"]
    if _sha256(data_path) != config["data"]["expected_sha256"]:
        raise ValueError("Dataset SHA-256 does not match config")
    frame = load_daily_data(data_path)
    data_audit = _data_audit(frame, data_path)
    data_audit["path"] = Path(config["data"]["csv_path"]).as_posix()
    horizon = max(config["target"]["horizons"])
    lookback = max(config["lookback"]["candidates"])
    samples = build_supervised_samples(
        frame,
        horizon=horizon,
        lookback=lookback,
        external_lag=int(config["feature_set"]["external_lag"]),
    )
    smoke_strategy = strategy_plan["tuning"][0]
    split = make_fixed_split(
        samples,
        ratios=FIXED_STRATEGY_RATIOS[smoke_strategy],
        embargo=horizon,
    )
    candidate = {
        "n_estimators": 10,
        "max_depth": 8,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = Path(temporary_directory) / "smoke.joblib"
        predictions, _, _ = run_fixed_setting(
            samples,
            split,
            config,
            candidate,
            seed=42,
            checkpoint_path=checkpoint,
        )
        loaded = joblib.load(checkpoint)
        if not isinstance(loaded["model"], RandomForestPriceRegressor):
            raise TypeError("Checkpoint round-trip failed")
        expanding_smoke_predictions = 0
        if strategy_plan["run_expanding"]:
            expanding_predictions, _, expanding_audits, _ = run_expanding_setting(
                samples,
                split.test_idx,
                config,
                candidate,
                seed=42,
                checkpoint_path=Path(temporary_directory) / "expanding_smoke.joblib",
                limit_origins=2,
            )
            if len(expanding_predictions) != 2 or len(expanding_audits) != 2:
                raise AssertionError("Expanding smoke test did not produce two audited origins")
            expanding_smoke_predictions = len(expanding_predictions)
    smoke_metrics, _ = evaluate_predictions(
        predictions,
        horizon=horizon,
        mase_scale=float(predictions["mase_scale"].iloc[0]),
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
    )

    benchmark_candidate = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 0.5,
    }
    benchmark_indices = split.refit_idx
    preprocessor, X_train, _ = _prepare_fold(
        samples,
        benchmark_indices,
        split.test_idx[:1],
        float(config["feature_set"]["max_training_missing_rate"]),
    )
    del preprocessor
    started = time.perf_counter()
    RandomForestPriceRegressor(
        **_model_params(config, benchmark_candidate, 42)
    ).fit(X_train, samples.y.iloc[benchmark_indices].to_numpy(dtype=float))
    benchmark_seconds = time.perf_counter() - started

    grid_size = len(make_parameter_grid(config["tuning"]["grid"]))
    combinations = len(config["target"]["horizons"]) * len(
        config["lookback"]["candidates"]
    )
    tuning_fits = combinations * len(strategy_plan["tuning"]) * grid_size
    fixed_fits = (
        combinations
        * len(strategy_plan["fixed"])
        * len(config["formal"]["seeds"])
    )
    expanding_fits = 0
    if strategy_plan["run_expanding"]:
        for current_horizon in config["target"]["horizons"]:
            for current_lookback in config["lookback"]["candidates"]:
                current_samples = build_supervised_samples(
                    frame,
                    horizon=current_horizon,
                    lookback=current_lookback,
                    external_lag=int(config["feature_set"]["external_lag"]),
                )
                current_split = make_fixed_split(
                    current_samples,
                    ratios=FIXED_STRATEGY_RATIOS["chronological_712"],
                    embargo=current_horizon,
                )
                expanding_fits += len(current_split.test_idx) * len(
                    config["formal"]["seeds"]
                )
    total_fits = tuning_fits + fixed_fits + expanding_fits
    return {
        "status": "PASS",
        "data_audit": data_audit,
        "smoke_price_rmse": smoke_metrics["price_rmse"],
        "expanding_smoke_predictions": expanding_smoke_predictions,
        "benchmark_worst_grid_fit_seconds": benchmark_seconds,
        "tuning_fits": tuning_fits,
        "fixed_fits": fixed_fits,
        "expanding_fits": expanding_fits,
        "total_fits": total_fits,
        "estimated_serial_hours_at_benchmark_rate": benchmark_seconds * total_fits / 3600.0,
    }


def _decorate(
    frame: pd.DataFrame,
    model: str,
    feature_set: str,
    horizon: int,
    lookback: int,
    split_strategy: str,
    seed: int,
) -> pd.DataFrame:
    decorated = frame.copy()
    decorated["horizon"] = horizon
    decorated["lookback"] = lookback
    decorated["split_strategy"] = split_strategy
    decorated["model"] = model
    decorated["feature_set"] = feature_set
    decorated["seed"] = seed
    return decorated


def _save_group(
    result_root: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    horizon: int,
    lookback: int,
    split_strategy: str,
    model_name: str,
    seed: int,
    fold_audits: list[dict[str, Any]],
    selected_columns: list[str],
    config: dict[str, Any],
) -> None:
    group = (
        result_root
        / f"horizon_{horizon}"
        / f"lookback_{lookback}"
        / split_strategy
        / model_name
        / f"seed_{seed}"
    )
    group.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(group / "predictions.csv", index=False)
    _write_json(group / "metrics.json", metrics)
    pd.DataFrame(fold_audits).to_csv(group / "fold_audit.csv", index=False)
    _write_json(
        group / "preprocessing_manifest.json",
        {
            "fit_on_train_only": True,
            "external_lag": config["feature_set"]["external_lag"],
            "max_training_missing_rate": config["feature_set"][
                "max_training_missing_rate"
            ],
            "selected_feature_count_final": len(selected_columns),
            "selected_columns_hash_final": _columns_hash(selected_columns),
            "selected_columns_final": selected_columns,
        },
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_report(
    path: Path,
    run_id: str,
    data_audit: dict[str, Any],
    metrics: pd.DataFrame,
    failures: pd.DataFrame,
    config: dict[str, Any],
    seed_stability: pd.DataFrame,
) -> None:
    summary = (
        metrics.groupby("split_strategy", as_index=False)[
            ["price_rmse", "price_mae", "trend_direction_accuracy"]
        ]
        .mean()
        .sort_values("split_strategy")
    )
    detail_columns = [
        "horizon",
        "lookback",
        "split_strategy",
        "seed",
        "n_predictions",
        "price_rmse",
        "price_mae",
        "trend_direction_accuracy",
        "economic_0bp_mean_cumulative_return",
        "economic_0bp_mean_sharpe",
        "economic_0bp_mean_max_drawdown",
        "economic_10bp_mean_cumulative_return",
    ]
    detail = metrics[[column for column in detail_columns if column in metrics]].copy()
    stability_columns = [
        "horizon",
        "lookback",
        "split_strategy",
        "seed_count",
        "price_rmse_mean",
        "price_rmse_std",
        "price_rmse_worst",
        "trend_direction_accuracy_mean",
        "trend_direction_accuracy_std",
        "trend_direction_accuracy_worst",
    ]
    stability = seed_stability[
        [column for column in stability_columns if column in seed_stability]
    ].copy()
    lines = [
        "# Random Forest 日度回测报告",
        "",
        f"- 运行标识：`{run_id}`",
        f"- 数据哈希：`{data_audit['sha256']}`",
        f"- 数据范围：{data_audit['date_min']} 至 {data_audit['date_max']}",
        f"- 样本：{data_audit['rows']} 行，{data_audit['columns']} 列",
        "- 目标：直接预测未来 `dce_corn_close`",
        "- 趋势与经济指标：由预测价格派生",
        "",
        "## 已配置划分的测试结果",
        "",
        _markdown_table(summary),
        "",
        "## Horizon / lookback 明细",
        "",
        _markdown_table(detail),
        "",
        "## 种子稳定性（均值 / 标准差 / 最差）",
        "",
        _markdown_table(stability),
        "",
        "## 运行范围",
        "",
        f"- horizons：{config['target']['horizons']}",
        f"- lookbacks：{config['lookback']['candidates']}",
        f"- 数据划分：{list(config['split']['strategies'])}",
        f"- seeds：{config['formal']['seeds']}",
        f"- 特征：`{config['feature_set']['name']}`",
        "",
        "## 失败记录",
        "",
        "无。" if failures.empty else _markdown_table(failures),
        "",
        "## 限制",
        "",
        "经济指标使用非重叠持有期子序列，不能解释为逐日多周期收益直接连乘。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_formal(
    config: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    strategy_plan = resolve_strategy_plan(config)
    runner = config["run"]["runner"]
    if not runner or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for character in runner):
        raise ValueError("runner must contain only letters, numbers, or hyphens")
    run_id = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{runner}"
    dataset_name = config["run"]["dataset_name"]
    model_name = config["run"]["model_name"]
    result_root = project_root / "results" / dataset_name / model_name / run_id
    checkpoint_root = project_root / "checkpoints" / dataset_name / model_name / run_id
    report_root = project_root / "report" / dataset_name / model_name / run_id
    result_root.mkdir(parents=True, exist_ok=False)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    report_root.mkdir(parents=True, exist_ok=False)

    data_path = project_root / config["data"]["csv_path"]
    if _sha256(data_path) != config["data"]["expected_sha256"]:
        raise ValueError("Dataset SHA-256 does not match config")
    frame = load_daily_data(data_path)
    data_audit = _data_audit(frame, data_path)
    data_audit["path"] = Path(config["data"]["csv_path"]).as_posix()
    _write_json(result_root / "data_audit.json", data_audit)
    resolved = deepcopy(config)
    resolved["run"]["run_id"] = run_id
    resolved["run"]["source_commit"] = source_commit
    (result_root / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "status": "RUNNING",
        "runner": runner,
        "started_at": datetime.now(),
        "operating_system": platform.platform(),
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "source_commit": source_commit,
        "data_sha256": data_audit["sha256"],
    }
    _write_json(result_root / "experiment_manifest.json", manifest)

    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_counts: dict[tuple[int, int, str], int] = {}
    try:
        for horizon in config["target"]["horizons"]:
            for lookback in config["lookback"]["candidates"]:
                samples = build_supervised_samples(
                    frame,
                    horizon=horizon,
                    lookback=lookback,
                    external_lag=int(config["feature_set"]["external_lag"]),
                )
                splits = {
                    name: make_fixed_split(
                        samples, FIXED_STRATEGY_RATIOS[name], embargo=horizon
                    )
                    for name in strategy_plan["tuning"]
                }
                best_by_strategy: dict[str, dict[str, Any]] = {}
                for split_name in strategy_plan["tuning"]:
                    best_params, split_tuning_rows = tune_parameters(
                        samples, splits[split_name], config, split_name
                    )
                    best_by_strategy[split_name] = best_params
                    for row in split_tuning_rows:
                        tuning_rows.append(
                            {"horizon": horizon, "lookback": lookback, **row}
                        )

                settings = [
                    (split_name, splits[split_name], best_by_strategy[split_name])
                    for split_name in strategy_plan["fixed"]
                ]
                for split_name, split, _ in settings:
                    expected_counts[(horizon, lookback, split_name)] = len(split.test_idx)
                if strategy_plan["run_expanding"]:
                    expected_counts[(horizon, lookback, EXPANDING_STRATEGY)] = len(
                        splits["chronological_712"].test_idx
                    )
                for split_name, split, best_params in settings:
                    for seed in config["formal"]["seeds"]:
                        try:
                            checkpoint = (
                                checkpoint_root
                                / f"horizon_{horizon}"
                                / f"lookback_{lookback}"
                                / split_name
                                / f"seed_{seed}.joblib"
                            )
                            predictions, params, audit = run_fixed_setting(
                                samples, split, config, best_params, seed, checkpoint
                            )
                            metrics, enriched = evaluate_predictions(
                                predictions,
                                horizon,
                                float(predictions["mase_scale"].iloc[0]),
                                config["evaluation"]["transaction_cost_bps"],
                            )
                            enriched = _decorate(
                                enriched,
                                model_name,
                                config["feature_set"]["name"],
                                horizon,
                                lookback,
                                split_name,
                                seed,
                            )
                            metric_row = {
                                "model": model_name,
                                "feature_set": config["feature_set"]["name"],
                                "horizon": horizon,
                                "lookback": lookback,
                                "split_strategy": split_name,
                                "seed": seed,
                                "test_start": enriched["anchor_date"].min(),
                                "test_end": enriched["anchor_date"].max(),
                                "n_predictions": len(enriched),
                                "params": json.dumps(params, sort_keys=True),
                                **metrics,
                            }
                            all_predictions.append(enriched)
                            metric_rows.append(metric_row)
                            selected_columns = list(audit["selected_columns"])
                            fold_audit = {
                                key: value
                                for key, value in audit.items()
                                if key != "selected_columns"
                            }
                            audit_rows.append(
                                {
                                    "horizon": horizon,
                                    "lookback": lookback,
                                    "split_strategy": split_name,
                                    "seed": seed,
                                    **fold_audit,
                                }
                            )
                            _save_group(
                                result_root,
                                enriched,
                                metric_row,
                                horizon,
                                lookback,
                                split_name,
                                model_name,
                                seed,
                                [fold_audit],
                                selected_columns,
                                config,
                            )
                        except Exception as error:
                            raise_formal_failure(
                                result_root,
                                failures,
                                horizon,
                                lookback,
                                split_name,
                                seed,
                                error,
                            )

                if strategy_plan["run_expanding"]:
                    expanding_split = splits["chronological_712"]
                    expanding_params = best_by_strategy["chronological_712"]
                expanding_seeds = (
                    config["formal"]["seeds"]
                    if strategy_plan["run_expanding"]
                    else []
                )
                for seed in expanding_seeds:
                    split_name = "expanding_rolling_backtest"
                    try:
                        checkpoint = (
                            checkpoint_root
                            / f"horizon_{horizon}"
                            / f"lookback_{lookback}"
                            / split_name
                            / f"seed_{seed}_final.joblib"
                        )
                        predictions, params, audits, selected_columns = run_expanding_setting(
                            samples,
                            expanding_split.test_idx,
                            config,
                            expanding_params,
                            seed,
                            checkpoint,
                        )
                        metrics, enriched = evaluate_predictions(
                            predictions,
                            horizon,
                            float(predictions["mase_scale"].mean()),
                            config["evaluation"]["transaction_cost_bps"],
                        )
                        enriched = _decorate(
                            enriched,
                            model_name,
                            config["feature_set"]["name"],
                            horizon,
                            lookback,
                            split_name,
                            seed,
                        )
                        metric_row = {
                            "model": model_name,
                            "feature_set": config["feature_set"]["name"],
                            "horizon": horizon,
                            "lookback": lookback,
                            "split_strategy": split_name,
                            "seed": seed,
                            "test_start": enriched["anchor_date"].min(),
                            "test_end": enriched["anchor_date"].max(),
                            "n_predictions": len(enriched),
                            "params": json.dumps(params, sort_keys=True),
                            **metrics,
                        }
                        all_predictions.append(enriched)
                        metric_rows.append(metric_row)
                        validation_start = samples.metadata.iloc[
                            expanding_split.validation_idx[0]
                        ]["anchor_date"]
                        validation_end = samples.metadata.iloc[
                            expanding_split.validation_idx[-1]
                        ]["anchor_date"]
                        group_audits: list[dict[str, Any]] = []
                        for audit in audits:
                            complete_audit = {
                                **audit,
                                "validation_start_date": validation_start,
                                "validation_end_date": validation_end,
                            }
                            group_audits.append(complete_audit)
                            audit_rows.append(
                                {
                                    "horizon": horizon,
                                    "lookback": lookback,
                                    "split_strategy": split_name,
                                    "seed": seed,
                                    **complete_audit,
                                }
                            )
                        _save_group(
                            result_root,
                            enriched,
                            metric_row,
                            horizon,
                            lookback,
                            split_name,
                            model_name,
                            seed,
                            group_audits,
                            selected_columns,
                            config,
                        )
                    except Exception as error:
                        raise_formal_failure(
                            result_root,
                            failures,
                            horizon,
                            lookback,
                            split_name,
                            seed,
                            error,
                        )

        predictions_frame = pd.concat(all_predictions, ignore_index=True)
        metrics_frame = pd.DataFrame(metric_rows)
        failures_frame = pd.DataFrame(
            failures,
            columns=["horizon", "lookback", "split_strategy", "seed", "error"],
        )
        coverage = validate_formal_coverage(
            metrics_frame, predictions_frame, config, expected_counts
        )
        seed_stability = build_seed_stability(metrics_frame)
        predictions_frame.to_csv(result_root / "predictions.csv", index=False)
        predictions_frame.to_csv(result_root / "all_predictions.csv", index=False)
        metrics_frame.to_csv(result_root / "metrics.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(result_root / "tuning_results.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(result_root / "fold_audit.csv", index=False)
        failures_frame.to_csv(result_root / "model_failures.csv", index=False)
        coverage.to_csv(result_root / "coverage.csv", index=False)
        seed_stability.to_csv(result_root / "seed_stability.csv", index=False)
        for strategy in strategy_plan["configured"]:
            metrics_frame[metrics_frame["split_strategy"] == strategy].to_csv(
                result_root / f"test_results_{strategy}.csv", index=False
            )
        comparison = metrics_frame.groupby(
            ["model", "feature_set", "horizon", "lookback", "split_strategy"],
            as_index=False,
        ).mean(numeric_only=True)
        comparison.to_csv(result_root / "split_strategy_comparison.csv", index=False)
        _write_report(
            report_root / "report.md",
            run_id,
            data_audit,
            metrics_frame,
            failures_frame,
            config,
            seed_stability,
        )
        manifest.update(
            {
                "status": "COMPLETED",
                "finished_at": datetime.now(),
                "n_predictions": len(predictions_frame),
                "n_metric_rows": len(metrics_frame),
                "n_failures": len(failures_frame),
                "results_path": result_root.relative_to(project_root).as_posix(),
                "checkpoints_path": checkpoint_root.relative_to(project_root).as_posix(),
                "report_path": report_root.relative_to(project_root).as_posix(),
            }
        )
        _write_json(result_root / "experiment_manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update({"status": "FAILED", "finished_at": datetime.now(), "error": repr(error)})
        _write_json(result_root / "experiment_manifest.json", manifest)
        (result_root / "FAILED").write_text(repr(error), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Daily Corn price backtests")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "random_forest.yaml"
    )
    parser.add_argument("--mode", choices=["preflight", "formal"], default="preflight")
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--confirm-formal-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(_json_safe(run_preflight(config)), ensure_ascii=False, indent=2))
        return
    if not args.confirm_formal_run:
        raise SystemExit("Formal run blocked: pass --confirm-formal-run only after user approval")
    print(
        json.dumps(
            _json_safe(run_formal(config, source_commit=args.source_commit)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
