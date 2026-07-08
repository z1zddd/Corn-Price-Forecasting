"""Input conversion and probability helpers for official-pool estimators."""

from __future__ import annotations

import numpy as np

from corn_forecast.modeling.classical.sklearn import flatten_windows


def format_input(x: np.ndarray, input_kind: str) -> np.ndarray:
    """Convert framework windows [N, V, T] to the estimator's expected layout."""

    x = np.asarray(x)
    if input_kind == "tabular_flat":
        return flatten_windows(x)
    if input_kind == "keras_sequence":
        return np.ascontiguousarray(np.transpose(x, (0, 2, 1)).astype(np.float32))
    if input_kind == "aeon_collection":
        return np.ascontiguousarray(x.astype(np.float32))
    if input_kind == "aeon_collection_pad10":
        return pad_time_axis(np.ascontiguousarray(x.astype(np.float32)), min_timepoints=10)
    if input_kind == "aeon_collection_pad10_float64":
        return pad_time_axis(np.ascontiguousarray(x.astype(np.float64)), min_timepoints=10)
    raise ValueError(f"Unknown official_pool input_kind: {input_kind}")


def pad_time_axis(x: np.ndarray, *, min_timepoints: int) -> np.ndarray:
    if x.shape[-1] >= min_timepoints:
        return x
    pad_width = [(0, 0), (0, 0), (0, min_timepoints - x.shape[-1])]
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=0.0)


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=float)
        classes = list(getattr(model, "classes_", []))
        if proba.ndim == 2 and proba.shape[1] > 1:
            if 1 in classes:
                return proba[:, classes.index(1)]
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        return sigmoid(np.asarray(model.decision_function(x), dtype=float).reshape(-1))
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
