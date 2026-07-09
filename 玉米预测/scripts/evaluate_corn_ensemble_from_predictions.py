#!/usr/bin/env python3
"""Evaluate small no-leakage ensembles from completed rolling predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


@dataclass(frozen=True)
class Candidate:
    model: str
    feature_set: str
    lookback_months: int
    horizon_months: int
    head: str

    @property
    def cid(self) -> str:
        news = "news" if self.feature_set == "with_news_precomputed_pca" else "nonews"
        return f"{self.model}|{news}|lb{self.lookback_months}|h{self.horizon_months}|{self.head}"


H1_CANDIDATES = [
    Candidate("mlp_small_relu", "with_news_precomputed_pca", 6, 1, "cls"),
    Candidate("sgd_modified_huber", "with_news_precomputed_pca", 6, 1, "cls"),
    Candidate("lightgbm_dart", "with_news_precomputed_pca", 12, 1, "reg"),
    Candidate("keras_lstm_u16", "no_news", 12, 1, "cls"),
    Candidate("keras_tcn_filters16_k2_d1", "no_news", 9, 1, "reg"),
    Candidate("keras_gru_u16", "with_news_precomputed_pca", 12, 1, "reg"),
]

H2_CANDIDATES = [
    Candidate("aeon_knn_euclidean", "with_news_precomputed_pca", 6, 2, "reg"),
    Candidate("knn_5_distance", "with_news_precomputed_pca", 6, 2, "reg"),
    Candidate("aeon_deep_timecnn", "with_news_precomputed_pca", 12, 2, "cls"),
    Candidate("aeon_deep_timecnn", "no_news", 12, 2, "cls"),
    Candidate("knn_3_uniform", "with_news_precomputed_pca", 12, 2, "cls"),
    Candidate("aeon_rise", "no_news", 6, 2, "cls"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Result root containing live/all_rolling_predictions.csv")
    parser.add_argument("--out-dir", default="", help="Defaults to <root>/live")
    parser.add_argument("--rolling-window", type=int, default=24)
    parser.add_argument("--min-history", type=int, default=12)
    return parser.parse_args()


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def safe_ba(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, pred))


def summarize_predictions(name: str, horizon: int | str, variant: str, frame: pd.DataFrame) -> dict[str, object]:
    out = {
        "ensemble": name,
        "eval_target": horizon,
        "variant": variant,
        "n_predictions": int(len(frame)),
        "coverage": float(frame["predicted_direction"].notna().mean()) if len(frame) else float("nan"),
    }
    scored = frame.dropna(subset=["predicted_direction"]).copy()
    if scored.empty:
        out.update(
            {
                "auc": float("nan"),
                "average_precision": float("nan"),
                "balanced_accuracy": float("nan"),
                "accuracy": float("nan"),
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                "predicted_positive_rate": float("nan"),
                "actual_positive_rate": float("nan"),
                "mean_score": float("nan"),
            }
        )
        return out
    y = scored["actual_direction"].to_numpy(int)
    pred = scored["predicted_direction"].to_numpy(int)
    score = scored["ensemble_score"].to_numpy(float)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out.update(
        {
            "auc": safe_auc(y, score),
            "average_precision": safe_ap(y, score),
            "balanced_accuracy": safe_ba(y, pred),
            "accuracy": float(accuracy_score(y, pred)),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "predicted_positive_rate": float(np.mean(pred == 1)),
            "actual_positive_rate": float(np.mean(y == 1)),
            "mean_score": float(np.mean(score)),
        }
    )
    return out


def threshold_from_past(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 0.5
    best = (float("-inf"), 0.5)
    for thr in np.linspace(0.35, 0.65, 31):
        pred = (score >= thr).astype(int)
        ba = safe_ba(y, pred)
        if np.isnan(ba):
            continue
        candidate = (ba, -abs(thr - 0.5))
        current = (best[0], -abs(best[1] - 0.5))
        if candidate > current:
            best = (ba, float(thr))
    return best[1]


def rolling_thresholds(y: np.ndarray, score: np.ndarray, window: int, min_history: int) -> np.ndarray:
    thresholds = np.full(len(y), 0.5, dtype=float)
    for idx in range(len(y)):
        start = max(0, idx - window)
        if idx - start < min_history:
            continue
        thresholds[idx] = threshold_from_past(y[start:idx], score[start:idx])
    return thresholds


def capped_weights(raw: np.ndarray, cap: float = 0.35) -> np.ndarray:
    if raw.sum() <= 0:
        return np.ones_like(raw) / len(raw)
    weights = raw / raw.sum()
    for _ in range(10):
        over = weights > cap
        if not over.any():
            break
        excess = weights[over].sum() - cap * over.sum()
        weights[over] = cap
        under = ~over
        if under.any() and weights[under].sum() > 0:
            weights[under] += excess * weights[under] / weights[under].sum()
        else:
            break
    return weights / weights.sum()


def rolling_weighted_score(base: pd.DataFrame, candidate_ids: list[str], window: int, min_history: int) -> tuple[np.ndarray, list[str]]:
    y = base["actual_direction"].to_numpy(int)
    prob_cols = [f"prob::{cid}" for cid in candidate_ids]
    pred_cols = [f"pred::{cid}" for cid in candidate_ids]
    scores = np.zeros(len(base), dtype=float)
    weight_text: list[str] = []
    for idx in range(len(base)):
        start = max(0, idx - window)
        if idx - start < min_history:
            weights = np.ones(len(candidate_ids), dtype=float) / len(candidate_ids)
        else:
            raw = []
            for col in pred_cols:
                ba = safe_ba(y[start:idx], base[col].to_numpy(int)[start:idx])
                raw.append(max(0.0, 0.0 if np.isnan(ba) else ba - 0.5))
            weights = capped_weights(np.asarray(raw, dtype=float))
        scores[idx] = float(np.dot(base.loc[base.index[idx], prob_cols].to_numpy(float), weights))
        weight_text.append(";".join(f"{candidate_ids[i]}={weights[i]:.4f}" for i in range(len(candidate_ids))))
    return scores, weight_text


def build_base(predictions: pd.DataFrame, name: str, candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    for cand in candidates:
        mask = (
            predictions["model"].eq(cand.model)
            & predictions["feature_set"].eq(cand.feature_set)
            & predictions["lookback_months"].eq(cand.lookback_months)
            & predictions["horizon_months"].eq(cand.horizon_months)
            & predictions["head"].eq(cand.head)
        )
        part = predictions.loc[mask].copy()
        if part.empty:
            raise RuntimeError(f"Missing candidate predictions: {cand}")
        part["candidate_id"] = cand.cid
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    # Origin ids are local to each lookback/horizon sample set, so different
    # lookbacks can use different ids for the same calendar test month. Align
    # ensemble members by calendar anchor/target month instead.
    index_cols = ["anchor_month", "target_month", "horizon_months", "actual_direction"]
    probs = df.pivot_table(index=index_cols, columns="candidate_id", values="predicted_probability", aggfunc="first")
    preds = df.pivot_table(index=index_cols, columns="candidate_id", values="predicted_direction", aggfunc="first")
    probs.columns = [f"prob::{c}" for c in probs.columns]
    preds.columns = [f"pred::{c}" for c in preds.columns]
    base = pd.concat([probs, preds], axis=1).reset_index()
    expected = len(candidates)
    present = base[[f"prob::{c.cid}" for c in candidates]].notna().sum(axis=1)
    incomplete = int((~present.eq(expected)).sum())
    if incomplete:
        print(f"[warn] {name}: dropping {incomplete} early/incomplete calendar rows")
        base = base.loc[present.eq(expected)].copy()
    base["anchor_month"] = pd.to_datetime(base["anchor_month"])
    base["target_month"] = pd.to_datetime(base["target_month"])
    base = base.sort_values(["anchor_month", "target_month"]).reset_index(drop=True)
    base["origin_id"] = np.arange(len(base), dtype=int)
    return base


def horizon_ensembles(name: str, base: pd.DataFrame, candidates: list[Candidate], window: int, min_history: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_ids = [c.cid for c in candidates]
    prob_cols = [f"prob::{cid}" for cid in candidate_ids]
    pred_cols = [f"pred::{cid}" for cid in candidate_ids]
    y = base["actual_direction"].to_numpy(int)
    uniform_score = base[prob_cols].mean(axis=1).to_numpy(float)
    vote_score = base[pred_cols].mean(axis=1).to_numpy(float)
    weighted_score, weight_text = rolling_weighted_score(base, candidate_ids, window, min_history)
    uniform_thr = rolling_thresholds(y, uniform_score, window, min_history)
    weighted_thr = rolling_thresholds(y, weighted_score, window, min_history)

    detail_rows = []
    summary_rows = []

    variants = [
        ("hard_vote_strict", vote_score, np.where(vote_score > 0.5, 1, 0), np.full(len(base), 0.5)),
        ("soft_uniform_05", uniform_score, np.where(uniform_score >= 0.5, 1, 0), np.full(len(base), 0.5)),
        ("soft_uniform_rollthr", uniform_score, np.where(uniform_score >= uniform_thr, 1, 0), uniform_thr),
        ("soft_weighted_05", weighted_score, np.where(weighted_score >= 0.5, 1, 0), np.full(len(base), 0.5)),
        ("soft_weighted_rollthr", weighted_score, np.where(weighted_score >= weighted_thr, 1, 0), weighted_thr),
    ]
    for variant, score, pred, thr in variants:
        frame = base[["anchor_month", "target_month", "origin_id", "horizon_months", "actual_direction"]].copy()
        frame["ensemble"] = name
        frame["variant"] = variant
        frame["ensemble_score"] = score
        frame["threshold"] = thr
        frame["predicted_direction"] = pred
        frame["weights"] = weight_text if variant.startswith("soft_weighted") else ""
        detail_rows.append(frame)
        summary_rows.append(summarize_predictions(name, int(base["horizon_months"].iloc[0]), variant, frame))

    for variant, score in [("soft_uniform_band_55_45", uniform_score), ("soft_weighted_band_55_45", weighted_score)]:
        pred = np.full(len(base), np.nan)
        pred[score >= 0.55] = 1
        pred[score <= 0.45] = 0
        frame = base[["anchor_month", "target_month", "origin_id", "horizon_months", "actual_direction"]].copy()
        frame["ensemble"] = name
        frame["variant"] = variant
        frame["ensemble_score"] = score
        frame["threshold"] = np.nan
        frame["predicted_direction"] = pred
        frame["weights"] = weight_text if "weighted" in variant else ""
        detail_rows.append(frame)
        summary_rows.append(summarize_predictions(name, int(base["horizon_months"].iloc[0]), variant, frame))

    return pd.concat(detail_rows, ignore_index=True), pd.DataFrame(summary_rows)


def confirmation_ensembles(h1: pd.DataFrame, h2: pd.DataFrame, window: int, min_history: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    h1_candidates = [c.cid for c in H1_CANDIDATES]
    h2_candidates = [c.cid for c in H2_CANDIDATES]
    h1_score = h1[[f"prob::{c}" for c in h1_candidates]].mean(axis=1).to_numpy(float)
    h2_score = h2[[f"prob::{c}" for c in h2_candidates]].mean(axis=1).to_numpy(float)
    h1_weighted, _ = rolling_weighted_score(h1, h1_candidates, window, min_history)
    h2_weighted, _ = rolling_weighted_score(h2, h2_candidates, window, min_history)
    left = h1[["anchor_month", "target_month", "origin_id", "actual_direction"]].copy()
    left = left.rename(columns={"target_month": "h1_target_month", "actual_direction": "actual_direction_h1"})
    left["h1_score"] = h1_score
    left["h1_weighted_score"] = h1_weighted
    right = h2[["anchor_month", "target_month", "origin_id", "actual_direction"]].copy()
    right = right.rename(columns={"target_month": "h2_target_month", "actual_direction": "actual_direction_h2"})
    right["h2_score"] = h2_score
    right["h2_weighted_score"] = h2_weighted
    merged = left.merge(right, on="anchor_month", suffixes=("_h1", "_h2"))

    detail_rows = []
    summary_rows = []
    for variant, s1, s2 in [
        ("confirm_uniform_55_45", "h1_score", "h2_score"),
        ("confirm_weighted_55_45", "h1_weighted_score", "h2_weighted_score"),
    ]:
        signal = np.full(len(merged), np.nan)
        up = (merged[s1] >= 0.55) & (merged[s2] >= 0.55)
        down = (merged[s1] <= 0.45) & (merged[s2] <= 0.45)
        signal[up.to_numpy()] = 1
        signal[down.to_numpy()] = 0
        for target in ["h1", "h2"]:
            frame = pd.DataFrame(
                {
                    "anchor_month": merged["anchor_month"],
                    "target_month": merged[f"{target}_target_month"],
                    "origin_id": merged[f"origin_id_{target}"],
                    "horizon_months": 1 if target == "h1" else 2,
                    "actual_direction": merged[f"actual_direction_{target}"],
                    "ensemble": "h1_h2_same_anchor_confirmation",
                    "variant": variant,
                    "ensemble_score": (merged[s1] + merged[s2]) / 2,
                    "threshold": np.nan,
                    "predicted_direction": signal,
                    "weights": "",
                }
            )
            detail_rows.append(frame)
            summary_rows.append(
                summarize_predictions("h1_h2_same_anchor_confirmation", f"{target}_target", variant, frame)
            )
    return pd.concat(detail_rows, ignore_index=True), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(root / "live" / "all_rolling_predictions.csv")

    h1_base = build_base(predictions, "top6_h1", H1_CANDIDATES)
    h2_base = build_base(predictions, "top6_h2", H2_CANDIDATES)
    h1_detail, h1_summary = horizon_ensembles("top6_h1", h1_base, H1_CANDIDATES, args.rolling_window, args.min_history)
    h2_detail, h2_summary = horizon_ensembles("top6_h2", h2_base, H2_CANDIDATES, args.rolling_window, args.min_history)
    confirm_detail, confirm_summary = confirmation_ensembles(h1_base, h2_base, args.rolling_window, args.min_history)

    detail = pd.concat([h1_detail, h2_detail, confirm_detail], ignore_index=True)
    summary = pd.concat([h1_summary, h2_summary, confirm_summary], ignore_index=True)
    summary = summary.sort_values(["balanced_accuracy", "coverage", "average_precision", "auc"], ascending=False)
    detail.to_csv(out_dir / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "ensemble_summary_metrics.csv", index=False, encoding="utf-8-sig")

    print("ENSEMBLE_SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {out_dir / 'ensemble_summary_metrics.csv'}")
    print(f"Saved: {out_dir / 'ensemble_predictions.csv'}")


if __name__ == "__main__":
    main()
