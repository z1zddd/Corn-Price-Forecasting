from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.config_loader import load_config
from src.data.loader import load_and_window, resolve_input_path
from src.data.pipeline import DataPipeline
from src.eval.metrics import apply_platt, evaluate_classification, evaluate_model, fit_positive_platt, sigmoid_np
from src.eval.reporter import generate_classification_report, generate_report
from src.models.classical.baseline import LastReturnBaseline, MeanReturnBaseline, MovingAverageReturnBaseline, ZeroReturnBaseline
from src.models.classical.random_forest import RandomForestModel
from src.models.classical.sklearn_models import ExtraTreesModel, LinearSVCModel, LogisticRegressionModel
from src.models.deep.cnn import FCNClassifier, InceptionTimeClassifier, ResNet1DClassifier, TCNClassifier
from src.models.deep.dual_stream_lstm import DualStreamLSTMClassifier
from src.models.deep.itransformer import ITransformerClassifier
from src.models.deep.structured_lstm import StructuredLSTMClassifier
from src.models.deep.timexer import TimeXerClassifier


def test_data_pipeline_contract() -> None:
    cfg = load_config("data")
    cfg["smoke_train_tail_n"] = 80
    cfg["smoke_val_n"] = 20
    cfg["smoke_test_n"] = 20
    bundle = DataPipeline(cfg).run()
    assert bundle.X_train.ndim == 3
    assert bundle.X_train.shape[1] == len(bundle.feature_cols)
    assert bundle.X_train.shape[2] == cfg["seq_len"]
    assert bundle.y_train.ndim == 1
    assert np.isfinite(bundle.X_train).all()
    assert np.isfinite(bundle.X_val).all()
    assert np.isfinite(bundle.X_test).all()
    assert (pd.to_datetime(bundle.meta_test["target_date"]) > pd.to_datetime(bundle.meta_test["date"])).all()
    assert pd.to_datetime(bundle.meta_train["target_date"]).max() < pd.to_datetime(bundle.meta_val["target_date"]).min()
    assert pd.to_datetime(bundle.meta_val["target_date"]).max() < pd.to_datetime(bundle.meta_test["target_date"]).min()


def test_return_target_alignment() -> None:
    cfg = load_config("data")
    x, y, meta = load_and_window(cfg)
    assert x.shape[0] == len(y) == len(meta)
    raw = pd.read_csv(resolve_input_path(cfg["csv_path"]))
    close = raw[cfg.get("price_col", "dce_corn_close")].to_numpy(dtype=float)
    sample_ids = [0, 1, 2, len(meta) // 2, len(meta) - 1]
    for sample_id in sample_ids:
        row = meta.iloc[sample_id]
        anchor_idx = int(row["anchor_idx"])
        target_idx = int(row["target_idx"])
        expected = close[target_idx] / close[anchor_idx] - 1.0
        assert np.isclose(y[sample_id], expected, rtol=1e-6, atol=1e-6)


def test_monthly_price_target_alignment() -> None:
    cfg = load_config("data_monthly_github")
    x, y, meta = load_and_window(cfg)
    assert x.shape[0] == len(y) == len(meta)
    assert x.shape == (109, 91, 12)
    raw = pd.read_csv(resolve_input_path(cfg["csv_path"]))
    close = raw[cfg.get("price_col", "dce_corn_close")].to_numpy(dtype=float)
    sample_ids = [0, 1, 2, len(meta) // 2, len(meta) - 1]
    for sample_id in sample_ids:
        target_idx = int(meta.iloc[sample_id]["target_idx"])
        assert np.isclose(y[sample_id], close[target_idx], rtol=1e-6, atol=1e-6)


def test_dual_stream_spike_target_alignment() -> None:
    cfg = load_config("data_dual_stream_spike")
    x, y, meta = load_and_window(cfg)
    assert x.shape == (109, 79, 12)
    features = meta.attrs["feature_cols"]
    assert len([c for c in features if c.startswith("pca_")]) == 22
    assert len([c for c in features if not c.startswith("pca_")]) == 57
    raw = pd.read_csv(resolve_input_path(cfg["csv_path"]))
    raw[cfg["date_col"]] = pd.to_datetime(raw[cfg["date_col"]].astype(str), format=cfg["date_format"])
    raw = raw.sort_values(cfg["date_col"]).reset_index(drop=True)
    labels = raw[cfg["target_col"]].to_numpy(dtype=float)
    sample_ids = [0, 1, len(meta) // 2, len(meta) - 1]
    for sample_id in sample_ids:
        target_idx = int(meta.iloc[sample_id]["target_idx"])
        assert np.isclose(y[sample_id], labels[target_idx], rtol=1e-6, atol=1e-6)


def test_structured_spike_ablation_feature_selection() -> None:
    cfg = load_config("data_structured_spike")
    x, y, meta = load_and_window(cfg)
    assert x.shape == (109, 57, 12)
    features = meta.attrs["feature_cols"]
    assert len([c for c in features if c.startswith("pca_")]) == 0
    assert len(features) == 57
    assert x.shape[0] == len(y) == len(meta)


def test_spike_24_lookback_ablation_feature_selection() -> None:
    dual_cfg = load_config("data_dual_stream_spike_l24")
    x_dual, y_dual, meta_dual = load_and_window(dual_cfg)
    assert x_dual.shape == (97, 79, 24)
    dual_features = meta_dual.attrs["feature_cols"]
    assert len([c for c in dual_features if c.startswith("pca_")]) == 22
    assert len(dual_features) == 79
    assert x_dual.shape[0] == len(y_dual) == len(meta_dual)

    structured_cfg = load_config("data_structured_spike_l24")
    x_structured, y_structured, meta_structured = load_and_window(structured_cfg)
    assert x_structured.shape == (97, 57, 24)
    structured_features = meta_structured.attrs["feature_cols"]
    assert len([c for c in structured_features if c.startswith("pca_")]) == 0
    assert len(structured_features) == 57
    assert x_structured.shape[0] == len(y_structured) == len(meta_structured)


def test_all_numeric_spike_feature_selection() -> None:
    cfg = load_config("data_all_numeric_spike")
    x, y, meta = load_and_window(cfg)
    assert x.shape == (109, 122, 12)
    features = meta.attrs["feature_cols"]
    assert len(features) == 122
    assert "spike" not in features
    assert "dce_corn_close_next_month" not in features
    assert "dce_corn_close_next_month_ret" not in features
    assert len([c for c in features if c.startswith("pca_")]) == 32
    assert x.shape[0] == len(y) == len(meta)


def test_timexer_all_numeric_feature_selection() -> None:
    cfg = load_config("data_timexer_all_numeric_spike")
    x, y, meta = load_and_window(cfg)
    assert x.shape == (109, 122, 12)
    features = meta.attrs["feature_cols"]
    assert len(features) == 122
    assert features[-1] == "dce_corn_close"
    assert "spike" not in features
    assert "dce_corn_close_next_month" not in features
    assert "dce_corn_close_next_month_ret" not in features
    assert len([c for c in features if c.startswith("pca_")]) == 32
    assert x.shape[0] == len(y) == len(meta)


def test_timexer_platt_config_matches_feature_selection() -> None:
    cfg = load_config("data_timexer_all_numeric_spike_platt")
    x, y, meta = load_and_window(cfg)
    assert cfg["classification_evaluation"] == "validation_platt"
    assert x.shape == (109, 122, 12)
    features = meta.attrs["feature_cols"]
    assert features[-1] == "dce_corn_close"
    assert len([c for c in features if c.startswith("pca_")]) == 32
    assert x.shape[0] == len(y) == len(meta)


def test_timepfn_official_platt_config_matches_feature_selection() -> None:
    cfg = load_config("data_timepfn_official_spike_platt")
    x, y, meta = load_and_window(cfg)
    assert cfg["classification_evaluation"] == "validation_platt"
    assert cfg["scale_x"] is False
    assert x.shape == (109, 122, 12)
    features = meta.attrs["feature_cols"]
    assert features[-1] == "dce_corn_close"
    assert "spike" not in features
    assert "dce_corn_close_next_month" not in features
    assert "dce_corn_close_next_month_ret" not in features
    assert len([c for c in features if c.startswith("pca_")]) == 32
    assert x.shape[0] == len(y) == len(meta)


def test_ml_and_cnn_platt_configs_match_feature_selection() -> None:
    for config_name in ["data_ml_all_numeric_spike_platt", "data_cnn_all_numeric_spike_platt"]:
        cfg = load_config(config_name)
        x, y, meta = load_and_window(cfg)
        assert cfg["classification_evaluation"] == "validation_platt"
        assert cfg["scale_x"] is True
        assert x.shape == (109, 122, 12)
        features = meta.attrs["feature_cols"]
        assert features[-1] == "dce_corn_close"
        assert "spike" not in features
        assert "dce_corn_close_next_month" not in features
        assert "dce_corn_close_next_month_ret" not in features
        assert len([c for c in features if c.startswith("pca_")]) == 32
        assert x.shape[0] == len(y) == len(meta)


def test_chronos2_official_platt_config_matches_feature_selection() -> None:
    cfg = load_config("data_chronos2_official_spike_platt")
    x, y, meta = load_and_window(cfg)
    assert cfg["classification_evaluation"] == "validation_platt"
    assert cfg["scale_x"] is False
    assert x.shape == (109, 122, 12)
    features = meta.attrs["feature_cols"]
    assert features[-1] == "dce_corn_close"
    assert "spike" not in features
    assert "dce_corn_close_next_month" not in features
    assert "dce_corn_close_next_month_ret" not in features
    assert len([c for c in features if c.startswith("pca_")]) == 32
    assert x.shape[0] == len(y) == len(meta)


def test_tirex_official_platt_config_matches_feature_selection() -> None:
    cfg = load_config("data_tirex_official_spike_platt")
    x, y, meta = load_and_window(cfg)
    assert cfg["classification_evaluation"] == "validation_platt"
    assert cfg["scale_x"] is False
    assert x.shape == (109, 1, 12)
    features = meta.attrs["feature_cols"]
    assert features == ["dce_corn_close"]
    assert x.shape[0] == len(y) == len(meta)


def test_timer_official_platt_config_matches_feature_selection() -> None:
    cfg = load_config("data_timer_official_spike_platt")
    x, y, meta = load_and_window(cfg)
    assert cfg["classification_evaluation"] == "validation_platt"
    assert cfg["scale_x"] is False
    assert x.shape == (109, 1, 12)
    features = meta.attrs["feature_cols"]
    assert features == ["dce_corn_close"]
    assert x.shape[0] == len(y) == len(meta)


def test_model_contracts() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 3, 8)).astype("float32")
    y = rng.normal(size=40).astype("float32")
    models = [
        ZeroReturnBaseline(scaled_value=0.0),
        MeanReturnBaseline(),
        LastReturnBaseline(close_mean=10.0, close_std=2.0, y_mean=0.0, y_std=1.0),
        MovingAverageReturnBaseline(close_mean=10.0, close_std=2.0, y_mean=0.0, y_std=1.0),
        RandomForestModel(n_estimators=3, max_depth=2, random_state=42, n_jobs=1),
    ]
    for model in models:
        model.fit(X[:30], y[:30], X[30:35], y[30:35])
        pred = model.predict(X[35:])
        assert pred.shape == (5,)
        assert np.isfinite(pred).all()

    y_cls = np.asarray([0, 1] * 20, dtype="float32")
    sklearn_classifiers = [
        LogisticRegressionModel(C=0.5, solver="liblinear", class_weight="balanced", max_iter=200, random_state=42),
        LinearSVCModel(C=0.2, dual="auto", class_weight="balanced", max_iter=5000, random_state=42),
        ExtraTreesModel(n_estimators=5, max_depth=2, min_samples_leaf=2, random_state=42, n_jobs=1),
    ]
    for model in sklearn_classifiers:
        model.fit(X[:30], y_cls[:30], X[30:35], y_cls[30:35])
        prob = model.predict_proba(X[35:])
        logits = model.predict_logits(X[35:])
        assert prob.shape == (5,)
        assert logits.shape == (5,)
        assert ((prob >= 0.0) & (prob <= 1.0)).all()
        assert np.isfinite(logits).all()

    feature_cols = ["price_a", "price_b", "pca_001", "pca_002"]
    clf = DualStreamLSTMClassifier(feature_cols=feature_cols, epochs=1, batch_size=4, hidden_dim=4, attn_dim=3, dense_dim=4, device="cpu")
    y_cls = np.asarray([0, 1] * 20, dtype="float32")
    X_cls = rng.normal(size=(40, 4, 5)).astype("float32")
    clf.fit(X_cls[:30], y_cls[:30], X_cls[30:35], y_cls[30:35])
    prob = clf.predict_proba(X_cls[35:])
    assert prob.shape == (5,)
    assert ((prob >= 0.0) & (prob <= 1.0)).all()

    structured_clf = StructuredLSTMClassifier(epochs=1, batch_size=4, hidden_dim=4, dense_dim=4, device="cpu")
    X_struct = rng.normal(size=(40, 3, 5)).astype("float32")
    structured_clf.fit(X_struct[:30], y_cls[:30], X_struct[30:35], y_cls[30:35])
    struct_prob = structured_clf.predict_proba(X_struct[35:])
    assert struct_prob.shape == (5,)
    assert ((struct_prob >= 0.0) & (struct_prob <= 1.0)).all()

    itransformer_clf = ITransformerClassifier(epochs=1, batch_size=4, hidden_size=8, num_layers=1, n_heads=2, device="cpu")
    X_itransformer = rng.normal(size=(40, 6, 5)).astype("float32")
    itransformer_clf.fit(X_itransformer[:30], y_cls[:30], X_itransformer[30:35], y_cls[30:35])
    itransformer_prob = itransformer_clf.predict_proba(X_itransformer[35:])
    assert itransformer_prob.shape == (5,)
    assert ((itransformer_prob >= 0.0) & (itransformer_prob <= 1.0)).all()

    timexer_clf = TimeXerClassifier(epochs=1, batch_size=4, d_model=8, d_ff=8, e_layers=1, n_heads=2, patch_len=1, device="cpu")
    X_timexer = rng.normal(size=(40, 6, 5)).astype("float32")
    timexer_clf.fit(X_timexer[:30], y_cls[:30], X_timexer[30:35], y_cls[30:35])
    timexer_prob = timexer_clf.predict_proba(X_timexer[35:])
    assert timexer_prob.shape == (5,)
    assert ((timexer_prob >= 0.0) & (timexer_prob <= 1.0)).all()

    for cls in [FCNClassifier, ResNet1DClassifier, InceptionTimeClassifier, TCNClassifier]:
        cnn = cls(epochs=1, batch_size=4, hidden_size=4, dropout=0.1, device="cpu")
        X_cnn = rng.normal(size=(40, 5, 6)).astype("float32")
        cnn.fit(X_cnn[:30], y_cls[:30], X_cnn[30:35], y_cls[30:35])
        prob = cnn.predict_proba(X_cnn[35:])
        logits = cnn.predict_logits(X_cnn[35:])
        assert prob.shape == (5,)
        assert logits.shape == (5,)
        assert ((prob >= 0.0) & (prob <= 1.0)).all()
        assert np.isfinite(logits).all()


def test_eval_contract() -> None:
    today = np.array([100.0, 100.0, 100.0, 100.0])
    y_true = np.array([110.0, 90.0, 105.0, 95.0])
    y_pred = np.array([108.0, 92.0, 99.0, 101.0])
    metrics = evaluate_model(y_true, y_pred, today)
    assert metrics["direction_accuracy"] == 0.5
    assert "profit_factor" in metrics
    with tempfile.TemporaryDirectory() as tmp:
        meta = pd.DataFrame(
            {
                "series_id": ["corn"] * 4,
                "date": pd.date_range("2024-01-01", periods=4),
                "target_date": pd.date_range("2024-02-01", periods=4),
                "horizon": [30] * 4,
            }
        )
        generate_report(y_true, y_pred, today, "dummy", tmp, meta=meta)
        pred_path = Path(tmp) / "predictions.csv"
        metrics_path = Path(tmp) / "metrics.json"
        assert pred_path.exists()
        assert metrics_path.exists()
        cols = pd.read_csv(pred_path, nrows=0).columns.tolist()
        expected = [
            "series_id",
            "today_date",
            "target_date",
            "horizon",
            "today_close",
            "y_true_return",
            "y_pred_return",
            "actual_price",
            "pred_price",
            "predicted_change",
            "actual_direction",
            "pred_direction",
            "actual_label",
            "predicted_label",
            "direction_correct",
            "actual_return",
            "strategy_return",
            "equity",
        ]
        assert cols == expected
        assert json.loads(metrics_path.read_text())["direction_accuracy"] == 0.5

        y_cls = np.array([0, 1, 0, 1])
        prob = np.array([0.1, 0.8, 0.7, 0.9])
        class_metrics = evaluate_classification(y_cls, prob, threshold=0.5)
        assert class_metrics["accuracy"] == 0.75
        generate_classification_report(y_cls, prob, "dummy_cls", tmp, meta=meta, threshold=0.5)
        class_cols = pd.read_csv(pred_path, nrows=0).columns.tolist()
        assert "probability" in class_cols

        logits = np.array([-2.0, -0.5, 0.5, 2.0], dtype="float32")
        calibrator = fit_positive_platt(logits, y_cls, l2=1e-3, max_iter=20)
        calibrated = apply_platt(logits, calibrator)
        assert calibrator["a"] > 0
        assert calibrated.shape == (4,)
        assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()
        assert np.allclose(sigmoid_np(logits), 1.0 / (1.0 + np.exp(-logits)))


if __name__ == "__main__":
    test_data_pipeline_contract()
    test_return_target_alignment()
    test_monthly_price_target_alignment()
    test_dual_stream_spike_target_alignment()
    test_structured_spike_ablation_feature_selection()
    test_spike_24_lookback_ablation_feature_selection()
    test_all_numeric_spike_feature_selection()
    test_timexer_all_numeric_feature_selection()
    test_timexer_platt_config_matches_feature_selection()
    test_ml_and_cnn_platt_configs_match_feature_selection()
    test_timepfn_official_platt_config_matches_feature_selection()
    test_chronos2_official_platt_config_matches_feature_selection()
    test_tirex_official_platt_config_matches_feature_selection()
    test_timer_official_platt_config_matches_feature_selection()
    test_model_contracts()
    test_eval_contract()
    print("contract tests passed")
