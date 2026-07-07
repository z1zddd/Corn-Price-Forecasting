from pathlib import Path

import pandas as pd
import pytest
import yaml

from backtest.engine import run_backtest
from data.scaler import SequenceStandardizer


def test_sequence_standardizer_fits_train_only_statistics():
    x_train = [[[1.0, 3.0]], [[5.0, 7.0]]]
    x_test = [[[100.0, 102.0]]]
    scaler = SequenceStandardizer().fit(x_train)

    transformed = scaler.transform_x(x_test)

    # Train-only mean is 4 and std is sqrt(5). The future test values must not
    # affect those statistics.
    assert scaler.x_mean.tolist() == [4.0]
    assert round(float(scaler.x_std[0]), 6) == 2.236068
    assert float(transformed[0, 0, 0]) == pytest.approx((100.0 - 4.0) / (5.0**0.5))


def test_backtest_predictions_include_temporal_train_val_test_boundaries(tmp_path: Path):
    cfg = yaml.safe_load(Path("configs/corn.yaml").read_text(encoding="utf-8"))
    cfg["models"] = ["last_return"]
    cfg["split"] = {"val_ratio": 0.25}
    output_dir = tmp_path / "experiment"

    run_backtest(cfg, output_dir=output_dir)

    predictions = pd.read_csv(output_dir / "model_outputs" / "last_return" / "predictions.csv")
    required = {
        "train_start_date",
        "train_end_date",
        "val_start_date",
        "val_end_date",
        "test_date",
    }
    assert required.issubset(predictions.columns)

    first = predictions.iloc[0]
    assert pd.to_datetime(first["train_end_date"]) < pd.to_datetime(first["val_start_date"])
    assert pd.to_datetime(first["val_end_date"]) < pd.to_datetime(first["test_date"])


def test_backtest_horizon_two_uses_only_known_targets(tmp_path: Path):
    cfg = yaml.safe_load(Path("configs/corn.yaml").read_text(encoding="utf-8"))
    cfg["models"] = ["last_return"]
    cfg["target"]["horizon"] = 2
    cfg["lookback"]["default"] = 6
    cfg["lookback"]["candidates"] = [6]
    cfg["train_window"]["min_train_periods"] = 24
    cfg["train_window"]["target_known_only"] = True
    cfg["split"] = {"val_ratio": 0.0}
    output_dir = tmp_path / "horizon_two"

    run_backtest(cfg, output_dir=output_dir)

    predictions = pd.read_csv(output_dir / "model_outputs" / "last_return" / "predictions.csv")
    assert not predictions.empty
    assert "train_max_target_date" in predictions.columns
    assert "test_target_date" in predictions.columns
    train_target_max = pd.to_datetime(predictions["train_max_target_date"])
    test_anchor = pd.to_datetime(predictions["test_date"])
    assert bool((train_target_max <= test_anchor).all())
