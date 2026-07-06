from pathlib import Path

import numpy as np
import pandas as pd

from commodity_backtest.data.diagnosis import diagnose_frame
from commodity_backtest.data.loader import load_commodity_csv
from commodity_backtest.models.sklearn_models import SklearnClassifierAdapter


def test_gbk_csv_can_be_loaded(tmp_path: Path):
    path = tmp_path / "?? ??.csv"
    df = pd.DataFrame(
        {
            "??": pd.date_range("2020-01-01", periods=3, freq="MS").strftime("%Y-%m-%d"),
            "???": [100, 101, 102],
        }
    )
    df.to_csv(path, index=False, encoding="gbk")

    loaded, encoding = load_commodity_csv(path, date_col="??", encodings=["utf-8", "gbk"])

    assert encoding == "gbk"
    assert loaded["???"].tolist() == [100, 101, 102]


def test_diagnosis_reports_price_candidates():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "close": [100.0, 101.0, 99.0, 102.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )

    report = diagnose_frame(df)

    assert report["rows"] == 4
    assert "date" in report["date_candidates"]
    assert "close" in report["price_candidates"]
    assert report["status"] == "usable"


def test_diagnosis_distinguishes_warning_and_unusable_statuses():
    warning_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "close": [100.0, None, 102.0, 103.0, 104.0],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    unusable_df = pd.DataFrame({"note": ["a", "b", "c"]})

    warning_report = diagnose_frame(warning_df)
    unusable_report = diagnose_frame(unusable_df)

    assert warning_report["status"] == "usable_with_warnings"
    assert unusable_report["status"] == "unusable"


class _MalformedOneColumnEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        return np.full((len(x), 1), 0.7)


def test_sklearn_adapter_handles_malformed_one_column_probability_output():
    adapter = SklearnClassifierAdapter(_MalformedOneColumnEstimator())
    x = np.ones((3, 2, 2))

    proba = adapter.predict_proba(x)

    assert proba.shape == (3,)
    assert proba.tolist() == [0.7, 0.7, 0.7]