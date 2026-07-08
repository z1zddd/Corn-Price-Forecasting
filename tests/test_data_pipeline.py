import numpy as np
import pandas as pd

from data.loader import load_commodity_csv, select_feature_columns
from data.targets import add_forward_targets
from data.windowing import make_windows


def test_add_forward_targets_uses_price_col():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
            "close": [100.0, 105.0, 102.0, 110.0],
            "spike": [0, 0, 0, 0],
        }
    )

    out = add_forward_targets(df, price_col="close", horizon=1, spike_threshold=0.0)

    assert out["target_price_fwd"].tolist() == [105.0, 102.0, 110.0]
    assert np.round(out["target_return_fwd"].to_numpy(), 6).tolist() == [0.05, -0.028571, 0.078431]
    assert out["target_direction_fwd"].tolist() == [1, 0, 1]


def test_auto_numeric_excludes_targets_and_date():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "close": [100.0, 101.0, 102.0],
            "volume": [10.0, 11.0, 12.0],
            "target_return_fwd": [0.1, 0.2, 0.3],
            "name": ["a", "b", "c"],
        }
    )

    cols = select_feature_columns(df, "auto_numeric", date_col="date", exclude_feature_cols=[])

    assert cols == ["close", "volume"]


def test_auto_numeric_exclude_patterns_remove_pca_news():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
            "close": [100.0, 101.0, 102.0],
            "basis": [0.1, 0.2, 0.3],
            "pca_001": [1.0, 0.0, 1.0],
            "PCA002": [0.0, 1.0, 0.0],
        }
    )

    cols = select_feature_columns(
        df,
        "auto_numeric",
        date_col="date",
        exclude_feature_cols=[],
        exclude_feature_patterns=["pca_*", "PCA*"],
    )

    assert cols == ["close", "basis"]


def test_make_windows_shape_and_meta():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "target_direction_fwd": [1, 1, 1, 1, 1, 1],
        }
    )

    x, y, meta = make_windows(
        df,
        feature_cols=["close", "volume"],
        target_col="target_direction_fwd",
        date_col="date",
        lookback=3,
    )

    assert x.shape == (4, 2, 3)
    assert y.tolist() == [1, 1, 1, 1]
    assert meta["row_idx"].tolist() == [2, 3, 4, 5]


def test_load_commodity_csv_sorts_dates():
    df, encoding = load_commodity_csv("examples/corn/sample_data.csv", date_col="date", encodings=["utf-8"])

    assert encoding == "utf-8"
    assert df["date"].is_monotonic_increasing
    assert len(df) >= 24


def test_load_commodity_csv_uses_explicit_date_format(tmp_path):
    csv_path = tmp_path / "month_format.csv"
    csv_path.write_text("month,close\n16-Jun,100\n16-Jul,101\n", encoding="utf-8")

    df, encoding = load_commodity_csv(csv_path, date_col="month", date_format="%y-%b", encodings=["utf-8"])

    assert encoding == "utf-8"
    assert df["month"].dt.strftime("%Y-%m-%d").tolist() == ["2016-06-01", "2016-07-01"]
