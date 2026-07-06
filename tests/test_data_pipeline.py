import numpy as np
import pandas as pd

from commodity_backtest.data.loader import load_commodity_csv, select_feature_columns
from commodity_backtest.data.targets import add_forward_targets
from commodity_backtest.data.windowing import make_windows


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