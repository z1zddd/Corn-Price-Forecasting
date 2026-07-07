import numpy as np
import pandas as pd
import pytest

from backtest.splits import assert_temporal_holdout, make_backtest_windows


def test_expanding_windows():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")

    windows = make_backtest_windows(dates, mode="expanding", min_train_periods=4, stride_periods=1)

    assert len(windows) == 4
    assert windows[0].train_idx.tolist() == [0, 1, 2, 3]
    assert windows[0].test_idx.tolist() == [4]
    assert windows[-1].train_idx.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert windows[-1].test_idx.tolist() == [7]


def test_rolling_windows():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")

    windows = make_backtest_windows(
        dates,
        mode="rolling",
        min_train_periods=4,
        stride_periods=1,
        window_size_periods=4,
    )

    assert windows[0].train_idx.tolist() == [0, 1, 2, 3]
    assert windows[1].train_idx.tolist() == [1, 2, 3, 4]
    assert windows[1].test_idx.tolist() == [5]


def test_expanding_with_cap_windows():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")

    windows = make_backtest_windows(
        dates,
        mode="expanding_with_cap",
        min_train_periods=3,
        stride_periods=1,
        max_train_periods=5,
    )

    assert windows[0].train_idx.tolist() == [0, 1, 2]
    assert windows[2].train_idx.tolist() == [0, 1, 2, 3, 4]
    assert windows[3].train_idx.tolist() == [1, 2, 3, 4, 5]


def test_temporal_holdout_rejects_overlap():
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")

    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_temporal_holdout(dates, np.array([0, 1, 2]), np.array([2]), np.array([3]))


def test_target_known_windows_drop_unavailable_horizon_labels():
    dates = pd.date_range("2020-01-01", periods=10, freq="MS")
    target_dates = dates + pd.DateOffset(months=2)

    windows = make_backtest_windows(
        dates,
        mode="expanding",
        min_train_periods=4,
        stride_periods=1,
        target_dates=target_dates,
        target_known_only=True,
    )

    assert windows[0].test_idx.tolist() == [5]
    assert windows[0].train_idx.tolist() == [0, 1, 2, 3]
    for window in windows:
        test_anchor = dates[window.test_idx[0]]
        train_target_max = target_dates[window.train_idx].max()
        assert train_target_max <= test_anchor


def test_temporal_holdout_rejects_unknown_train_targets():
    dates = pd.date_range("2020-01-01", periods=5, freq="MS")
    target_dates = dates + pd.DateOffset(months=2)

    with pytest.raises(ValueError, match="train target max"):
        assert_temporal_holdout(
            dates,
            np.array([0, 1, 2]),
            np.array([], dtype=int),
            np.array([3]),
            target_dates=target_dates,
        )
