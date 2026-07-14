from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_corn_daily_factors.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_corn_daily_factors", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_raw(rows: int = 80) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 + index
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-02", periods=rows).strftime("%Y/%m/%d"),
            "dce_corn_high": close + 2.0,
            "dce_corn_low": close - 2.0,
            "dce_corn_close": close,
            "dce_corn_volume": 1000.0 + index * 10.0,
            "dce_corn_open_interest": 2000.0 + index * 5.0,
            "cs_c_spread_close": 400.0 + index,
            "corn_basis_rate": 0.10 + index / 1000.0,
            "corn_100ppi_main_futures_price_cny_t": 100.0 + index,
            "corn_100ppi_nearby_futures_price_cny_t": 102.0 + index * 1.1,
            "cbot_corn_close": 300.0 + index * 2.0,
            "cbot_wheat_corn_ratio": 1.20 + index / 100.0,
            "ne_avg_precip_mm": 1.0 + index / 10.0,
            "ne_avg_t2m_c": 15.0 + index / 20.0,
        }
    )


class DailyFactorBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def test_lagged_sources_use_only_prior_dce_row(self) -> None:
        raw = synthetic_raw()
        factors = self.builder.build_factors(raw)

        self.assertAlmostEqual(factors.loc[10, "basis_rate_level_lag1d"], raw.loc[9, "corn_basis_rate"])
        expected_term = (
            raw.loc[9, "corn_100ppi_nearby_futures_price_cny_t"]
            / raw.loc[9, "corn_100ppi_main_futures_price_cny_t"]
            - 1.0
        )
        self.assertAlmostEqual(factors.loc[10, "nearby_main_spread_ratio_lag1d"], expected_term)
        expected_cbot = raw.loc[9, "cbot_corn_close"] / raw.loc[8, "cbot_corn_close"] - 1.0
        self.assertAlmostEqual(factors.loc[10, "cbot_corn_momentum_1d_lag1d"], expected_cbot)

    def test_rolling_windows_preserve_warmup_gaps(self) -> None:
        factors = self.builder.build_factors(synthetic_raw())

        self.assertTrue(factors["price_ma_gap_60d"].iloc[:59].isna().all())
        self.assertTrue(pd.notna(factors.loc[59, "price_ma_gap_60d"]))
        self.assertTrue(factors["price_volatility_20d"].iloc[:20].isna().all())
        self.assertTrue(pd.notna(factors.loc[20, "price_volatility_20d"]))
        self.assertTrue(factors["temperature_deviation_20d"].iloc[:19].isna().all())
        self.assertTrue(pd.notna(factors.loc[19, "temperature_deviation_20d"]))

    def test_matrix_is_target_free_and_marks_latest_row(self) -> None:
        raw = synthetic_raw()
        factors = self.builder.build_factors(raw)
        matrix = self.builder.build_matrix(raw, factors)

        forbidden = ("target", "next_day", "spike")
        self.assertFalse(any(token in column.lower() for column in matrix for token in forbidden))
        self.assertEqual(len(matrix), len(raw))
        self.assertEqual(matrix["is_latest_observation"].sum(), 1)
        self.assertTrue(bool(matrix.iloc[-1]["is_latest_observation"]))
        self.assertEqual(list(factors.columns), self.builder.output_columns())


if __name__ == "__main__":
    unittest.main()
