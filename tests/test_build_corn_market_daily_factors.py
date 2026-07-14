from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_corn_market_daily_factors.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_corn_market_daily_factors", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_sources(rows: int = 25) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    normalized_rows: list[dict[str, object]] = []
    record_number = 0

    def add(date: pd.Timestamp, code: str, price: float, region: str, source: str, confidence: float) -> None:
        nonlocal record_number
        record_number += 1
        normalized_rows.append(
            {
                "记录ID": f"id-{record_number}",
                "是否可训练": "yes",
                "状态": "normalized",
                "产品编码": code,
                "价格": price,
                "单位": "元/吨",
                "地区": region,
                "置信度": confidence,
                "来源名称": source,
                "发布时间": date.strftime("%m/%d 08:00"),
                "发布日期": date.strftime("%Y-%m-%d"),
                "是否授权来源": "unknown",
            }
        )

    for index, date in enumerate(dates):
        corn = 100.0 + index
        for offset, region, source, confidence in (
            (-1.0, "north", "source-a", 0.8),
            (0.0, "central", "source-b", 0.9),
            (1.0, "south", "source-c", 1.0),
        ):
            add(date, "corn", corn + offset, region, source, confidence)
        add(date, "corn_starch", 130.0 + index, "national", "source-s", 0.9)
        add(date, "corn_husk", 200.0 + index, "national", "source-h", 0.8)
        add(date, "germ", 300.0 + 2.0 * index, "national", "source-g", 0.8)
        add(date, "protein_powder", 400.0 + 3.0 * index, "national", "source-p", 0.8)
        add(date, "corn_glucose", 500.0 + index, "national", "source-cg", 0.8)
        add(date, "corn_fructose", 600.0 + 2.0 * index, "national", "source-cf", 0.8)
        add(date, "maltodextrin", 700.0 + 3.0 * index, "national", "source-m", 0.8)

    normalized = pd.DataFrame(normalized_rows)
    raw = normalized[["记录ID"]].copy()
    dce = pd.DataFrame(
        {
            "date": dates.strftime("%Y/%m/%d"),
            "dce_corn_close": 90.0 + np.arange(rows, dtype=float),
        }
    )
    return raw, normalized, dce


class MarketDailyFactorBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()

    def test_source_ids_must_match(self) -> None:
        raw, normalized, _ = synthetic_sources()
        self.builder.validate_source_pair(raw, normalized)

        mismatched = normalized.iloc[:-1].copy()
        with self.assertRaisesRegex(AssertionError, "record IDs"):
            self.builder.validate_source_pair(raw, mismatched)

    def test_release_time_maps_to_first_eligible_dce_row(self) -> None:
        calendar = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]))
        releases = pd.DataFrame(
            {
                "发布日期": ["2025-01-03", "2025-01-03", "2025-01-04"],
                "发布时间": ["01/03 14:00", "01/03 16:00", "01/04 08:00"],
            }
        )

        available = self.builder.map_available_dates(releases, calendar, cutoff="15:00")

        self.assertEqual(available.dt.strftime("%Y-%m-%d").tolist(), [
            "2025-01-03",
            "2025-01-06",
            "2025-01-06",
        ])

    def test_daily_factors_are_target_free_and_use_robust_aggregates(self) -> None:
        raw, normalized, dce = synthetic_sources()
        self.builder.validate_source_pair(raw, normalized)
        panel = self.builder.build_daily_panel(normalized, dce)
        factors = self.builder.build_factors(panel)
        matrix = self.builder.build_matrix(dce, panel, factors)

        self.assertEqual(list(factors.columns), self.builder.output_columns())
        self.assertEqual(len(factors.columns), 12)
        self.assertFalse(any("target" in column.lower() for column in matrix.columns))
        self.assertAlmostEqual(factors.loc[0, "corn_spot_dce_basis"], 100.0 / 90.0 - 1.0)
        self.assertAlmostEqual(factors.loc[0, "corn_regional_dispersion"], 1.0 / 100.0)
        self.assertAlmostEqual(factors.loc[0, "corn_quote_count_log"], np.log1p(3.0))
        self.assertAlmostEqual(factors.loc[0, "corn_source_count_log"], np.log1p(3.0))
        self.assertAlmostEqual(factors.loc[0, "corn_confidence_mean"], 0.9)
        self.assertAlmostEqual(factors.loc[5, "corn_spot_momentum_5d"], 105.0 / 100.0 - 1.0)
        self.assertTrue(factors["corn_spot_momentum_20d"].iloc[:20].isna().all())
        self.assertTrue(pd.notna(factors.loc[20, "corn_spot_momentum_20d"]))
        self.assertTrue(factors["processing_chain_momentum_20d"].iloc[:20].isna().all())
        self.assertTrue(pd.notna(factors.loc[20, "processing_chain_momentum_20d"]))
        self.assertFalse(bool(matrix["point_in_time_verified"].any()))


if __name__ == "__main__":
    unittest.main()
