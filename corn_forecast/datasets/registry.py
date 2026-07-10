"""Lightweight dataset material registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetAsset:
    """One registered dataset material or material slot."""

    name: str
    commodity: str
    layer: str
    path: str
    schema: str
    description: str
    tracked_in_git: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DATASET_ASSETS: tuple[DatasetAsset, ...] = (
    DatasetAsset(
        name="corn_sample_data",
        commodity="corn",
        layer="compat_sample",
        path="corn_forecast/datasets/corn/processed/corn_sample_data.csv",
        schema="corn_monthly_modeling_table",
        description="Small processed corn CSV used by templates and quick diagnostics.",
        tracked_in_git=True,
        notes=("Moved from the old root datasets/ directory into the corn processed material area.",),
    ),
    DatasetAsset(
        name="corn_monthly_mixed_features",
        commodity="corn",
        layer="processed",
        path="corn_forecast/datasets/corn/processed/玉米价格月度_混合特征版 .csv",
        schema="corn_monthly_modeling_table",
        description="Repository-compatible monthly corn feature table with mixed price, factor, and PCA materials.",
        tracked_in_git=True,
        notes=("Monthly modeling fixture kept under corn processed materials.",),
    ),
    DatasetAsset(
        name="corn_monthly_mixed_features_no_missing_dual_lstm",
        commodity="corn",
        layer="processed",
        path="corn_forecast/datasets/corn/processed/玉米价格月度_混合特征无缺失值双头LSTM版.csv",
        schema="corn_monthly_modeling_table",
        description="Default monthly corn modeling table used by configs/corn.yaml.",
        tracked_in_git=True,
        notes=("Default tracked processed fixture for corn configs.",),
    ),
    DatasetAsset(
        name="corn_raw_price_source",
        commodity="corn",
        layer="raw",
        path="corn_forecast/datasets/corn/raw/玉米价格原始数据.csv",
        schema="corn_monthly_modeling_table",
        description="Small repository-compatible raw corn price source used for reproducible examples.",
        tracked_in_git=True,
        notes=("Private or large raw sources should stay outside git, for example under local_data/.",),
    ),
    DatasetAsset(
        name="corn_all_rolling_predictions",
        commodity="corn",
        layer="prediction_library",
        path="corn_forecast/datasets/corn/prediction_library/all_rolling_predictions.csv",
        schema="all_rolling_predictions",
        description="Expected location and schema for completed rolling prediction streams.",
        tracked_in_git=False,
        notes=("Generated prediction libraries are usually too large for git and should not be committed by default.",),
    ),
)


def list_dataset_assets() -> list[dict[str, object]]:
    """Return registered dataset materials as plain dictionaries."""

    return [asset.as_dict() for asset in DATASET_ASSETS]


def get_dataset_asset(name: str) -> DatasetAsset:
    """Return a registered dataset material by name."""

    for asset in DATASET_ASSETS:
        if asset.name == name:
            return asset
    raise KeyError(f"Unknown dataset asset: {name}")
