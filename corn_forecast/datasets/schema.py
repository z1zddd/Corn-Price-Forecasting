"""Lightweight dataset schema declarations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FieldSpec:
    """One expected field in a dataset material."""

    name: str
    dtype: str
    required: bool
    description: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSchema:
    """Human-readable schema metadata for a dataset material."""

    name: str
    description: str
    fields: tuple[FieldSpec, ...]
    notes: tuple[str, ...] = ()

    @property
    def required_fields(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.required)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "required_fields": self.required_fields,
            "fields": [field.as_dict() for field in self.fields],
            "notes": self.notes,
        }


ALL_ROLLING_PREDICTIONS_SCHEMA = DatasetSchema(
    name="all_rolling_predictions",
    description="Completed rolling prediction streams consumed by deployment vote aggregation operators.",
    fields=(
        FieldSpec("model", "string", True, "Base model or prediction stream name."),
        FieldSpec("feature_set", "string", True, "Feature material variant, such as no_news or with_news_precomputed_pca."),
        FieldSpec("lookback_months", "integer", True, "Lookback window length used to produce the prediction stream."),
        FieldSpec("horizon_months", "integer", True, "Forecast horizon in monthly periods."),
        FieldSpec("head", "string", True, "Prediction head, commonly cls or reg."),
        FieldSpec("anchor_month", "date/month", True, "Information date when the prediction is anchored."),
        FieldSpec("target_month", "date/month", True, "Forward target month being predicted."),
        FieldSpec("actual_direction", "integer", True, "Realized binary direction label, normally 0 or 1."),
        FieldSpec("actual_return", "float", True, "Realized forward return for the target month."),
        FieldSpec("predicted_direction", "integer", True, "Binary prediction emitted by the base stream."),
        FieldSpec("predicted_probability", "float", True, "Positive-class score or probability emitted by the base stream."),
    ),
    notes=(
        "This is a prediction-library material, not a raw feature table.",
        "Deployment vote aggregation consumes these streams without retraining base models.",
    ),
)


CORN_MONTHLY_SCHEMA = DatasetSchema(
    name="corn_monthly_modeling_table",
    description="Monthly corn modeling table used by configs such as configs/corn.yaml.",
    fields=(
        FieldSpec("month", "date/month", True, "Monthly timestamp column used as the default corn date column."),
        FieldSpec("dce_corn_close", "float", True, "Default corn price column used to generate forward targets."),
        FieldSpec("feature_columns", "numeric columns", True, "Model inputs selected explicitly or through auto_numeric rules."),
        FieldSpec("target_price_fwd", "float", False, "Generated or precomputed forward target price."),
        FieldSpec("target_return_fwd", "float", False, "Generated or precomputed forward return."),
        FieldSpec("target_direction_fwd", "integer", False, "Generated or precomputed binary direction target."),
        FieldSpec("target_date_fwd", "date/month", False, "Forward target date for leakage-aware training windows."),
        FieldSpec("pca_*", "float", False, "News or text-derived PCA feature columns when a with-news material is used."),
        FieldSpec("spike", "integer/float", False, "Legacy target-like column that should be excluded from feature inputs."),
    ),
    notes=(
        "Configs may use date_col=month and price_col=dce_corn_close.",
        "Target-like columns should be excluded from feature inputs.",
        "No-news configs exclude pca_* and PCA* columns; with-news configs retain them.",
    ),
)


SCHEMAS: dict[str, DatasetSchema] = {
    ALL_ROLLING_PREDICTIONS_SCHEMA.name: ALL_ROLLING_PREDICTIONS_SCHEMA,
    CORN_MONTHLY_SCHEMA.name: CORN_MONTHLY_SCHEMA,
}


def get_schema(name: str) -> DatasetSchema:
    """Return schema metadata by name."""

    try:
        return SCHEMAS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset schema: {name}") from exc


def list_schemas() -> list[dict[str, object]]:
    """Return all schema metadata as plain dictionaries."""

    return [schema.as_dict() for schema in SCHEMAS.values()]
