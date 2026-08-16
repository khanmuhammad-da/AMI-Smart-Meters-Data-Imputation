# =============================================================================
# src/feature_engineering.py
# =============================================================================
"""
Random Forest CDP Load Profile Imputation
Feature Engineering Stage

Responsibilities
----------------
1. Load the validated single-meter dataframe.
2. Validate the input dataframe.
3. Create deterministic, timestamp-derived features.
4. DO NOT create target lags.
5. DO NOT create target leads.
6. DO NOT create gap information.
7. DO NOT create supervised samples.
8. DO NOT normalize/scale data.
9. Validate engineered features.
10. Save the feature dataframe.

Important
---------
The target is intentionally NOT used to create features.

This is critical because, during a real missing-LP event, the target
value is unavailable.

The resulting feature dataframe will later be used by:
    gap_generator.py
    supervised_dataset.py
    train.py
    evaluation.py
    impute.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def section(title: str) -> None:
    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def info(message: str) -> None:
    print(message)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> dict:

    config_path = (
        PROJECT_ROOT
        / "config"
        / "config.yaml"
    )

    if not config_path.exists():

        fail(
            "Configuration file not found:\n"
            f"    {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = yaml.safe_load(file)

    if not isinstance(config, dict):

        fail(
            "config.yaml did not load as a dictionary."
        )

    return config


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_config(
    config: dict,
    *keys: str,
    default=None
):
    """
    Safely retrieve nested configuration values.
    """

    value = config

    for key in keys:

        if not isinstance(value, dict):
            return default

        if key not in value:
            return default

        value = value[key]

    return value


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_input_path(
    config: dict
) -> Path:

    selected_meter = get_config(
        config,
        "data",
        "selected_meter"
    )

    target_direction = get_config(
        config,
        "data",
        "target_direction"
    )

    processed_dir = get_config(
        config,
        "outputs",
        "processed_data_dir",
        default="data/processed"
    )

    filename = (
        f"{selected_meter}_"
        f"{target_direction}_selected.parquet"
    )

    path = (
        PROJECT_ROOT
        / processed_dir
        / filename
    )

    return path


def get_output_path(
    config: dict
) -> Path:

    selected_meter = get_config(
        config,
        "data",
        "selected_meter"
    )

    target_direction = get_config(
        config,
        "data",
        "target_direction"
    )

    processed_dir = get_config(
        config,
        "outputs",
        "processed_data_dir",
        default="data/processed"
    )

    filename = (
        f"{selected_meter}_"
        f"{target_direction}_features.parquet"
    )

    path = (
        PROJECT_ROOT
        / processed_dir
        / filename
    )

    return path


def get_metadata_path(
    config: dict
) -> Path:

    processed_dir = get_config(
        config,
        "outputs",
        "processed_data_dir",
        default="data/processed"
    )

    return (
        PROJECT_ROOT
        / processed_dir
        / "feature_engineering_metadata.json"
    )


# =============================================================================
# LOAD SELECTED METER
# =============================================================================

def load_selected_meter(
    config: dict
) -> pd.DataFrame:

    input_path = get_input_path(config)

    section("LOADING SELECTED METER DATAFRAME")

    info(
        f"Input file : {input_path}"
    )

    if not input_path.exists():

        fail(
            "Selected meter dataframe was not found:\n"
            f"    {input_path}\n\n"
            "Run the dataset stage first:\n"
            "    python -m src.dataset"
        )

    dataframe = pd.read_parquet(
        input_path
    )

    info(
        f"Rows       : {len(dataframe):,}"
    )

    info(
        f"Columns    : {len(dataframe.columns):,}"
    )

    return dataframe


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_input_dataframe(
    dataframe: pd.DataFrame,
    config: dict
) -> None:

    section("FEATURE ENGINEERING INPUT VALIDATION")

    required_columns = [
        "Time",
        "target",
        "opposite",
        "meter_id",
        "target_direction",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        fail(
            "Required columns are missing:\n"
            f"{missing_columns}"
        )

    info(
        "Required columns       : PASSED"
    )

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if len(dataframe) == 0:

        fail(
            "Input dataframe is empty."
        )

    info(
        f"Rows                   : {len(dataframe):,}"
    )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):

        fail(
            "Time column is not datetime."
        )

    info(
        "Timestamp type         : PASSED"
    )

    # -------------------------------------------------------------------------
    # Timestamp uniqueness
    # -------------------------------------------------------------------------

    if dataframe["Time"].duplicated().any():

        fail(
            "Duplicate timestamps detected."
        )

    info(
        "Timestamp uniqueness   : PASSED"
    )

    # -------------------------------------------------------------------------
    # Timestamp ordering
    # -------------------------------------------------------------------------

    if not dataframe["Time"].is_monotonic_increasing:

        fail(
            "Timestamps are not monotonically increasing."
        )

    info(
        "Timestamp ordering     : PASSED"
    )

    # -------------------------------------------------------------------------
    # 30-minute interval
    # -------------------------------------------------------------------------

    expected_minutes = get_config(
        config,
        "data",
        "expected_interval_minutes",
        default=30
    )

    time_difference = (
        dataframe["Time"]
        .diff()
        .dropna()
    )

    expected_delta = pd.Timedelta(
        minutes=expected_minutes
    )

    if not (time_difference == expected_delta).all():

        unique_intervals = (
            time_difference
            .value_counts()
            .head(10)
        )

        fail(
            "Timestamp interval validation failed.\n"
            f"Expected: {expected_delta}\n"
            f"Observed:\n{unique_intervals}"
        )

    info(
        f"{expected_minutes}-minute interval   : PASSED"
    )

    # -------------------------------------------------------------------------
    # Target numeric
    # -------------------------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        dataframe["target"]
    ):

        fail(
            "Target column is not numeric."
        )

    if dataframe["target"].isna().any():

        fail(
            "Target contains NaN values before feature engineering."
        )

    if np.isinf(
        dataframe["target"].to_numpy(
            dtype=np.float64
        )
    ).any():

        fail(
            "Target contains infinite values."
        )

    info(
        "Target numerical data : PASSED"
    )

    # -------------------------------------------------------------------------
    # Opposite direction
    # -------------------------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        dataframe["opposite"]
    ):

        fail(
            "Opposite-direction column is not numeric."
        )

    if dataframe["opposite"].isna().any():

        fail(
            "Opposite-direction column contains NaN values."
        )

    if np.isinf(
        dataframe["opposite"].to_numpy(
            dtype=np.float64
        )
    ).any():

        fail(
            "Opposite-direction column contains infinite values."
        )

    info(
        "Opposite direction    : PASSED"
    )

    # -------------------------------------------------------------------------
    # Single meter
    # -------------------------------------------------------------------------

    meter_count = dataframe[
        "meter_id"
    ].nunique()

    if meter_count != 1:

        fail(
            "Input dataframe contains more than one meter.\n"
            f"Meter count: {meter_count}"
        )

    info(
        "Single meter          : PASSED"
    )

    # -------------------------------------------------------------------------
    # Single direction
    # -------------------------------------------------------------------------

    direction_count = dataframe[
        "target_direction"
    ].nunique()

    if direction_count != 1:

        fail(
            "Input dataframe contains more than one target direction.\n"
            f"Direction count: {direction_count}"
        )

    info(
        "Single direction      : PASSED"
    )


# =============================================================================
# CREATE DETERMINISTIC FEATURES
# =============================================================================

def create_features(
    dataframe: pd.DataFrame,
    config: dict
) -> tuple[pd.DataFrame, list[str]]:

    section("CREATING DETERMINISTIC FEATURES")

    df = dataframe.copy()

    timestamp = df["Time"]

    feature_columns: list[str] = []

    # =========================================================================
    # CALENDAR FEATURES
    # =========================================================================

    include_calendar = get_config(
        config,
        "feature_engineering",
        "include_calendar_features",
        default=True
    )

    include_cyclic = get_config(
        config,
        "feature_engineering",
        "include_cyclic_features",
        default=True
    )

    if include_calendar:

        # ---------------------------------------------------------------------
        # Basic clock features
        # ---------------------------------------------------------------------

        df["hour"] = timestamp.dt.hour.astype(
            np.int16
        )

        df["minute"] = timestamp.dt.minute.astype(
            np.int16
        )

        # 0 ... 47
        df["half_hour_slot"] = (
            timestamp.dt.hour * 2
            + (
                timestamp.dt.minute // 30
            )
        ).astype(
            np.int16
        )

        # Monday = 0 ... Sunday = 6
        df["day_of_week"] = (
            timestamp.dt.dayofweek
            .astype(np.int16)
        )

        df["day_of_month"] = (
            timestamp.dt.day
            .astype(np.int16)
        )

        df["day_of_year"] = (
            timestamp.dt.dayofyear
            .astype(np.int16)
        )

        df["week_of_year"] = (
            timestamp.dt.isocalendar()
            .week
            .astype(np.int16)
        )

        df["month"] = (
            timestamp.dt.month
            .astype(np.int16)
        )

        df["quarter"] = (
            timestamp.dt.quarter
            .astype(np.int16)
        )

        # ---------------------------------------------------------------------
        # Boolean/calendar indicators
        # ---------------------------------------------------------------------

        df["is_weekend"] = (
            timestamp.dt.dayofweek >= 5
        ).astype(
            np.int8
        )

        df["is_month_start"] = (
            timestamp.dt.is_month_start
        ).astype(
            np.int8
        )

        df["is_month_end"] = (
            timestamp.dt.is_month_end
        ).astype(
            np.int8
        )

        # ---------------------------------------------------------------------
        # Start/end of day
        # ---------------------------------------------------------------------

        df["is_day_start"] = (
            (
                timestamp.dt.hour == 0
            )
            &
            (
                timestamp.dt.minute == 0
            )
        ).astype(
            np.int8
        )

        df["is_day_end"] = (
            (
                timestamp.dt.hour == 23
            )
            &
            (
                timestamp.dt.minute == 30
            )
        ).astype(
            np.int8
        )

        feature_columns.extend([
            "hour",
            "minute",
            "half_hour_slot",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_day_start",
            "is_day_end",
        ])

    # =========================================================================
    # CYCLIC FEATURES
    # =========================================================================

    if include_cyclic:

        # ---------------------------------------------------------------------
        # Half-hour slot
        # 48 observations/day
        # ---------------------------------------------------------------------

        slot = df[
            "half_hour_slot"
        ].astype(
            np.float64
        )

        df["half_hour_sin"] = np.sin(
            2.0
            * math.pi
            * slot
            / 48.0
        )

        df["half_hour_cos"] = np.cos(
            2.0
            * math.pi
            * slot
            / 48.0
        )

        # ---------------------------------------------------------------------
        # Hour
        # 24 hours/day
        # ---------------------------------------------------------------------

        hour = df[
            "hour"
        ].astype(
            np.float64
        )

        df["hour_sin"] = np.sin(
            2.0
            * math.pi
            * hour
            / 24.0
        )

        df["hour_cos"] = np.cos(
            2.0
            * math.pi
            * hour
            / 24.0
        )

        # ---------------------------------------------------------------------
        # Day of week
        # 7 days/week
        # ---------------------------------------------------------------------

        dow = df[
            "day_of_week"
        ].astype(
            np.float64
        )

        df["day_of_week_sin"] = np.sin(
            2.0
            * math.pi
            * dow
            / 7.0
        )

        df["day_of_week_cos"] = np.cos(
            2.0
            * math.pi
            * dow
            / 7.0
        )

        # ---------------------------------------------------------------------
        # Day of year
        # ---------------------------------------------------------------------

        doy = (
            df[
                "day_of_year"
            ]
            .astype(np.float64)
            - 1.0
        )

        # 365 because this dataset covers a normal
        # July-to-July non-leap-year period.
        df["day_of_year_sin"] = np.sin(
            2.0
            * math.pi
            * doy
            / 365.0
        )

        df["day_of_year_cos"] = np.cos(
            2.0
            * math.pi
            * doy
            / 365.0
        )

        feature_columns.extend([
            "half_hour_sin",
            "half_hour_cos",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ])

    # =========================================================================
    # TARGET LAGS / LEADS
    # =========================================================================

    include_lags = get_config(
        config,
        "feature_engineering",
        "include_target_lags",
        default=False
    )

    include_leads = get_config(
        config,
        "feature_engineering",
        "include_target_leads",
        default=False
    )

    if include_lags:

        fail(
            "Target-derived lag features are disabled "
            "for this project and must remain disabled."
        )

    if include_leads:

        fail(
            "Target-derived lead features are disabled "
            "for this project and must remain disabled."
        )

    return df, feature_columns


# =============================================================================
# FEATURE VALIDATION
# =============================================================================

def validate_features(
    original_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    feature_columns: list[str]
) -> None:

    section("FEATURE VALIDATION")

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if len(
        original_dataframe
    ) != len(
        feature_dataframe
    ):

        fail(
            "Row count changed during feature engineering."
        )

    info(
        "Row count              : PASSED"
    )

    # -------------------------------------------------------------------------
    # Timestamp preservation
    # -------------------------------------------------------------------------

    if not original_dataframe[
        "Time"
    ].reset_index(
        drop=True
    ).equals(
        feature_dataframe[
            "Time"
        ].reset_index(
            drop=True
        )
    ):

        fail(
            "Timestamp values changed."
        )

    info(
        "Timestamp preservation : PASSED"
    )

    # -------------------------------------------------------------------------
    # Target preservation
    # -------------------------------------------------------------------------

    if not np.array_equal(
        original_dataframe[
            "target"
        ].to_numpy(),
        feature_dataframe[
            "target"
        ].to_numpy(),
        equal_nan=True
    ):

        fail(
            "Target values changed during feature engineering."
        )

    info(
        "Target preservation    : PASSED"
    )

    # -------------------------------------------------------------------------
    # Opposite preservation
    # -------------------------------------------------------------------------

    if not np.array_equal(
        original_dataframe[
            "opposite"
        ].to_numpy(),
        feature_dataframe[
            "opposite"
        ].to_numpy(),
        equal_nan=True
    ):

        fail(
            "Opposite-direction values changed."
        )

    info(
        "Opposite preservation  : PASSED"
    )

    # -------------------------------------------------------------------------
    # Meter preservation
    # -------------------------------------------------------------------------

    if not original_dataframe[
        "meter_id"
    ].reset_index(
        drop=True
    ).equals(
        feature_dataframe[
            "meter_id"
        ].reset_index(
            drop=True
        )
    ):

        fail(
            "Meter ID changed."
        )

    info(
        "Meter ID preservation  : PASSED"
    )

    # -------------------------------------------------------------------------
    # Direction preservation
    # -------------------------------------------------------------------------

    if not original_dataframe[
        "target_direction"
    ].reset_index(
        drop=True
    ).equals(
        feature_dataframe[
            "target_direction"
        ].reset_index(
            drop=True
        )
    ):

        fail(
            "Target direction changed."
        )

    info(
        "Direction preservation : PASSED"
    )

    # -------------------------------------------------------------------------
    # Expected feature columns
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in feature_dataframe.columns
    ]

    if missing_features:

        fail(
            "Expected engineered features are missing:\n"
            f"{missing_features}"
        )

    info(
        "Expected features      : PASSED"
    )

    # -------------------------------------------------------------------------
    # Feature NaN
    # -------------------------------------------------------------------------

    feature_nan_count = int(
        feature_dataframe[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if feature_nan_count != 0:

        fail(
            "Engineered features contain NaN values.\n"
            f"NaN count: {feature_nan_count}"
        )

    info(
        "Feature NaN check      : PASSED"
    )

    # -------------------------------------------------------------------------
    # Feature infinity
    # -------------------------------------------------------------------------

    numeric_features = feature_dataframe[
        feature_columns
    ].select_dtypes(
        include=[np.number]
    )

    infinity_count = int(
        np.isinf(
            numeric_features.to_numpy(
                dtype=np.float64
            )
        ).sum()
    )

    if infinity_count != 0:

        fail(
            "Engineered features contain infinite values.\n"
            f"Inf count: {infinity_count}"
        )

    info(
        "Feature Inf check      : PASSED"
    )

    # -------------------------------------------------------------------------
    # Half-hour slot
    # -------------------------------------------------------------------------

    if "half_hour_slot" in feature_dataframe.columns:

        slot = feature_dataframe[
            "half_hour_slot"
        ]

        if not (
            slot.between(
                0,
                47
            ).all()
        ):

            fail(
                "half_hour_slot is outside 0–47."
            )

    info(
        "Half-hour slot range   : PASSED"
    )

    # -------------------------------------------------------------------------
    # Cyclic feature bounds
    # -------------------------------------------------------------------------

    cyclic_columns = [
        "half_hour_sin",
        "half_hour_cos",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    existing_cyclic = [
        column
        for column in cyclic_columns
        if column in feature_dataframe.columns
    ]

    for column in existing_cyclic:

        values = feature_dataframe[
            column
        ]

        if not (
            values.between(
                -1.000001,
                1.000001
            ).all()
        ):

            fail(
                f"Cyclic feature '{column}' "
                "is outside [-1, 1]."
            )

    info(
        "Cyclic feature bounds  : PASSED"
    )

    # -------------------------------------------------------------------------
    # TARGET LEAKAGE CHECK
    # -------------------------------------------------------------------------

    forbidden_terms = [
        "target_lag",
        "target_lead",
        "target_previous",
        "target_next",
        "target_shift",
    ]

    leakage_columns = []

    for column in feature_dataframe.columns:

        lower_column = column.lower()

        for forbidden in forbidden_terms:

            if forbidden in lower_column:

                leakage_columns.append(
                    column
                )

                break

    if leakage_columns:

        fail(
            "Potential target-derived feature detected:\n"
            f"{leakage_columns}"
        )

    info(
        "Target leakage check   : PASSED"
    )


# =============================================================================
# FEATURE SUMMARY
# =============================================================================

def print_feature_summary(
    dataframe: pd.DataFrame,
    feature_columns: list[str]
) -> None:

    section("FEATURE SUMMARY")

    original_columns = [
        "Time",
        "target",
        "opposite",
        "meter_id",
        "target_direction",
    ]

    info(
        f"Total rows             : {len(dataframe):,}"
    )

    info(
        f"Total columns          : {len(dataframe.columns):,}"
    )

    info(
        f"Original columns       : {len(original_columns)}"
    )

    info(
        f"Engineered columns     : {len(feature_columns)}"
    )

    print()

    info(
        "Engineered feature columns:"
    )

    for number, column in enumerate(
        feature_columns,
        start=1
    ):

        info(
            f"  {number:2d}. {column}"
        )

    print()

    info(
        "Feature dataframe preview:"
    )

    preview_columns = [
        "Time",
        "target",
        *feature_columns[:8],
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in dataframe.columns
    ]

    print(
        dataframe[
            preview_columns
        ].head(10).to_string(
            index=False
        )
    )


# =============================================================================
# SAVE FEATURE DATAFRAME
# =============================================================================

def save_feature_dataframe(
    dataframe: pd.DataFrame,
    config: dict,
    feature_columns: list[str]
) -> None:

    section("SAVING FEATURE DATAFRAME")

    output_path = get_output_path(
        config
    )

    metadata_path = get_metadata_path(
        config
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Save parquet
    # -------------------------------------------------------------------------

    dataframe.to_parquet(
        output_path,
        index=False
    )

    info(
        "Saved dataframe:\n"
        f"    {output_path}"
    )

    info(
        f"Shape: {dataframe.shape}"
    )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    metadata = {

        "project": get_config(
            config,
            "project",
            "name"
        ),

        "selected_meter": get_config(
            config,
            "data",
            "selected_meter"
        ),

        "target_direction": get_config(
            config,
            "data",
            "target_direction"
        ),

        "input_file": str(
            get_input_path(
                config
            )
        ),

        "output_file": str(
            output_path
        ),

        "rows": int(
            len(dataframe)
        ),

        "columns": int(
            len(dataframe.columns)
        ),

        "original_columns": [
            "Time",
            "target",
            "opposite",
            "meter_id",
            "target_direction",
        ],

        "engineered_features": feature_columns,

        "number_engineered_features": len(
            feature_columns
        ),

        "target_lags": False,

        "target_leads": False,

        "normalization": False,

        "gap_generation": False,

        "supervised_dataset": False,

        "model_training": False,

        "source_modified": False,

        "timestamp_interval_minutes": get_config(
            config,
            "data",
            "expected_interval_minutes",
            default=30
        ),

    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    info(
        "Metadata saved:\n"
        f"    {metadata_path}"
    )


# =============================================================================
# SAVED FILE VERIFICATION
# =============================================================================

def verify_saved_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path
) -> None:

    section("SAVED DATAFRAME VERIFICATION")

    if not output_path.exists():

        fail(
            "Feature dataframe was not created."
        )

    reloaded = pd.read_parquet(
        output_path
    )

    if reloaded.shape != dataframe.shape:

        fail(
            "Saved dataframe shape does not match "
            "the original dataframe.\n"
            f"Original: {dataframe.shape}\n"
            f"Reloaded: {reloaded.shape}"
        )

    info(
        "Parquet reload          : PASSED"
    )

    info(
        f"Verified shape          : {reloaded.shape}"
    )

    # -------------------------------------------------------------------------
    # Timestamp check
    # -------------------------------------------------------------------------

    if not reloaded[
        "Time"
    ].equals(
        dataframe[
            "Time"
        ]
    ):

        fail(
            "Timestamp mismatch after parquet reload."
        )

    # -------------------------------------------------------------------------
    # Target check
    # -------------------------------------------------------------------------

    if not np.array_equal(
        reloaded[
            "target"
        ].to_numpy(),
        dataframe[
            "target"
        ].to_numpy(),
        equal_nan=True
    ):

        fail(
            "Target mismatch after parquet reload."
        )

    # -------------------------------------------------------------------------
    # Feature columns check
    # -------------------------------------------------------------------------

    if list(
        reloaded.columns
    ) != list(
        dataframe.columns
    ):

        fail(
            "Column order/name mismatch after parquet reload."
        )

    info(
        "Data integrity          : PASSED"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    section(
        "RANDOM FOREST CDP IMPUTATION — "
        "FEATURE ENGINEERING STAGE"
    )

    info(
        "Stage: selected meter → deterministic features"
    )

    info(
        "Target-derived lag/lead features: DISABLED"
    )

    info(
        "Gap simulation: DISABLED"
    )

    info(
        "Supervised dataset creation: DISABLED"
    )

    info(
        "Model training: DISABLED"
    )

    info(
        "Evaluation: DISABLED"
    )

    info(
        "Normalization: DISABLED"
    )

    info(
        "Source modification: DISABLED"
    )

    # =========================================================================
    # CONFIG
    # =========================================================================

    config = load_config()

    # =========================================================================
    # LOAD
    # =========================================================================

    dataframe = load_selected_meter(
        config
    )

    # =========================================================================
    # VALIDATE INPUT
    # =========================================================================

    validate_input_dataframe(
        dataframe,
        config
    )

    # =========================================================================
    # CREATE FEATURES
    # =========================================================================

    feature_dataframe, feature_columns = create_features(
        dataframe,
        config
    )

    # =========================================================================
    # VALIDATE FEATURES
    # =========================================================================

    validate_features(
        original_dataframe=dataframe,
        feature_dataframe=feature_dataframe,
        feature_columns=feature_columns
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_feature_summary(
        feature_dataframe,
        feature_columns
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_feature_dataframe(
        feature_dataframe,
        config,
        feature_columns
    )

    # =========================================================================
    # VERIFY
    # =========================================================================

    verify_saved_dataframe(
        feature_dataframe,
        get_output_path(config)
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    section(
        "FEATURE ENGINEERING STAGE COMPLETE"
    )

    info(
        "Selected meter dataframe : VALIDATED"
    )

    info(
        "Calendar features        : CREATED"
    )

    info(
        "Cyclic features          : CREATED"
    )

    info(
        "Target-derived features  : NOT CREATED"
    )

    info(
        "Target leakage check     : PASSED"
    )

    info(
        "Gap generation           : NOT PERFORMED"
    )

    info(
        "Supervised dataset       : NOT CREATED"
    )

    info(
        "Model training           : NOT PERFORMED"
    )

    info(
        "Normalization            : NOT PERFORMED"
    )

    info(
        "Output:"
    )

    info(
        f"    {get_output_path(config)}"
    )

    info(
        "\nNEXT STAGE:"
    )

    info(
        "Review the feature dataframe."
    )

    info(
        "Then generate the independent "
        "1, 6, 24 and 48 LP gap experiments."
    )

    info(
        "Each experiment will use fixed event windows."
    )

    info(
        "The 96 LP gap experiment has been removed."
    )

    info(
        "Random Forest has NOT been trained."
    )

    info(
        "=" * 80
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()