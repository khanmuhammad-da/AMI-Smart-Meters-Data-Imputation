"""
================================================================================
RANDOM FOREST CDP IMPUTATION — V2.1 SUPERVISED FEATURE LAYER
================================================================================

V2.1 purpose:
    Reduced feature set derived from V2 feature-importance + ablation analysis.

V1:
    22 original predictor features.

V2:
    43 predictor features.

V2.1:
    29 predictor features.

V2.1 feature composition:
    - 22 V1 original features
    - 3 historical features
    - 4 recent-context features

Removed from V2:
    - 12 context-statistics features
    - 2 trend features

V2.1 predictor features:

    V1 ORIGINAL — 22
    -----------------
    hour
    minute
    half_hour_slot
    day_of_week
    day_of_month
    day_of_year
    week_of_year
    month
    quarter
    is_weekend
    is_month_start
    is_month_end
    is_day_start
    is_day_end
    half_hour_sin
    half_hour_cos
    hour_sin
    hour_cos
    day_of_week_sin
    day_of_week_cos
    day_of_year_sin
    day_of_year_cos

    HISTORICAL — 3
    --------------
    target_previous_day_same_slot
    target_previous_week_same_slot
    previous_week_available

    RECENT CONTEXT — 4
    ------------------
    left_recent_mean
    right_recent_mean
    left_last_value
    right_first_value

Total:
    22 + 3 + 4 = 29 predictor features

Design constraints:
    - V1 pipeline remains untouched.
    - V2 pipeline remains untouched.
    - V1 datasets remain untouched.
    - V2 datasets remain untouched.
    - Source CSV remains untouched.
    - Ground truth is never used as a predictor.
    - Values from the current missing gap are never used as predictors.
    - One supervised sample is created for every missing LP.
    - Fixed event formulation.
    - 96 LP gap is excluded.
    - Gap lengths: 1, 6, 24, 48 LP.
    - Context: 96 LP left + 96 LP right.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

GAPS_DIR = PROCESSED_DIR / "gaps"

OUTPUT_DIR = PROCESSED_DIR / "supervised_v21"

CDP_METER = "CDP_00526_P_01_A-"

GAP_LENGTHS = [1, 6, 24, 48]

CONTEXT_LEFT = 96
CONTEXT_RIGHT = 96

HALF_HOUR_PER_DAY = 48
HALF_HOUR_PER_WEEK = 336


# =============================================================================
# V1 FEATURE DEFINITION
# =============================================================================

V1_FEATURES = [
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
    "half_hour_sin",
    "half_hour_cos",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# =============================================================================
# V2.1 FEATURE DEFINITION
# =============================================================================

HISTORICAL_FEATURES = [
    "target_previous_day_same_slot",
    "target_previous_week_same_slot",
    "previous_week_available",
]


RECENT_CONTEXT_FEATURES = [
    "left_recent_mean",
    "right_recent_mean",
    "left_last_value",
    "right_first_value",
]


V21_FEATURES = (
    V1_FEATURES
    + HISTORICAL_FEATURES
    + RECENT_CONTEXT_FEATURES
)


# =============================================================================
# FEATURES EXPLICITLY REMOVED FROM V2
# =============================================================================

REMOVED_CONTEXT_STATISTICS = [
    "left_mean",
    "left_median",
    "left_std",
    "left_min",
    "left_max",
    "left_range",
    "right_mean",
    "right_median",
    "right_std",
    "right_min",
    "right_max",
    "right_range",
]


REMOVED_TREND_FEATURES = [
    "left_slope",
    "right_slope",
]


# =============================================================================
# NON-PREDICTOR COLUMNS
# =============================================================================

NON_PREDICTOR_COLUMNS = {
    "event_id",
    "gap_length",
    "split",
    "target_index",
    "gap_position",
    "gap_position_fraction",
    "Time",
    "ground_truth",
    "feature_version",
}


# =============================================================================
# LOGGING
# =============================================================================

def separator(char: str = "-", width: int = 80) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    separator()
    print(title)
    separator()


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# BASIC VALIDATION
# =============================================================================

def validate_feature_definition() -> None:

    if len(V1_FEATURES) != 22:
        fail(
            f"Expected 22 V1 features, found {len(V1_FEATURES)}."
        )

    if len(HISTORICAL_FEATURES) != 3:
        fail(
            "Expected 3 historical features."
        )

    if len(RECENT_CONTEXT_FEATURES) != 4:
        fail(
            "Expected 4 recent-context features."
        )

    if len(V21_FEATURES) != 29:
        fail(
            f"Expected 29 V2.1 features, found {len(V21_FEATURES)}."
        )

    if len(set(V21_FEATURES)) != len(V21_FEATURES):
        fail(
            "Duplicate V2.1 feature names detected."
        )

    overlap = set(V21_FEATURES) & set(REMOVED_CONTEXT_STATISTICS)

    if overlap:
        fail(
            f"Removed context-statistics features still present: {sorted(overlap)}"
        )

    overlap = set(V21_FEATURES) & set(REMOVED_TREND_FEATURES)

    if overlap:
        fail(
            f"Removed trend features still present: {sorted(overlap)}"
        )


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def gap_dataframe_path(gap_length: int) -> Path:

    return (
        GAPS_DIR
        / f"gap_{gap_length}"
        / f"{CDP_METER}_gap_{gap_length}.parquet"
    )


def gap_metadata_path(gap_length: int) -> Path:

    return (
        GAPS_DIR
        / f"gap_{gap_length}"
        / "gap_metadata.csv"
    )


# =============================================================================
# LOAD GAP DATA
# =============================================================================

def load_gap_dataframe(gap_length: int) -> pd.DataFrame:

    path = gap_dataframe_path(gap_length)

    if not path.exists():
        fail(
            f"Gap dataframe not found:\n{path}"
        )

    dataframe = pd.read_parquet(path)

    if not isinstance(dataframe, pd.DataFrame):
        fail(
            f"Failed to load dataframe:\n{path}"
        )

    return dataframe


# =============================================================================
# LOAD GAP METADATA
# =============================================================================

def load_gap_metadata(gap_length: int) -> pd.DataFrame:

    path = gap_metadata_path(gap_length)

    if not path.exists():
        fail(
            f"Gap metadata not found:\n{path}"
        )

    metadata = pd.read_csv(path)

    if metadata.empty:
        fail(
            f"Gap metadata is empty:\n{path}"
        )

    # -------------------------------------------------------------------------
    # Normalize column names.
    # -------------------------------------------------------------------------

    metadata.columns = [
        str(column).strip()
        for column in metadata.columns
    ]

    # -------------------------------------------------------------------------
    # The existing project metadata uses:
    #
    # gap_id
    # split
    # gap_length
    # gap_start_index
    # gap_end_index
    # gap_start_time
    # gap_end_time
    # left_context_start_index
    # left_context_end_index
    # right_context_start_index
    # right_context_end_index
    # context_left_length
    # context_right_length
    # gap_length_check
    # -------------------------------------------------------------------------

    required = [
        "gap_id",
        "split",
        "gap_length",
        "gap_start_index",
        "gap_end_index",
        "left_context_start_index",
        "left_context_end_index",
        "right_context_start_index",
        "right_context_end_index",
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: metadata missing columns: {missing}"
        )

    return metadata


# =============================================================================
# GAP DATAFRAME VALIDATION
# =============================================================================

def validate_gap_dataframe(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:

    required = [
        "Time",
        "target",
        "ground_truth",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: missing required columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # Timestamp.
    # -------------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):
        try:
            dataframe["Time"] = pd.to_datetime(
                dataframe["Time"]
            )
        except Exception as exc:
            fail(
                f"Gap {gap_length}: invalid Time column: {exc}"
            )

    if dataframe["Time"].isna().any():
        fail(
            f"Gap {gap_length}: Time contains NaN."
        )

    # -------------------------------------------------------------------------
    # 30-minute continuity.
    # -------------------------------------------------------------------------

    time_diff = (
        dataframe["Time"]
        .sort_values()
        .diff()
        .dropna()
    )

    invalid = time_diff[
        time_diff != pd.Timedelta(minutes=30)
    ]

    if not invalid.empty:
        fail(
            f"Gap {gap_length}: 30-minute continuity validation failed."
        )

    # -------------------------------------------------------------------------
    # Ground truth must be complete.
    # -------------------------------------------------------------------------

    if dataframe["ground_truth"].isna().any():
        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    # -------------------------------------------------------------------------
    # There must be masked targets.
    # -------------------------------------------------------------------------

    if not dataframe["target"].isna().any():
        fail(
            f"Gap {gap_length}: no masked target values found."
        )


# =============================================================================
# METADATA VALIDATION
# =============================================================================

def validate_gap_metadata(
    metadata: pd.DataFrame,
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:

    for _, event in metadata.iterrows():

        event_gap_length = int(
            event["gap_length"]
        )

        if event_gap_length != gap_length:
            fail(
                f"Gap {gap_length}: metadata event "
                f"{event['gap_id']} has gap length "
                f"{event_gap_length}."
            )

        gap_start = int(
            event["gap_start_index"]
        )

        gap_end = int(
            event["gap_end_index"]
        )

        if gap_end - gap_start + 1 != gap_length:
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "gap range does not match gap length."
            )

        left_start = int(
            event["left_context_start_index"]
        )

        left_end = int(
            event["left_context_end_index"]
        )

        right_start = int(
            event["right_context_start_index"]
        )

        right_end = int(
            event["right_context_end_index"]
        )

        if left_end - left_start + 1 != CONTEXT_LEFT:
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "left context length is incorrect."
            )

        if right_end - right_start + 1 != CONTEXT_RIGHT:
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "right context length is incorrect."
            )

        # ---------------------------------------------------------------------
        # Critical boundary check.
        # ---------------------------------------------------------------------

        if left_end >= gap_start:
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "left context enters missing gap."
            )

        if right_start <= gap_end:
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "right context enters missing gap."
            )

        if left_start < 0:
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "left context starts before dataframe."
            )

        if right_end >= len(dataframe):
            fail(
                f"Gap {gap_length}, event {event['gap_id']}: "
                "right context exceeds dataframe."
            )


# =============================================================================
# OBSERVED VALUE ACCESS
# =============================================================================

def observed_value(
    dataframe: pd.DataFrame,
    index: int,
) -> float | None:

    if index < 0 or index >= len(dataframe):
        return None

    value = dataframe.iloc[index]["target"]

    if pd.isna(value):
        return None

    value = float(value)

    if not math.isfinite(value):
        return None

    return value


# =============================================================================
# HISTORICAL VALUE
# =============================================================================

def historical_value(
    dataframe: pd.DataFrame,
    target_index: int,
    lag: int,
    gap_start: int,
    gap_end: int,
) -> Tuple[float, bool]:

    candidate_index = target_index - lag

    # -------------------------------------------------------------------------
    # Never allow the lookup to enter the current missing event.
    # -------------------------------------------------------------------------

    if gap_start <= candidate_index <= gap_end:
        return np.nan, False

    value = observed_value(
        dataframe=dataframe,
        index=candidate_index,
    )

    if value is not None:
        return value, True

    return np.nan, False


# =============================================================================
# PREVIOUS DAY SAME SLOT
# =============================================================================

def previous_day_same_slot(
    dataframe: pd.DataFrame,
    target_index: int,
    gap_start: int,
    gap_end: int,
) -> float:

    value, available = historical_value(
        dataframe=dataframe,
        target_index=target_index,
        lag=HALF_HOUR_PER_DAY,
        gap_start=gap_start,
        gap_end=gap_end,
    )

    if available:
        return value

    # -------------------------------------------------------------------------
    # If unavailable, search backward for the nearest valid observed
    # historical value. This does NOT use the current missing gap.
    # -------------------------------------------------------------------------

    for extra_day in range(2, 8):

        lag = HALF_HOUR_PER_DAY * extra_day

        value, available = historical_value(
            dataframe=dataframe,
            target_index=target_index,
            lag=lag,
            gap_start=gap_start,
            gap_end=gap_end,
        )

        if available:
            return value

    fail(
        "No valid observed previous-day historical value found. "
        f"target_index={target_index}, "
        f"gap={gap_start}:{gap_end}"
    )

    return np.nan


# =============================================================================
# PREVIOUS WEEK SAME SLOT
# =============================================================================

def previous_week_same_slot(
    dataframe: pd.DataFrame,
    target_index: int,
    gap_start: int,
    gap_end: int,
) -> Tuple[float, int]:

    value, available = historical_value(
        dataframe=dataframe,
        target_index=target_index,
        lag=HALF_HOUR_PER_WEEK,
        gap_start=gap_start,
        gap_end=gap_end,
    )

    if available:
        return value, 1

    # -------------------------------------------------------------------------
    # V2 behavior retained:
    #
    # If the exact previous-week slot is unavailable, use the previous-day
    # same-slot value as a deterministic fallback.
    #
    # previous_week_available = 0
    #
    # This flag explicitly tells the model that the exact previous-week value
    # was unavailable.
    # -------------------------------------------------------------------------

    fallback = previous_day_same_slot(
        dataframe=dataframe,
        target_index=target_index,
        gap_start=gap_start,
        gap_end=gap_end,
    )

    return fallback, 0


# =============================================================================
# RECENT CONTEXT
# =============================================================================

def left_recent_values(
    dataframe: pd.DataFrame,
    target_index: int,
    gap_start: int,
) -> List[float]:

    # -------------------------------------------------------------------------
    # Four most recent observed LPs immediately before the gap.
    #
    # For a target inside the same missing event, do NOT use earlier target
    # positions inside the gap.
    #
    # Therefore the anchor is gap_start - 1.
    # -------------------------------------------------------------------------

    values: List[float] = []

    for offset in range(1, 5):

        index = gap_start - offset

        value = observed_value(
            dataframe=dataframe,
            index=index,
        )

        if value is None:
            continue

        values.append(value)

    if not values:
        fail(
            f"No observed left recent context available. "
            f"target_index={target_index}, gap_start={gap_start}"
        )

    return values


def right_recent_values(
    dataframe: pd.DataFrame,
    target_index: int,
    gap_end: int,
) -> List[float]:

    # -------------------------------------------------------------------------
    # Four earliest observed LPs immediately after the gap.
    #
    # Anchor is gap_end + 1.
    # -------------------------------------------------------------------------

    values: List[float] = []

    for offset in range(1, 5):

        index = gap_end + offset

        value = observed_value(
            dataframe=dataframe,
            index=index,
        )

        if value is None:
            continue

        values.append(value)

    if not values:
        fail(
            f"No observed right recent context available. "
            f"target_index={target_index}, gap_end={gap_end}"
        )

    return values


# =============================================================================
# V2.1 FEATURE CREATION
# =============================================================================

def create_v21_features(
    dataframe: pd.DataFrame,
    target_index: int,
    gap_start: int,
    gap_end: int,
) -> Dict[str, float]:

    row = dataframe.iloc[target_index]

    features: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # V1 ORIGINAL FEATURES
    # -------------------------------------------------------------------------

    for feature in V1_FEATURES:

        if feature not in dataframe.columns:
            fail(
                f"Missing V1 feature: {feature}"
            )

        value = row[feature]

        if pd.isna(value):
            fail(
                f"V1 feature {feature} is NaN at "
                f"target index {target_index}."
            )

        value = float(value)

        if not math.isfinite(value):
            fail(
                f"V1 feature {feature} is non-finite at "
                f"target index {target_index}."
            )

        features[feature] = value

    # -------------------------------------------------------------------------
    # HISTORICAL — PREVIOUS DAY
    # -------------------------------------------------------------------------

    previous_day = previous_day_same_slot(
        dataframe=dataframe,
        target_index=target_index,
        gap_start=gap_start,
        gap_end=gap_end,
    )

    features[
        "target_previous_day_same_slot"
    ] = previous_day

    # -------------------------------------------------------------------------
    # HISTORICAL — PREVIOUS WEEK
    # -------------------------------------------------------------------------

    previous_week, previous_week_available = (
        previous_week_same_slot(
            dataframe=dataframe,
            target_index=target_index,
            gap_start=gap_start,
            gap_end=gap_end,
        )
    )

    features[
        "target_previous_week_same_slot"
    ] = previous_week

    features[
        "previous_week_available"
    ] = float(previous_week_available)

    # -------------------------------------------------------------------------
    # RECENT LEFT CONTEXT
    # -------------------------------------------------------------------------

    left_values = left_recent_values(
        dataframe=dataframe,
        target_index=target_index,
        gap_start=gap_start,
    )

    features[
        "left_recent_mean"
    ] = float(np.mean(left_values))

    features[
        "left_last_value"
    ] = float(left_values[0])

    # -------------------------------------------------------------------------
    # RECENT RIGHT CONTEXT
    # -------------------------------------------------------------------------

    right_values = right_recent_values(
        dataframe=dataframe,
        target_index=target_index,
        gap_end=gap_end,
    )

    features[
        "right_recent_mean"
    ] = float(np.mean(right_values))

    features[
        "right_first_value"
    ] = float(right_values[0])

    return features


# =============================================================================
# SAMPLE CREATION
# =============================================================================

def create_event_samples(
    dataframe: pd.DataFrame,
    event: pd.Series,
    gap_length: int,
) -> List[Dict]:

    gap_start = int(
        event["gap_start_index"]
    )

    gap_end = int(
        event["gap_end_index"]
    )

    split = str(
        event["split"]
    )

    event_id = int(
        event["gap_id"]
    )

    # -------------------------------------------------------------------------
    # Validate event.
    # -------------------------------------------------------------------------

    if gap_end - gap_start + 1 != gap_length:
        fail(
            f"Gap {gap_length}, event {event_id}: "
            "gap range does not match gap length."
        )

    samples: List[Dict] = []

    # -------------------------------------------------------------------------
    # One sample per missing LP.
    # -------------------------------------------------------------------------

    for position in range(gap_length):

        target_index = gap_start + position

        row = dataframe.iloc[target_index]

        # ---------------------------------------------------------------------
        # Target must be masked.
        # ---------------------------------------------------------------------

        if not pd.isna(row["target"]):
            fail(
                f"Gap {gap_length}, event {event_id}: "
                f"target index {target_index} is not masked."
            )

        # ---------------------------------------------------------------------
        # Ground truth must exist.
        # ---------------------------------------------------------------------

        ground_truth = row["ground_truth"]

        if pd.isna(ground_truth):
            fail(
                f"Gap {gap_length}, event {event_id}: "
                f"ground truth missing at index {target_index}."
            )

        # ---------------------------------------------------------------------
        # Sample metadata.
        # ---------------------------------------------------------------------

        sample: Dict = {
            "event_id": event_id,
            "gap_length": gap_length,
            "split": split,
            "target_index": target_index,
            "gap_position": position + 1,
            "gap_position_fraction": (
                (position + 1) / gap_length
            ),
            "Time": row["Time"],
            "ground_truth": float(ground_truth),
            "feature_version": "v2.1",
        }

        # ---------------------------------------------------------------------
        # V2.1 predictor features.
        # ---------------------------------------------------------------------

        features = create_v21_features(
            dataframe=dataframe,
            target_index=target_index,
            gap_start=gap_start,
            gap_end=gap_end,
        )

        sample.update(features)

        samples.append(sample)

    return samples


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_supervised_dataset(
    dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    all_samples: List[Dict] = []

    for _, event in metadata.iterrows():

        samples = create_event_samples(
            dataframe=dataframe,
            event=event,
            gap_length=gap_length,
        )

        all_samples.extend(samples)

    if not all_samples:
        fail(
            f"Gap {gap_length}: no V2.1 samples created."
        )

    return pd.DataFrame(all_samples)


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_supervised_dataset(
    supervised: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> None:

    # -------------------------------------------------------------------------
    # Expected row count.
    # -------------------------------------------------------------------------

    expected_rows = len(metadata) * gap_length

    if len(supervised) != expected_rows:
        fail(
            f"Gap {gap_length}: expected "
            f"{expected_rows} rows, found {len(supervised)}."
        )

    # -------------------------------------------------------------------------
    # Feature validation.
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in V21_FEATURES
        if feature not in supervised.columns
    ]

    if missing_features:
        fail(
            f"Gap {gap_length}: missing V2.1 features: "
            f"{missing_features}"
        )

    # -------------------------------------------------------------------------
    # Exact predictor feature count.
    # -------------------------------------------------------------------------

    actual_features = [
        column
        for column in supervised.columns
        if column not in NON_PREDICTOR_COLUMNS
    ]

    if len(actual_features) != 29:
        fail(
            f"Gap {gap_length}: expected 29 predictor features, "
            f"found {len(actual_features)}."
        )

    if set(actual_features) != set(V21_FEATURES):
        fail(
            f"Gap {gap_length}: predictor feature set mismatch."
        )

    # -------------------------------------------------------------------------
    # No forbidden V2 features.
    # -------------------------------------------------------------------------

    forbidden = set(
        REMOVED_CONTEXT_STATISTICS
        + REMOVED_TREND_FEATURES
    )

    present_forbidden = (
        set(supervised.columns) & forbidden
    )

    if present_forbidden:
        fail(
            f"Gap {gap_length}: forbidden features present: "
            f"{sorted(present_forbidden)}"
        )

    # -------------------------------------------------------------------------
    # Ground-truth leakage check.
    # -------------------------------------------------------------------------

    if "ground_truth" not in supervised.columns:
        fail(
            "ground_truth column missing."
        )

    # Ground truth must never be among predictors.
    if "ground_truth" in V21_FEATURES:
        fail(
            "GROUND-TRUTH LEAKAGE: ground_truth is a predictor."
        )

    # -------------------------------------------------------------------------
    # NaN check.
    # -------------------------------------------------------------------------

    feature_data = supervised[V21_FEATURES]

    if feature_data.isna().any().any():
        bad_columns = (
            feature_data.columns[
                feature_data.isna().any()
            ].tolist()
        )

        fail(
            f"Gap {gap_length}: NaN found in features: "
            f"{bad_columns}"
        )

    # -------------------------------------------------------------------------
    # Infinite check.
    # -------------------------------------------------------------------------

    numeric_values = feature_data.select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(
        numeric_values.to_numpy()
    ).all():

        fail(
            f"Gap {gap_length}: infinite/non-finite feature values detected."
        )

    # -------------------------------------------------------------------------
    # Split validation.
    # -------------------------------------------------------------------------

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        supervised["split"].unique()
    )

    if not actual_splits.issubset(valid_splits):
        fail(
            f"Gap {gap_length}: invalid split labels: "
            f"{actual_splits}"
        )

    # -------------------------------------------------------------------------
    # One sample per missing LP.
    # -------------------------------------------------------------------------

    duplicate_targets = supervised.duplicated(
        subset=["event_id", "target_index"]
    )

    if duplicate_targets.any():
        fail(
            f"Gap {gap_length}: duplicate "
            "(event_id, target_index) samples found."
        )

    # -------------------------------------------------------------------------
    # Gap positions.
    # -------------------------------------------------------------------------

    for event_id, group in supervised.groupby(
        "event_id"
    ):

        positions = sorted(
            group["gap_position"].astype(int).tolist()
        )

        expected = list(
            range(1, gap_length + 1)
        )

        if positions != expected:
            fail(
                f"Gap {gap_length}, event {event_id}: "
                "gap positions are invalid."
            )

    # -------------------------------------------------------------------------
    # Feature version.
    # -------------------------------------------------------------------------

    versions = set(
        supervised["feature_version"].astype(str)
    )

    if versions != {"v2.1"}:
        fail(
            f"Unexpected feature versions: {versions}"
        )


# =============================================================================
# V1 PRESERVATION CHECK
# =============================================================================

def validate_v1_preservation(
    dataframe: pd.DataFrame,
    supervised: pd.DataFrame,
    gap_length: int,
) -> None:

    # -------------------------------------------------------------------------
    # V1 features must exist.
    # -------------------------------------------------------------------------

    missing = [
        feature
        for feature in V1_FEATURES
        if feature not in dataframe.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: V1 features missing from source dataframe: "
            f"{missing}"
        )

    # -------------------------------------------------------------------------
    # Reconstruct a small deterministic sample and compare.
    # -------------------------------------------------------------------------

    sample_count = min(
        10,
        len(supervised)
    )

    sample_rows = supervised.head(
        sample_count
    )

    for _, row in sample_rows.iterrows():

        target_index = int(
            row["target_index"]
        )

        source_row = dataframe.iloc[
            target_index
        ]

        for feature in V1_FEATURES:

            source_value = float(
                source_row[feature]
            )

            supervised_value = float(
                row[feature]
            )

            if not np.isclose(
                source_value,
                supervised_value,
                rtol=0.0,
                atol=1e-9,
            ):
                fail(
                    f"V1 preservation failed for "
                    f"{feature} at target index "
                    f"{target_index}."
                )


# =============================================================================
# SAVE DATASET
# =============================================================================

def save_dataset(
    supervised: pd.DataFrame,
    gap_length: int,
) -> Tuple[Path, Path]:

    output_dir = (
        OUTPUT_DIR
        / f"gap_{gap_length}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        output_dir
        / f"{CDP_METER}_supervised_gap_{gap_length}_v21.parquet"
    )

    metadata_path = (
        output_dir
        / "supervised_metadata_v21.json"
    )

    supervised.to_parquet(
        parquet_path,
        index=False,
    )

    metadata = {
        "feature_version": "v2.1",
        "gap_length": gap_length,
        "gap_lengths": GAP_LENGTHS,
        "context_left": CONTEXT_LEFT,
        "context_right": CONTEXT_RIGHT,
        "sliding_windows": False,
        "one_prediction_per_missing_lp": True,
        "ground_truth_leakage": False,
        "current_gap_feature_use": False,
        "v1_features": V1_FEATURES,
        "historical_features": HISTORICAL_FEATURES,
        "recent_context_features": RECENT_CONTEXT_FEATURES,
        "removed_context_statistics": REMOVED_CONTEXT_STATISTICS,
        "removed_trend_features": REMOVED_TREND_FEATURES,
        "predictor_feature_count": len(V21_FEATURES),
        "predictor_features": V21_FEATURES,
        "rows": len(supervised),
        "train_samples": int(
            (supervised["split"] == "train").sum()
        ),
        "validation_samples": int(
            (supervised["split"] == "validation").sum()
        ),
        "test_samples": int(
            (supervised["split"] == "test").sum()
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
            default=str,
        ),
        encoding="utf-8",
    )

    return parquet_path, metadata_path


# =============================================================================
# RELOAD VERIFICATION
# =============================================================================

def verify_saved_dataset(
    parquet_path: Path,
    expected_shape: Tuple[int, int],
    gap_length: int,
) -> None:

    if not parquet_path.exists():
        fail(
            f"Saved parquet not found:\n{parquet_path}"
        )

    reloaded = pd.read_parquet(
        parquet_path
    )

    if reloaded.shape != expected_shape:
        fail(
            f"Gap {gap_length}: reload shape mismatch. "
            f"Expected {expected_shape}, "
            f"found {reloaded.shape}."
        )

    if list(
        reloaded[V21_FEATURES].columns
    ) != V21_FEATURES:
        fail(
            f"Gap {gap_length}: feature order changed after reload."
        )


# =============================================================================
# PROCESS ONE GAP
# =============================================================================

def process_gap(
    gap_length: int,
) -> Dict:

    section(
        f"V2.1 FEATURE GENERATION — GAP {gap_length} LP"
    )

    dataframe = load_gap_dataframe(
        gap_length
    )

    metadata = load_gap_metadata(
        gap_length
    )

    print("Input dataframe:")
    print(f"    {gap_dataframe_path(gap_length)}")
    print(
        f"Rows                 : {len(dataframe):,}"
    )
    print(
        f"Columns              : {len(dataframe.columns)}"
    )

    validate_gap_dataframe(
        dataframe=dataframe,
        gap_length=gap_length,
    )

    print("Gap dataframe         : PASSED")

    validate_gap_metadata(
        metadata=metadata,
        dataframe=dataframe,
        gap_length=gap_length,
    )

    print("Gap metadata          : PASSED")

    # -------------------------------------------------------------------------
    # Create samples.
    # -------------------------------------------------------------------------

    supervised = create_supervised_dataset(
        dataframe=dataframe,
        metadata=metadata,
        gap_length=gap_length,
    )

    print(
        f"V2.1 samples created  : {len(supervised):,}"
    )

    # -------------------------------------------------------------------------
    # Validate.
    # -------------------------------------------------------------------------

    validate_supervised_dataset(
        supervised=supervised,
        metadata=metadata,
        gap_length=gap_length,
    )

    print(
        "V2.1 feature validation: PASSED"
    )

    validate_v1_preservation(
        dataframe=dataframe,
        supervised=supervised,
        gap_length=gap_length,
    )

    print(
        "V1 preservation       : PASSED"
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    parquet_path, metadata_path = save_dataset(
        supervised=supervised,
        gap_length=gap_length,
    )

    print("Saved dataframe:")
    print(f"    {parquet_path}")
    print(
        f"Shape                 : {supervised.shape}"
    )

    print("Saved metadata:")
    print(f"    {metadata_path}")

    # -------------------------------------------------------------------------
    # Reload.
    # -------------------------------------------------------------------------

    verify_saved_dataset(
        parquet_path=parquet_path,
        expected_shape=supervised.shape,
        gap_length=gap_length,
    )

    print(
        "Parquet reload        : PASSED"
    )

    return {
        "gap_length": gap_length,
        "rows": len(supervised),
        "train": int(
            (supervised["split"] == "train").sum()
        ),
        "validation": int(
            (supervised["split"] == "validation").sum()
        ),
        "test": int(
            (supervised["split"] == "test").sum()
        ),
        "features": len(V21_FEATURES),
        "path": str(parquet_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION — "
        "V2.1 SUPERVISED FEATURE LAYER"
    )
    print("=" * 80)

    print()
    print("V1 pipeline                 : FROZEN")
    print("V2 pipeline                 : FROZEN")
    print("V2.1 feature version        : v2.1")
    print("Gap lengths                 : 1, 6, 24, 48 LP")
    print("96 LP gap                   : REMOVED")
    print("Context                     : 96 LP left + 96 LP right")
    print("Sliding windows             : DISABLED")
    print("One prediction / missing LP : ENABLED")
    print("V1 features                 : 22")
    print("Historical features         : 3")
    print("Recent-context features     : 4")
    print("V2.1 predictor features     : 29")
    print("Context statistics          : REMOVED")
    print("Trend features              : REMOVED")
    print("Ground-truth leakage        : PROHIBITED")
    print("Current-gap feature use     : PROHIBITED")
    print("Random Forest training      : DISABLED")
    print("MLflow                      : NOT YET")
    print("Source modification        : DISABLED")

    print()
    print("V2.1 FEATURE SET")
    separator()

    print()
    print("V1 ORIGINAL FEATURES — 22")
    for i, feature in enumerate(
        V1_FEATURES,
        start=1,
    ):
        print(
            f"    {i:02d}. {feature}"
        )

    print()
    print("HISTORICAL FEATURES — 3")
    for i, feature in enumerate(
        HISTORICAL_FEATURES,
        start=1,
    ):
        print(
            f"    {i:02d}. {feature}"
        )

    print()
    print("RECENT CONTEXT FEATURES — 4")
    for i, feature in enumerate(
        RECENT_CONTEXT_FEATURES,
        start=1,
    ):
        print(
            f"    {i:02d}. {feature}"
        )

    print()
    print(
        f"TOTAL V2.1 FEATURES : {len(V21_FEATURES)}"
    )

    print()
    print("REMOVED FROM V2")
    for feature in REMOVED_CONTEXT_STATISTICS:
        print(
            f"    - {feature}"
        )

    for feature in REMOVED_TREND_FEATURES:
        print(
            f"    - {feature}"
        )

    validate_feature_definition()

    print()
    print(
        "Feature definition          : PASSED"
    )

    results: Dict[int, Dict] = {}

    for gap_length in GAP_LENGTHS:

        results[gap_length] = process_gap(
            gap_length=gap_length
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    section(
        "V2.1 SUPERVISED FEATURE LAYER SUMMARY"
    )

    summary_rows = []

    for gap_length in GAP_LENGTHS:

        result = results[gap_length]

        summary_rows.append(
            {
                "gap_length": gap_length,
                "total_samples": result["rows"],
                "train_samples": result["train"],
                "validation_samples": result["validation"],
                "test_samples": result["test"],
                "V1_features": len(V1_FEATURES),
                "historical_features": len(
                    HISTORICAL_FEATURES
                ),
                "recent_context_features": len(
                    RECENT_CONTEXT_FEATURES
                ),
                "V2.1_features": len(V21_FEATURES),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Save summary.
    # -------------------------------------------------------------------------

    summary_dir = (
        OUTPUT_DIR
        / "summary"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        summary_dir
        / "v21_feature_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "V2.1 SUPERVISED FEATURE LAYER COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "V1 datasets              : UNCHANGED"
    )
    print(
        "V2 datasets              : UNCHANGED"
    )
    print(
        "V2.1 datasets            : CREATED"
    )
    print(
        "Random Forest            : NOT TRAINED"
    )
    print(
        "MLflow                   : NOT ADDED"
    )
    print(
        "Source CSV               : UNCHANGED"
    )

    print()
    print("V2.1 feature count:")
    print(
        f"    V1 original          : {len(V1_FEATURES)}"
    )
    print(
        f"    Historical           : {len(HISTORICAL_FEATURES)}"
    )
    print(
        f"    Recent context       : {len(RECENT_CONTEXT_FEATURES)}"
    )
    print(
        f"    TOTAL                : {len(V21_FEATURES)}"
    )

    print()
    print("Output directory:")
    print(
        f"    {OUTPUT_DIR}"
    )

    print()
    print("Summary:")
    print(
        f"    {summary_path}"
    )


if __name__ == "__main__":
    main()