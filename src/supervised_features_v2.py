"""
RANDOM FOREST CDP IMPUTATION — V2 SUPERVISED FEATURE LAYER

V2 = V1 baseline + enhanced load-profile features.

V1 behavior and files remain unchanged.

V2 additions:
    1. Previous-day same-slot value
    2. Previous-week same-slot value
    3. Previous-week availability indicator
    4. Left-context statistics
    5. Right-context statistics
    6. Recent-context statistics
    7. Boundary values
    8. Context trends

Leakage rules:
    - ground_truth is NEVER used as a predictor
    - prediction is NEVER used as a predictor
    - current artificial gap values are NEVER used
    - future target values are NEVER used
    - V1 features remain unchanged
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(CONFIG["project_root"])

PROCESSED_DIR = (
    PROJECT_ROOT
    / CONFIG["outputs"]["processed_data_dir"]
)

GAPS_DIR = PROCESSED_DIR / "gaps"

V2_OUTPUT_DIR = (
    PROCESSED_DIR / "supervised_v2"
)

GAP_LENGTHS = [
    int(x)
    for x in CONFIG["gaps"]["lengths"]
]

CONTEXT_LEFT = int(
    CONFIG["gaps"]["context_left"]
)

CONTEXT_RIGHT = int(
    CONFIG["gaps"]["context_right"]
)

EXPECTED_INTERVAL_MINUTES = int(
    CONFIG["data"]["expected_interval_minutes"]
)

METER_ID = CONFIG["data"]["selected_meter"]

TARGET_DIRECTION = CONFIG["data"]["target_direction"]

TARGET_COLUMN = "target"

GROUND_TRUTH_COLUMN = "ground_truth"

FEATURE_VERSION = "v2"

RECENT_CONTEXT_LENGTH = 6

DAY_LAG = 48

WEEK_LAG = 336


# =============================================================================
# V1 FEATURES
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
# V2 NEW FEATURES
# =============================================================================

V2_FEATURES = [
    "target_previous_day_same_slot",
    "target_previous_week_same_slot",
    "previous_week_available",

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

    "left_recent_mean",
    "right_recent_mean",

    "left_last_value",
    "right_first_value",

    "left_slope",
    "right_slope",
]

assert len(V2_FEATURES) == 21


# =============================================================================
# LOGGING
# =============================================================================

def info(message: str = "") -> None:
    print(message)


def section(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# PATHS
# =============================================================================

def gap_dataframe_path(gap_length: int) -> Path:

    return (
        GAPS_DIR
        / f"gap_{gap_length}"
        / (
            f"{METER_ID}_{TARGET_DIRECTION}"
            f"_gap_{gap_length}.parquet"
        )
    )


def gap_metadata_path(gap_length: int) -> Path:

    return (
        GAPS_DIR
        / f"gap_{gap_length}"
        / "gap_metadata.csv"
    )


def v2_output_path(gap_length: int) -> Path:

    return (
        V2_OUTPUT_DIR
        / f"gap_{gap_length}"
        / (
            f"{METER_ID}_{TARGET_DIRECTION}"
            f"_supervised_gap_{gap_length}_v2.parquet"
        )
    )


def v2_metadata_path(gap_length: int) -> Path:

    return (
        V2_OUTPUT_DIR
        / f"gap_{gap_length}"
        / "supervised_features_v2_metadata.json"
    )


# =============================================================================
# GAP DATAFRAME VALIDATION
# =============================================================================

def validate_gap_dataframe(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:

    required = {
        "Time",
        TARGET_COLUMN,
        GROUND_TRUTH_COLUMN,
        "meter_id",
    }

    missing = sorted(
        required - set(dataframe.columns)
    )

    if missing:
        fail(
            f"Gap {gap_length}: missing required columns: "
            f"{missing}"
        )

    if len(dataframe) != 17520:
        fail(
            f"Gap {gap_length}: expected 17,520 rows; "
            f"found {len(dataframe)}."
        )

    time = pd.to_datetime(
        dataframe["Time"],
        errors="coerce",
    )

    if time.isna().any():
        fail(
            f"Gap {gap_length}: invalid timestamps."
        )

    if not time.is_unique:
        fail(
            f"Gap {gap_length}: timestamps are not unique."
        )

    if not time.is_monotonic_increasing:
        fail(
            f"Gap {gap_length}: timestamps are not chronological."
        )

    expected_delta = pd.Timedelta(
        minutes=EXPECTED_INTERVAL_MINUTES
    )

    actual_delta = time.diff().dropna()

    if not (actual_delta == expected_delta).all():
        fail(
            f"Gap {gap_length}: timestamp interval is not "
            f"{EXPECTED_INTERVAL_MINUTES} minutes."
        )

    if dataframe["meter_id"].nunique() != 1:
        fail(
            f"Gap {gap_length}: multiple meters found."
        )

    if dataframe["meter_id"].iloc[0] != METER_ID:
        fail(
            f"Gap {gap_length}: unexpected meter."
        )

    if dataframe[GROUND_TRUTH_COLUMN].isna().any():
        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )


# =============================================================================
# GAP METADATA VALIDATION
# =============================================================================

def validate_metadata(
    metadata: pd.DataFrame,
    gap_length: int,
) -> None:

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
        "left_context_length",
        "right_context_length",
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: metadata missing columns: "
            f"{missing}"
        )

    if metadata.empty:
        fail(
            f"Gap {gap_length}: metadata is empty."
        )

    if not (
        metadata["gap_length"] == gap_length
    ).all():
        fail(
            f"Gap {gap_length}: metadata gap length mismatch."
        )

    if not (
        metadata["left_context_length"] == CONTEXT_LEFT
    ).all():
        fail(
            f"Gap {gap_length}: left context length mismatch."
        )

    if not (
        metadata["right_context_length"] == CONTEXT_RIGHT
    ).all():
        fail(
            f"Gap {gap_length}: right context length mismatch."
        )

    if metadata["gap_id"].duplicated().any():
        fail(
            f"Gap {gap_length}: duplicate event IDs."
        )


# =============================================================================
# OBSERVATION HELPERS
# =============================================================================

def is_observed(
    dataframe: pd.DataFrame,
    index: int,
) -> bool:

    if index < 0 or index >= len(dataframe):
        return False

    value = dataframe.iloc[index][TARGET_COLUMN]

    if pd.isna(value):
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def observed_value(
    dataframe: pd.DataFrame,
    index: int,
) -> float:

    if not is_observed(
        dataframe,
        index,
    ):
        fail(
            f"Expected observed target at index {index}, "
            "but target is unavailable."
        )

    value = float(
        dataframe.iloc[index][TARGET_COLUMN]
    )

    if not math.isfinite(value):
        fail(
            f"Non-finite target at index {index}."
        )

    return value


# =============================================================================
# GAP MEMBERSHIP
# =============================================================================

def inside_gap(
    index: int,
    gap_start: int,
    gap_end: int,
) -> bool:

    return (
        gap_start
        <= index
        <= gap_end
    )


# =============================================================================
# CONTEXT INDEXES
# =============================================================================

def get_context_indexes(
    event: pd.Series,
    dataframe_length: int,
) -> Tuple[List[int], List[int]]:

    gap_start = int(
        event["gap_start_index"]
    )

    gap_end = int(
        event["gap_end_index"]
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

    if (
        left_end
        - left_start
        + 1
        != CONTEXT_LEFT
    ):
        fail(
            f"Event {event['gap_id']}: "
            "invalid left context length."
        )

    if (
        right_end
        - right_start
        + 1
        != CONTEXT_RIGHT
    ):
        fail(
            f"Event {event['gap_id']}: "
            "invalid right context length."
        )

    if left_end != gap_start - 1:
        fail(
            f"Event {event['gap_id']}: "
            "left context does not end immediately before gap."
        )

    if right_start != gap_end + 1:
        fail(
            f"Event {event['gap_id']}: "
            "right context does not start immediately after gap."
        )

    if left_start < 0:
        fail(
            f"Event {event['gap_id']}: "
            "left context outside dataframe."
        )

    if right_end >= dataframe_length:
        fail(
            f"Event {event['gap_id']}: "
            "right context outside dataframe."
        )

    left_indexes = list(
        range(
            left_start,
            left_end + 1,
        )
    )

    right_indexes = list(
        range(
            right_start,
            right_end + 1,
        )
    )

    return left_indexes, right_indexes


# =============================================================================
# CONTEXT VALIDATION
# =============================================================================

def validate_context(
    dataframe: pd.DataFrame,
    indexes: List[int],
    event: pd.Series,
    name: str,
) -> None:

    gap_start = int(
        event["gap_start_index"]
    )

    gap_end = int(
        event["gap_end_index"]
    )

    for index in indexes:

        if inside_gap(
            index,
            gap_start,
            gap_end,
        ):
            fail(
                f"Event {event['gap_id']}: "
                f"{name} enters artificial gap at index {index}."
            )

        if not is_observed(
            dataframe,
            index,
        ):
            fail(
                f"Event {event['gap_id']}: "
                f"{name} contains unavailable value at index {index}."
            )


# =============================================================================
# CONTEXT STATISTICS
# =============================================================================

def context_statistics(
    values: np.ndarray,
) -> Dict[str, float]:

    if len(values) == 0:
        fail(
            "Empty context."
        )

    if not np.isfinite(values).all():
        fail(
            "Context contains non-finite values."
        )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "std": float(
            np.std(values)
        ),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
    }


# =============================================================================
# CONTEXT SLOPE
# =============================================================================

def context_slope(
    values: np.ndarray,
) -> float:

    x = np.arange(
        len(values),
        dtype=float,
    )

    y = np.asarray(
        values,
        dtype=float,
    )

    if len(y) < 2:
        fail(
            "At least two observations are required for slope."
        )

    if not np.isfinite(y).all():
        fail(
            "Cannot calculate slope from non-finite values."
        )

    slope = float(
        np.polyfit(
            x,
            y,
            1,
        )[0]
    )

    if not math.isfinite(slope):
        fail(
            "Calculated slope is not finite."
        )

    return slope


# =============================================================================
# HISTORICAL FEATURE
# =============================================================================

def historical_value(
    dataframe: pd.DataFrame,
    target_index: int,
    lag: int,
    gap_start: int,
    gap_end: int,
) -> Tuple[float, bool]:
    """
    Find an observed historical value.

    Returns:
        value
        availability flag

    For the weekly feature:

        If target_index - 336 exists:
            use it
            availability = 1

        If target_index - 336 does not exist:
            use previous-day same-slot value
            availability = 0

    This prevents dataset-boundary failures while explicitly
    telling the model that the weekly historical observation
    was unavailable.

    No ground truth is accessed.
    """

    candidate = (
        target_index - lag
    )

    # -------------------------------------------------------------------------
    # Exact historical value exists.
    # -------------------------------------------------------------------------

    if (
        candidate >= 0
        and candidate < len(dataframe)
        and not inside_gap(
            candidate,
            gap_start,
            gap_end,
        )
        and is_observed(
            dataframe,
            candidate,
        )
    ):

        return (
            observed_value(
                dataframe,
                candidate,
            ),
            True,
        )

    # -------------------------------------------------------------------------
    # Weekly historical value unavailable.
    #
    # Only the weekly feature gets this fallback.
    # -------------------------------------------------------------------------

    if lag == WEEK_LAG:

        fallback = (
            target_index - DAY_LAG
        )

        if (
            fallback >= 0
            and fallback < len(dataframe)
            and not inside_gap(
                fallback,
                gap_start,
                gap_end,
            )
            and is_observed(
                dataframe,
                fallback,
            )
        ):

            return (
                observed_value(
                    dataframe,
                    fallback,
                ),
                False,
            )

    # -------------------------------------------------------------------------
    # Previous-day historical value should exist for all selected
    # events. If not, fail rather than silently inventing data.
    # -------------------------------------------------------------------------

    fail(
        "No valid observed historical value found. "
        f"target_index={target_index}, "
        f"lag={lag}, "
        f"gap={gap_start}:{gap_end}"
    )


# =============================================================================
# EVENT-LEVEL V2 FEATURES
# =============================================================================

def create_event_features(
    dataframe: pd.DataFrame,
    event: pd.Series,
) -> Dict[str, float]:

    left_indexes, right_indexes = (
        get_context_indexes(
            event,
            len(dataframe),
        )
    )

    validate_context(
        dataframe,
        left_indexes,
        event,
        "left context",
    )

    validate_context(
        dataframe,
        right_indexes,
        event,
        "right context",
    )

    left_values = np.asarray(
        [
            observed_value(
                dataframe,
                index,
            )
            for index in left_indexes
        ],
        dtype=float,
    )

    right_values = np.asarray(
        [
            observed_value(
                dataframe,
                index,
            )
            for index in right_indexes
        ],
        dtype=float,
    )

    left_stats = context_statistics(
        left_values
    )

    right_stats = context_statistics(
        right_values
    )

    features = {}

    # Left statistics

    features["left_mean"] = (
        left_stats["mean"]
    )

    features["left_median"] = (
        left_stats["median"]
    )

    features["left_std"] = (
        left_stats["std"]
    )

    features["left_min"] = (
        left_stats["min"]
    )

    features["left_max"] = (
        left_stats["max"]
    )

    features["left_range"] = (
        left_stats["range"]
    )

    # Right statistics

    features["right_mean"] = (
        right_stats["mean"]
    )

    features["right_median"] = (
        right_stats["median"]
    )

    features["right_std"] = (
        right_stats["std"]
    )

    features["right_min"] = (
        right_stats["min"]
    )

    features["right_max"] = (
        right_stats["max"]
    )

    features["right_range"] = (
        right_stats["range"]
    )

    # Recent context

    left_recent = left_values[
        -RECENT_CONTEXT_LENGTH:
    ]

    right_recent = right_values[
        :RECENT_CONTEXT_LENGTH
    ]

    if len(left_recent) != RECENT_CONTEXT_LENGTH:
        fail(
            f"Event {event['gap_id']}: "
            "insufficient recent left context."
        )

    if len(right_recent) != RECENT_CONTEXT_LENGTH:
        fail(
            f"Event {event['gap_id']}: "
            "insufficient recent right context."
        )

    features[
        "left_recent_mean"
    ] = float(
        np.mean(left_recent)
    )

    features[
        "right_recent_mean"
    ] = float(
        np.mean(right_recent)
    )

    # Boundary values

    features[
        "left_last_value"
    ] = observed_value(
        dataframe,
        left_indexes[-1],
    )

    features[
        "right_first_value"
    ] = observed_value(
        dataframe,
        right_indexes[0],
    )

    # Trends

    features[
        "left_slope"
    ] = context_slope(
        left_values
    )

    features[
        "right_slope"
    ] = context_slope(
        right_values
    )

    return features


# =============================================================================
# CREATE SUPERVISED SAMPLES
# =============================================================================

def create_samples(
    dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    samples: List[Dict] = []

    for _, event in metadata.iterrows():

        gap_id = int(
            event["gap_id"]
        )

        split = str(
            event["split"]
        )

        gap_start = int(
            event["gap_start_index"]
        )

        gap_end = int(
            event["gap_end_index"]
        )

        if (
            gap_end
            - gap_start
            + 1
            != gap_length
        ):
            fail(
                f"Event {gap_id}: gap length mismatch."
            )

        # Event-level features are calculated only from
        # observed left/right context.
        event_features = create_event_features(
            dataframe,
            event,
        )

        for position in range(
            gap_length
        ):

            target_index = (
                gap_start + position
            )

            row = dataframe.iloc[
                target_index
            ]

            # Target must be masked.

            if pd.notna(
                row[TARGET_COLUMN]
            ):
                fail(
                    f"Event {gap_id}: "
                    f"target index {target_index} "
                    "is not masked."
                )

            # Ground truth is target only.

            ground_truth = row[
                GROUND_TRUTH_COLUMN
            ]

            if pd.isna(
                ground_truth
            ):
                fail(
                    f"Event {gap_id}: "
                    f"ground truth missing at "
                    f"{target_index}."
                )

            sample = {
                "event_id": gap_id,

                "gap_length": gap_length,

                "split": split,

                "target_index": target_index,

                "gap_position": position + 1,

                "gap_position_fraction": (
                    (position + 1)
                    / gap_length
                ),

                "Time": row["Time"],

                "ground_truth": float(
                    ground_truth
                ),

                "feature_version": FEATURE_VERSION,
            }

            # -----------------------------------------------------------------
            # V1 deterministic features.
            # -----------------------------------------------------------------

            for feature in V1_FEATURES:

                if feature not in dataframe.columns:
                    fail(
                        f"V1 feature missing: {feature}"
                    )

                value = row[feature]

                if pd.isna(value):
                    fail(
                        f"V1 feature {feature} is NaN "
                        f"at index {target_index}."
                    )

                value = float(value)

                if not math.isfinite(value):
                    fail(
                        f"V1 feature {feature} is non-finite "
                        f"at index {target_index}."
                    )

                sample[feature] = value

            # -----------------------------------------------------------------
            # Historical features.
            # -----------------------------------------------------------------

            previous_day_value, previous_day_available = (
                historical_value(
                    dataframe=dataframe,
                    target_index=target_index,
                    lag=DAY_LAG,
                    gap_start=gap_start,
                    gap_end=gap_end,
                )
            )

            previous_week_value, previous_week_available = (
                historical_value(
                    dataframe=dataframe,
                    target_index=target_index,
                    lag=WEEK_LAG,
                    gap_start=gap_start,
                    gap_end=gap_end,
                )
            )

            sample[
                "target_previous_day_same_slot"
            ] = previous_day_value

            sample[
                "target_previous_week_same_slot"
            ] = previous_week_value

            sample[
                "previous_week_available"
            ] = int(
                previous_week_available
            )

            # -----------------------------------------------------------------
            # Event-level context features.
            # -----------------------------------------------------------------

            sample.update(
                event_features
            )

            samples.append(
                sample
            )

    if not samples:
        fail(
            f"Gap {gap_length}: no samples created."
        )

    return pd.DataFrame(
        samples
    )


# =============================================================================
# V1 COMPARISON
# =============================================================================

def validate_against_v1(
    v2: pd.DataFrame,
    gap_length: int,
) -> None:

    v1_path = (
        PROCESSED_DIR
        / "supervised"
        / f"gap_{gap_length}"
        / (
            f"{METER_ID}_{TARGET_DIRECTION}"
            f"_supervised_gap_{gap_length}.parquet"
        )
    )

    if not v1_path.exists():
        fail(
            f"V1 supervised dataset not found:\n{v1_path}"
        )

    v1 = pd.read_parquet(
        v1_path
    )

    if len(v1) != len(v2):
        fail(
            f"Gap {gap_length}: "
            f"V1 rows={len(v1)}, "
            f"V2 rows={len(v2)}."
        )

    identity = [
        "event_id",
        "gap_length",
        "split",
        "target_index",
        "gap_position",
        "Time",
        "ground_truth",
    ]

    for column in identity:

        if column not in v1.columns:
            fail(
                f"V1 missing column: {column}"
            )

        if column not in v2.columns:
            fail(
                f"V2 missing column: {column}"
            )

        if column == "Time":

            a = pd.to_datetime(
                v1[column]
            ).reset_index(
                drop=True
            )

            b = pd.to_datetime(
                v2[column]
            ).reset_index(
                drop=True
            )

            if not a.equals(b):
                fail(
                    f"Gap {gap_length}: "
                    "Time changed between V1/V2."
                )

        elif column == "ground_truth":

            a = v1[column].to_numpy(
                dtype=float
            )

            b = v2[column].to_numpy(
                dtype=float
            )

            if not np.allclose(
                a,
                b,
                equal_nan=True,
            ):
                fail(
                    f"Gap {gap_length}: "
                    "ground_truth changed."
                )

        else:

            a = v1[column].reset_index(
                drop=True
            )

            b = v2[column].reset_index(
                drop=True
            )

            if not a.equals(b):
                fail(
                    f"Gap {gap_length}: "
                    f"{column} changed."
                )

    for feature in V1_FEATURES:

        if feature not in v1.columns:
            fail(
                f"V1 feature missing: {feature}"
            )

        if feature not in v2.columns:
            fail(
                f"V2 feature missing: {feature}"
            )

        a = v1[feature].to_numpy(
            dtype=float
        )

        b = v2[feature].to_numpy(
            dtype=float
        )

        if not np.allclose(
            a,
            b,
            equal_nan=True,
        ):
            fail(
                f"Gap {gap_length}: "
                f"V1 feature changed: {feature}"
            )


# =============================================================================
# V2 VALIDATION
# =============================================================================

def validate_v2(
    dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> None:

    expected_rows = (
        len(metadata)
        * gap_length
    )

    if len(dataframe) != expected_rows:
        fail(
            f"Gap {gap_length}: expected "
            f"{expected_rows} rows; "
            f"found {len(dataframe)}."
        )

    missing = [
        feature
        for feature in V2_FEATURES
        if feature not in dataframe.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: missing V2 features: "
            f"{missing}"
        )

    # -------------------------------------------------------------------------
    # All model features must be finite.
    # -------------------------------------------------------------------------

    for feature in V1_FEATURES + V2_FEATURES:

        values = dataframe[
            feature
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            fail(
                f"Gap {gap_length}: "
                f"non-finite values in {feature}."
            )

    # -------------------------------------------------------------------------
    # Availability flag must be binary.
    # -------------------------------------------------------------------------

    available_values = set(
        dataframe[
            "previous_week_available"
        ].astype(int).unique()
    )

    if not available_values.issubset(
        {0, 1}
    ):
        fail(
            f"Gap {gap_length}: "
            "previous_week_available must be 0/1."
        )

    # -------------------------------------------------------------------------
    # One sample per missing LP.
    # -------------------------------------------------------------------------

    counts = (
        dataframe
        .groupby("event_id")
        .size()
    )

    for event_id, count in counts.items():

        if count != gap_length:
            fail(
                f"Gap {gap_length}: event {event_id} "
                f"has {count} samples."
            )

    # -------------------------------------------------------------------------
    # Event IDs match metadata.
    # -------------------------------------------------------------------------

    expected_events = set(
        metadata["gap_id"].astype(int)
    )

    actual_events = set(
        dataframe["event_id"].astype(int)
    )

    if expected_events != actual_events:
        fail(
            f"Gap {gap_length}: event IDs differ."
        )

    # -------------------------------------------------------------------------
    # Split labels match metadata.
    # -------------------------------------------------------------------------

    split_map = dict(
        zip(
            metadata["gap_id"].astype(int),
            metadata["split"].astype(str),
        )
    )

    for _, row in dataframe.iterrows():

        event_id = int(
            row["event_id"]
        )

        if str(
            row["split"]
        ) != split_map[event_id]:

            fail(
                f"Gap {gap_length}: split mismatch "
                f"for event {event_id}."
            )

    # -------------------------------------------------------------------------
    # Leakage checks.
    # -------------------------------------------------------------------------

    forbidden = {
        "ground_truth",
        "prediction",
        "target",
    }

    feature_set = set(
        V1_FEATURES + V2_FEATURES
    )

    leakage = (
        feature_set
        & forbidden
    )

    if leakage:
        fail(
            f"Gap {gap_length}: "
            f"forbidden features detected: {leakage}"
        )


# =============================================================================
# SAVE METADATA
# =============================================================================

def save_metadata(
    dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> None:

    output_path = v2_metadata_path(
        gap_length
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {

        "feature_version": FEATURE_VERSION,

        "meter_id": METER_ID,

        "direction": TARGET_DIRECTION,

        "gap_length": gap_length,

        "context_left": CONTEXT_LEFT,

        "context_right": CONTEXT_RIGHT,

        "recent_context_length":
            RECENT_CONTEXT_LENGTH,

        "day_lag_lp": DAY_LAG,

        "week_lag_lp": WEEK_LAG,

        "v1_feature_count":
            len(V1_FEATURES),

        "v2_new_feature_count":
            len(V2_FEATURES),

        "total_feature_count":
            len(V1_FEATURES)
            + len(V2_FEATURES),

        "v1_features": V1_FEATURES,

        "v2_features": V2_FEATURES,

        "train_events": int(
            (
                metadata["split"]
                == "train"
            ).sum()
        ),

        "validation_events": int(
            (
                metadata["split"]
                == "validation"
            ).sum()
        ),

        "test_events": int(
            (
                metadata["split"]
                == "test"
            ).sum()
        ),

        "supervised_rows": len(
            dataframe
        ),

        "one_prediction_per_missing_lp": True,

        "sliding_windows": False,

        "gap_96_removed": True,

        "ground_truth_used_as_feature": False,

        "current_gap_values_used": False,

        "previous_week_fallback": (
            "previous_day_same_slot"
        ),

        "previous_week_availability_indicator": True,

        "source_csv_modified": False,

        "mlflow_enabled": False,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
        )


# =============================================================================
# PROCESS ONE GAP
# =============================================================================

def process_gap(
    gap_length: int,
) -> pd.DataFrame:

    section(
        f"V2 FEATURE GENERATION — GAP {gap_length} LP"
    )

    parquet_path = gap_dataframe_path(
        gap_length
    )

    metadata_path = gap_metadata_path(
        gap_length
    )

    if not parquet_path.exists():
        fail(
            f"V1 gap Parquet not found:\n{parquet_path}"
        )

    if not metadata_path.exists():
        fail(
            f"V1 gap metadata not found:\n{metadata_path}"
        )

    info(
        f"Input dataframe:\n    {parquet_path}"
    )

    dataframe = pd.read_parquet(
        parquet_path
    )

    metadata = pd.read_csv(
        metadata_path
    )

    info(
        f"Rows                 : {len(dataframe):,}"
    )

    info(
        f"Columns              : {len(dataframe.columns)}"
    )

    validate_gap_dataframe(
        dataframe,
        gap_length,
    )

    validate_metadata(
        metadata,
        gap_length,
    )

    info(
        "Gap dataframe         : PASSED"
    )

    info(
        "Gap metadata          : PASSED"
    )

    supervised = create_samples(
        dataframe=dataframe,
        metadata=metadata,
        gap_length=gap_length,
    )

    info(
        f"V2 samples created    : {len(supervised):,}"
    )

    validate_v2(
        supervised,
        metadata,
        gap_length,
    )

    info(
        "V2 feature validation : PASSED"
    )

    validate_against_v1(
        supervised,
        gap_length,
    )

    info(
        "V1 preservation       : PASSED"
    )

    output_path = v2_output_path(
        gap_length
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    supervised.to_parquet(
        output_path,
        index=False,
    )

    save_metadata(
        dataframe=supervised,
        metadata=metadata,
        gap_length=gap_length,
    )

    reloaded = pd.read_parquet(
        output_path
    )

    if not reloaded.equals(
        supervised
    ):
        fail(
            f"Gap {gap_length}: "
            "saved Parquet differs after reload."
        )

    info(
        f"Saved dataframe:\n    {output_path}"
    )

    info(
        f"Shape                 : {reloaded.shape}"
    )

    return supervised


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    results: Dict[int, pd.DataFrame],
) -> None:

    section(
        "V2 FEATURE LAYER SUMMARY"
    )

    for gap_length, dataframe in results.items():

        train = int(
            (
                dataframe["split"]
                == "train"
            ).sum()
        )

        validation = int(
            (
                dataframe["split"]
                == "validation"
            ).sum()
        )

        test = int(
            (
                dataframe["split"]
                == "test"
            ).sum()
        )

        info(
            f"Gap {gap_length:>2} LP"
        )

        info(
            f"    Total samples      : {len(dataframe):,}"
        )

        info(
            f"    Train samples      : {train:,}"
        )

        info(
            f"    Validation samples : {validation:,}"
        )

        info(
            f"    Test samples       : {test:,}"
        )

        info(
            f"    V1 features        : {len(V1_FEATURES)}"
        )

        info(
            f"    New V2 features    : {len(V2_FEATURES)}"
        )

        info(
            f"    Total V2 features  : "
            f"{len(V1_FEATURES) + len(V2_FEATURES)}"
        )

    info()
    info("NEW V2 FEATURES:")

    for number, feature in enumerate(
        V2_FEATURES,
        start=1,
    ):

        info(
            f"    {number:02d}. {feature}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)

    print(
        "RANDOM FOREST CDP IMPUTATION — "
        "V2 SUPERVISED FEATURE LAYER"
    )

    print("=" * 80)

    info()
    info(
        "V1 pipeline                 : UNCHANGED"
    )
    info(
        "V2 feature version          : v2"
    )
    info(
        "Gap lengths                 : 1, 6, 24, 48 LP"
    )
    info(
        "96 LP gap                   : REMOVED"
    )
    info(
        "Context                     : 96 LP left + 96 LP right"
    )
    info(
        "Sliding windows             : DISABLED"
    )
    info(
        "One prediction / missing LP : ENABLED"
    )
    info(
        "Historical features         : ENABLED"
    )
    info(
        "Context statistics          : ENABLED"
    )
    info(
        "Context trend features      : ENABLED"
    )
    info(
        "Previous-week fallback      : PREVIOUS-DAY SAME SLOT"
    )
    info(
        "Previous-week availability  : ENABLED"
    )
    info(
        "Ground-truth leakage        : PROHIBITED"
    )
    info(
        "Current-gap feature use     : PROHIBITED"
    )
    info(
        "Random Forest training      : DISABLED"
    )
    info(
        "MLflow                      : NOT YET"
    )
    info(
        "Source modification        : DISABLED"
    )

    results = {}

    for gap_length in GAP_LENGTHS:

        results[gap_length] = process_gap(
            gap_length
        )

    print_summary(
        results
    )

    print()
    print("=" * 80)
    print(
        "V2 SUPERVISED FEATURE LAYER COMPLETE"
    )
    print("=" * 80)

    info()
    info(
        "V1 datasets              : UNCHANGED"
    )
    info(
        "V1 gap Parquets          : UNCHANGED"
    )
    info(
        "V2 datasets              : CREATED"
    )
    info(
        "Random Forest            : NOT TRAINED"
    )
    info(
        "MLflow                   : NOT ADDED"
    )
    info(
        "Source CSV               : UNCHANGED"
    )

    info()
    info(
        f"Output directory:\n    {V2_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()