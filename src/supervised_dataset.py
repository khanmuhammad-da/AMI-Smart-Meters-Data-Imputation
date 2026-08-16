# =============================================================================
# src/supervised_dataset.py
# =============================================================================
"""
Random Forest CDP Load Profile Imputation
------------------------------------------

SUPERVISED DATASET STAGE

Converts fixed artificial-gap events into supervised ML samples.

Project design
--------------
1. One meter only.
2. One direction only.
3. Interior gaps only.
4. Gap lengths:
       1 LP
       6 LP
       24 LP
       48 LP
5. 96 LP gap REMOVED.
6. Fixed event windows.
7. No sliding windows.
8. 96 observed LP context on the left.
9. 96 observed LP context on the right.
10. One supervised sample for EACH missing LP.
11. Every missing LP within the same event uses the SAME
    observed left/right context.
12. No target-derived lag/lead features.
13. No normalization.
14. No model training.
15. No source-data modification.

Critical leakage rule
---------------------
For a gap:

    gap_start ... gap_end

the right context MUST begin at:

    gap_end + 1

and NOT at:

    target_index + 1

This is essential for multi-LP gaps.

Example: 6-LP gap

    LEFT CONTEXT        GAP              RIGHT CONTEXT
    96 observed LPs     6 missing LPs    96 observed LPs

    L96 ... L1          G1 ... G6        R1 ... R96

Every supervised sample uses:

    same L96 ... L1
    same R1 ... R96

while the target changes:

    sample 1 -> G1
    sample 2 -> G2
    ...
    sample 6 -> G6

This guarantees that one missing LP is never used to predict another
missing LP in the same event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# =============================================================================
# LOGGING
# =============================================================================

def section(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def info(message: str) -> None:
    print(message)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> dict:
    """Load project configuration."""

    if not CONFIG_PATH.exists():
        fail(
            "Configuration file not found:\n"
            f"    {CONFIG_PATH}"
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        fail(
            "config.yaml did not load as a dictionary."
        )

    return config


CONFIG = load_config()


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_config(
    config: dict,
    *keys: str,
    default=None,
):
    """Retrieve nested configuration value."""

    value = config

    try:
        for key in keys:
            value = value[key]

        return value

    except (
        KeyError,
        TypeError,
    ):
        return default


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_CONFIG = CONFIG["data"]
GAP_CONFIG = CONFIG["gaps"]
SUPERVISED_CONFIG = CONFIG["supervised"]
OUTPUT_CONFIG = CONFIG["outputs"]


GAP_LENGTHS = [
    int(x)
    for x in GAP_CONFIG["lengths"]
]

CONTEXT_LEFT = int(
    GAP_CONFIG["context_left"]
)

CONTEXT_RIGHT = int(
    GAP_CONFIG["context_right"]
)

SUPERVISED_DIR = (
    PROJECT_ROOT
    / OUTPUT_CONFIG["supervised_data_dir"]
)


# =============================================================================
# IMPORTANT FEATURE DEFINITION
# =============================================================================
#
# These are the deterministic features created by feature_engineering.py.
#
# They are safe because they are derived only from timestamp/calendar
# information and not from the target.
# =============================================================================

TIME_FEATURES = [
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
# CONTEXT FEATURE OFFSETS
# =============================================================================
#
# The project has 96 LP physical context on each side.
#
# To avoid creating an unnecessarily huge RF input at this stage, we retain
# the context offsets currently used by the project:
#
#   1, 2, 24
#
# This produces:
#
#   3 left target features
#   3 right target features
#
# The complete 96-LP physical context is still validated and preserved as
# the event boundary. These are the actual predictor features used by the
# current supervised dataset.
#
# If later we decide to use all 96 + 96 target values, this list can be
# expanded deliberately and evaluated as a separate feature experiment.
# =============================================================================

CONTEXT_OFFSETS = [
    1,
    2,
    24,
]

LEFT_OFFSETS = CONTEXT_OFFSETS
RIGHT_OFFSETS = CONTEXT_OFFSETS


# =============================================================================
# EXPECTED GAP FILE
# =============================================================================

def get_gap_dataframe_path(
    gap_length: int,
) -> Path:
    """
    Return the artificial-gap parquet file for one gap length.
    """

    meter_id = DATA_CONFIG[
        "selected_meter"
    ]

    direction = DATA_CONFIG[
        "target_direction"
    ]

    filename = (
        f"{meter_id}_{direction}"
        f"_gap_{gap_length}.parquet"
    )

    return (
        PROJECT_ROOT
        / OUTPUT_CONFIG["processed_data_dir"]
        / "gaps"
        / f"gap_{gap_length}"
        / filename
    )


# =============================================================================
# EXPECTED METADATA FILE
# =============================================================================

def get_gap_metadata_path(
    gap_length: int,
) -> Path:
    """Return gap-event metadata path."""

    return (
        PROJECT_ROOT
        / OUTPUT_CONFIG["processed_data_dir"]
        / "gaps"
        / f"gap_{gap_length}"
        / "gap_metadata.csv"
    )


# =============================================================================
# OUTPUT PATHS
# =============================================================================

def get_output_dataframe_path(
    gap_length: int,
) -> Path:

    meter_id = DATA_CONFIG[
        "selected_meter"
    ]

    direction = DATA_CONFIG[
        "target_direction"
    ]

    output_dir = (
        SUPERVISED_DIR
        / f"gap_{gap_length}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_dir
        / (
            f"{meter_id}_{direction}"
            f"_supervised_gap_{gap_length}.parquet"
        )
    )


def get_output_metadata_path(
    gap_length: int,
) -> Path:

    output_dir = (
        SUPERVISED_DIR
        / f"gap_{gap_length}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_dir
        / "supervised_metadata.json"
    )


# =============================================================================
# GAP DATAFRAME LOADING
# =============================================================================

def load_gap_dataframe(
    gap_length: int,
) -> pd.DataFrame:

    dataframe_path = get_gap_dataframe_path(
        gap_length
    )

    if not dataframe_path.exists():
        fail(
            f"Gap {gap_length} LP dataframe not found:\n"
            f"    {dataframe_path}"
        )

    info(
        f"Input dataframe:\n"
        f"    {dataframe_path}"
    )

    dataframe = pd.read_parquet(
        dataframe_path
    )

    info(
        f"Rows                 : {len(dataframe):,}"
    )

    info(
        f"Columns              : {len(dataframe.columns):,}"
    )

    return dataframe


# =============================================================================
# GAP METADATA LOADING
# =============================================================================

def load_gap_metadata(
    gap_length: int,
) -> pd.DataFrame:

    metadata_path = get_gap_metadata_path(
        gap_length
    )

    if not metadata_path.exists():
        fail(
            f"Gap {gap_length} LP metadata not found:\n"
            f"    {metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    if metadata.empty:
        fail(
            f"Gap {gap_length}: metadata is empty."
        )

    return metadata


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_gap_input(
    dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> None:

    section(
        f"GAP {gap_length} INPUT VALIDATION"
    )

    # -------------------------------------------------------------------------
    # Required base columns
    # -------------------------------------------------------------------------

    required_columns = [
        "Time",
        "target",
        "ground_truth",
        "is_gap",
    ]

    required_columns.extend(
        TIME_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: missing required columns:\n"
            + "\n".join(
                f"    {column}"
                for column in missing
            )
        )

    info(
        "Required columns       : PASSED"
    )

    # -------------------------------------------------------------------------
    # Rows
    # -------------------------------------------------------------------------

    if len(dataframe) == 0:
        fail(
            f"Gap {gap_length}: dataframe is empty."
        )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):
        fail(
            f"Gap {gap_length}: Time is not datetime."
        )

    if dataframe["Time"].isna().any():
        fail(
            f"Gap {gap_length}: Time contains NaN."
        )

    if not dataframe["Time"].is_unique:
        fail(
            f"Gap {gap_length}: timestamps are not unique."
        )

    if not dataframe["Time"].is_monotonic_increasing:
        fail(
            f"Gap {gap_length}: timestamps are not chronological."
        )

    info(
        "Timestamp validation   : PASSED"
    )

    # -------------------------------------------------------------------------
    # 30-minute continuity
    # -------------------------------------------------------------------------

    deltas = (
        dataframe["Time"]
        .diff()
        .dropna()
    )

    expected_delta = pd.Timedelta(
        minutes=int(
            DATA_CONFIG[
                "expected_interval_minutes"
            ]
        )
    )

    if not (deltas == expected_delta).all():
        fail(
            f"Gap {gap_length}: timestamp continuity "
            "is not exactly 30 minutes."
        )

    info(
        "30-minute continuity   : PASSED"
    )

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    if dataframe["ground_truth"].isna().any():
        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    info(
        "Ground truth           : PASSED"
    )

    # -------------------------------------------------------------------------
    # Gap masking
    # -------------------------------------------------------------------------

    gap_count = int(
        dataframe["is_gap"].sum()
    )

    expected_gap_count = (
        len(metadata)
        * gap_length
    )

    actual_missing = int(
        dataframe["target"].isna().sum()
    )

    if actual_missing != expected_gap_count:
        fail(
            f"Gap {gap_length}: incorrect masked target count.\n"
            f"Expected: {expected_gap_count}\n"
            f"Actual  : {actual_missing}"
        )

    if gap_count != expected_gap_count:
        fail(
            f"Gap {gap_length}: incorrect is_gap count.\n"
            f"Expected: {expected_gap_count}\n"
            f"Actual  : {gap_count}"
        )

    info(
        "Gap masking            : PASSED"
    )

    # -------------------------------------------------------------------------
    # Metadata split counts
    # -------------------------------------------------------------------------

    train_events = int(
        (
            metadata["split"]
            == "train"
        ).sum()
    )

    validation_events = int(
        (
            metadata["split"]
            == "validation"
        ).sum()
    )

    test_events = int(
        (
            metadata["split"]
            == "test"
        ).sum()
    )

    info(
        f"Train events           : {train_events}"
    )

    info(
        f"Validation events      : {validation_events}"
    )

    info(
        f"Test events            : {test_events}"
    )


# =============================================================================
# EVENT METADATA VALIDATION
# =============================================================================

def validate_event_metadata(
    event: pd.Series,
    dataframe_length: int,
    gap_length: int,
) -> None:

    required = [
        "gap_id",
        "split",
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
        if column not in event.index
    ]

    if missing:
        fail(
            f"Gap {gap_length}, event metadata missing:\n"
            + "\n".join(
                f"    {column}"
                for column in missing
            )
        )

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

    # -------------------------------------------------------------------------
    # Gap length
    # -------------------------------------------------------------------------

    if (
        gap_end
        - gap_start
        + 1
        != gap_length
    ):
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "gap index range does not match gap length."
        )

    # -------------------------------------------------------------------------
    # Context lengths
    # -------------------------------------------------------------------------

    if (
        left_end
        - left_start
        + 1
        != CONTEXT_LEFT
    ):
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "left context length is incorrect."
        )

    if (
        right_end
        - right_start
        + 1
        != CONTEXT_RIGHT
    ):
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "right context length is incorrect."
        )

    # -------------------------------------------------------------------------
    # Context must not overlap gap
    # -------------------------------------------------------------------------

    if left_end >= gap_start:
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "left context enters missing gap."
        )

    if right_start <= gap_end:
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "right context enters missing gap."
        )

    # -------------------------------------------------------------------------
    # Context must be immediately adjacent
    # -------------------------------------------------------------------------

    if left_end != gap_start - 1:
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "left context is not immediately before gap."
        )

    if right_start != gap_end + 1:
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "right context is not immediately after gap."
        )

    # -------------------------------------------------------------------------
    # Dataset boundaries
    # -------------------------------------------------------------------------

    if left_start < 0:
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "left context extends before dataframe."
        )

    if right_end >= dataframe_length:
        fail(
            f"Gap {gap_length}, event "
            f"{event['gap_id']}: "
            "right context extends beyond dataframe."
        )


# =============================================================================
# FIXED EVENT CONTEXT FEATURES
# =============================================================================

def create_context_features(
    dataframe: pd.DataFrame,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> Dict[str, float]:
    """
    Create target-context predictors using the WHOLE EVENT boundaries.

    This function deliberately does NOT use target_index.

    Therefore, all missing LPs in one event receive the same observed
    left/right context.

    Example for a 6-LP gap:

        left context:
            L96 ... L2 L1

        missing:
            G1 G2 G3 G4 G5 G6

        right context:
            R1 R2 ... R95 R96

    Every G1...G6 gets the same context.
    """

    features: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # LEFT CONTEXT
    #
    # target_left_1 = closest observed LP before gap
    # target_left_2 = second closest
    # target_left_24 = 24th closest
    # -------------------------------------------------------------------------

    for offset in LEFT_OFFSETS:

        index = (
            left_end
            - (offset - 1)
        )

        if (
            index < left_start
            or index > left_end
        ):
            fail(
                f"Left context offset {offset} "
                "falls outside fixed context."
            )

        value = dataframe.iloc[
            index
        ]["target"]

        if pd.isna(value):
            fail(
                f"Left context contains NaN at "
                f"index {index}."
            )

        features[
            f"target_left_{offset}"
        ] = float(value)

    # -------------------------------------------------------------------------
    # RIGHT CONTEXT
    #
    # target_right_1 = first observed LP after COMPLETE gap
    # target_right_2 = second observed LP after COMPLETE gap
    # target_right_24 = 24th observed LP after COMPLETE gap
    # -------------------------------------------------------------------------

    for offset in RIGHT_OFFSETS:

        index = (
            right_start
            + (offset - 1)
        )

        if (
            index < right_start
            or index > right_end
        ):
            fail(
                f"Right context offset {offset} "
                "falls outside fixed context."
            )

        value = dataframe.iloc[
            index
        ]["target"]

        if pd.isna(value):
            fail(
                f"Right context contains NaN at "
                f"index {index}."
            )

        features[
            f"target_right_{offset}"
        ] = float(value)

    return features


# =============================================================================
# SINGLE EVENT SAMPLE CREATION
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

    # -------------------------------------------------------------------------
    # Validate event geometry
    # -------------------------------------------------------------------------

    validate_event_metadata(
        event=event,
        dataframe_length=len(dataframe),
        gap_length=gap_length,
    )

    samples: List[Dict] = []

    # -------------------------------------------------------------------------
    # Create ONE supervised sample per missing LP.
    # -------------------------------------------------------------------------

    for position in range(
        gap_length
    ):

        target_index = (
            gap_start
            + position
        )

        if target_index > gap_end:
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                "target index exceeds event gap."
            )

        row = dataframe.iloc[
            target_index
        ]

        # ---------------------------------------------------------------------
        # Target must be masked.
        # ---------------------------------------------------------------------

        if not pd.isna(
            row["target"]
        ):
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                f"target at index {target_index} "
                "is not masked."
            )

        # ---------------------------------------------------------------------
        # Ground truth must exist.
        # ---------------------------------------------------------------------

        ground_truth = row[
            "ground_truth"
        ]

        if pd.isna(
            ground_truth
        ):
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                f"ground truth missing at "
                f"index {target_index}."
            )

        # ---------------------------------------------------------------------
        # Basic event information.
        # ---------------------------------------------------------------------

        sample: Dict = {

            "event_id": int(
                event["gap_id"]
            ),

            "gap_length": gap_length,

            "split": str(
                event["split"]
            ),

            "target_index": target_index,

            "gap_position": (
                position + 1
            ),

            "gap_position_fraction": (
                (position + 1)
                / gap_length
            ),

            "Time": row[
                "Time"
            ],

            "ground_truth": float(
                ground_truth
            ),
        }

        # ---------------------------------------------------------------------
        # Deterministic timestamp/calendar features.
        # ---------------------------------------------------------------------

        for feature in TIME_FEATURES:

            if feature not in dataframe.columns:
                fail(
                    f"Missing deterministic feature: "
                    f"{feature}"
                )

            value = row[
                feature
            ]

            if pd.isna(value):
                fail(
                    f"Feature {feature} is NaN at "
                    f"target index {target_index}."
                )

            sample[
                feature
            ] = float(value)

        # ---------------------------------------------------------------------
        # Fixed event context.
        #
        # IMPORTANT:
        #
        # target_index is NOT passed here.
        #
        # This prevents:
        #
        #     G1 -> sees G2
        #     G2 -> sees G3
        #
        # type leakage.
        # ---------------------------------------------------------------------

        context_features = (
            create_context_features(
                dataframe=dataframe,
                left_start=left_start,
                left_end=left_end,
                right_start=right_start,
                right_end=right_end,
            )
        )

        sample.update(
            context_features
        )

        samples.append(
            sample
        )

    return samples


# =============================================================================
# SUPERVISED DATASET CREATION
# =============================================================================

def create_supervised_dataset(
    dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    all_samples: List[Dict] = []

    # Metadata is authoritative.
    for _, event in metadata.iterrows():

        event_samples = (
            create_event_samples(
                dataframe=dataframe,
                event=event,
                gap_length=gap_length,
            )
        )

        all_samples.extend(
            event_samples
        )

    if not all_samples:
        fail(
            f"Gap {gap_length}: "
            "no supervised samples created."
        )

    return pd.DataFrame(
        all_samples
    )


# =============================================================================
# SUPERVISED FEATURE IDENTIFICATION
# =============================================================================

def get_feature_columns(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Return the columns that will be supplied to Random Forest.

    Metadata / bookkeeping columns are deliberately excluded.
    """

    excluded = {
        "event_id",
        "gap_length",
        "split",
        "target_index",
        "gap_position",
        "gap_position_fraction",
        "Time",
        "ground_truth",
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    if not feature_columns:
        fail(
            "No ML feature columns found."
        )

    return feature_columns


# =============================================================================
# TARGET LEAKAGE VALIDATION
# =============================================================================

def validate_target_leakage(
    supervised: pd.DataFrame,
) -> None:

    feature_columns = get_feature_columns(
        supervised
    )

    forbidden_exact = {
        "target",
        "ground_truth",
    }

    for column in feature_columns:

        if column in forbidden_exact:
            fail(
                f"Target leakage detected: "
                f"feature '{column}'."
            )

        if column.startswith(
            "target_"
        ):

            # Context features are allowed.
            if not (
                column.startswith(
                    "target_left_"
                )
                or column.startswith(
                    "target_right_"
                )
            ):
                fail(
                    f"Target leakage detected: "
                    f"feature '{column}'."
                )

    info(
        "Target leakage check   : PASSED"
    )


# =============================================================================
# SUPERVISED DATASET VALIDATION
# =============================================================================

def validate_supervised_dataset(
    supervised: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int,
) -> None:

    section(
        f"SUPERVISED DATASET VALIDATION — "
        f"{gap_length} LP"
    )

    # -------------------------------------------------------------------------
    # Expected row count
    # -------------------------------------------------------------------------

    expected_rows = (
        len(metadata)
        * gap_length
    )

    if len(supervised) != expected_rows:
        fail(
            f"Gap {gap_length}: incorrect supervised "
            "row count.\n"
            f"Expected: {expected_rows}\n"
            f"Actual  : {len(supervised)}"
        )

    info(
        "Row count              : PASSED"
    )

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required_columns = [
        "event_id",
        "gap_length",
        "split",
        "target_index",
        "gap_position",
        "gap_position_fraction",
        "Time",
        "ground_truth",
    ]

    required_columns.extend(
        TIME_FEATURES
    )

    for offset in LEFT_OFFSETS:
        required_columns.append(
            f"target_left_{offset}"
        )

    for offset in RIGHT_OFFSETS:
        required_columns.append(
            f"target_right_{offset}"
        )

    missing = [
        column
        for column in required_columns
        if column not in supervised.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: required supervised "
            "features missing:\n"
            + "\n".join(
                f"    {column}"
                for column in missing
            )
        )

    info(
        "Required features      : PASSED"
    )

    # -------------------------------------------------------------------------
    # NaN check
    # -------------------------------------------------------------------------

    feature_columns = get_feature_columns(
        supervised
    )

    if supervised[
        feature_columns
    ].isna().any().any():

        bad_columns = [
            column
            for column in feature_columns
            if supervised[column].isna().any()
        ]

        fail(
            f"Gap {gap_length}: NaN found in "
            "ML features:\n"
            + "\n".join(
                f"    {column}"
                for column in bad_columns
            )
        )

    info(
        "NaN check              : PASSED"
    )

    # -------------------------------------------------------------------------
    # Infinite check
    # -------------------------------------------------------------------------

    numeric_features = supervised[
        feature_columns
    ].select_dtypes(
        include=[np.number]
    )

    if np.isinf(
        numeric_features.to_numpy()
    ).any():

        fail(
            f"Gap {gap_length}: Inf found in "
            "ML features."
        )

    info(
        "Inf check              : PASSED"
    )

    # -------------------------------------------------------------------------
    # Split integrity
    # -------------------------------------------------------------------------

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        supervised["split"].unique()
    )

    if not actual_splits.issubset(
        allowed_splits
    ):
        fail(
            f"Gap {gap_length}: unexpected split values:\n"
            f"{actual_splits}"
        )

    info(
        "Split integrity        : PASSED"
    )

    # -------------------------------------------------------------------------
    # One row per missing LP
    # -------------------------------------------------------------------------

    duplicate_targets = (
        supervised
        .duplicated(
            subset=[
                "event_id",
                "target_index",
            ],
            keep=False,
        )
    )

    if duplicate_targets.any():
        fail(
            f"Gap {gap_length}: duplicate "
            "(event_id, target_index) samples found."
        )

    expected_positions = set(
        range(
            1,
            gap_length + 1
        )
    )

    for event_id, group in supervised.groupby(
        "event_id"
    ):

        positions = set(
            group[
                "gap_position"
            ].astype(int)
        )

        if positions != expected_positions:
            fail(
                f"Gap {gap_length}, event {event_id}: "
                "gap positions are incorrect.\n"
                f"Expected: {expected_positions}\n"
                f"Actual  : {positions}"
            )

        if len(group) != gap_length:
            fail(
                f"Gap {gap_length}, event {event_id}: "
                f"expected {gap_length} samples, "
                f"got {len(group)}."
            )

    info(
        "One sample / missing LP: PASSED"
    )

    # -------------------------------------------------------------------------
    # Gap position validation
    # -------------------------------------------------------------------------

    if not (
        supervised["gap_position"].between(
            1,
            gap_length
        ).all()
    ):
        fail(
            f"Gap {gap_length}: invalid gap positions."
        )

    info(
        "Gap position check     : PASSED"
    )

    # -------------------------------------------------------------------------
    # Gap length validation
    # -------------------------------------------------------------------------

    if not (
        supervised["gap_length"]
        == gap_length
    ).all():
        fail(
            f"Gap {gap_length}: gap_length column "
            "contains incorrect values."
        )

    # -------------------------------------------------------------------------
    # Target leakage
    # -------------------------------------------------------------------------

    validate_target_leakage(
        supervised
    )

    # -------------------------------------------------------------------------
    # Context boundary validation
    #
    # This is the most important multi-LP check.
    # -------------------------------------------------------------------------

    for _, event in metadata.iterrows():

        gap_start = int(
            event["gap_start_index"]
        )

        gap_end = int(
            event["gap_end_index"]
        )

        right_start = int(
            event["right_context_start_index"]
        )

        right_end = int(
            event["right_context_end_index"]
        )

        left_start = int(
            event["left_context_start_index"]
        )

        left_end = int(
            event["left_context_end_index"]
        )

        # ---------------------------------------------------------------------
        # Verify fixed context geometry.
        # ---------------------------------------------------------------------

        if left_end != gap_start - 1:
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                "left context boundary incorrect."
            )

        if right_start != gap_end + 1:
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                "right context enters missing gap."
            )

        if (
            right_end
            - right_start
            + 1
            != CONTEXT_RIGHT
        ):
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                "right context length incorrect."
            )

        if (
            left_end
            - left_start
            + 1
            != CONTEXT_LEFT
        ):
            fail(
                f"Gap {gap_length}, event "
                f"{event['gap_id']}: "
                "left context length incorrect."
            )

        # ---------------------------------------------------------------------
        # Every supervised sample from this event must use the same
        # context-derived features.
        # ---------------------------------------------------------------------

        event_rows = supervised[
            supervised["event_id"]
            == int(event["gap_id"])
        ]

        for feature in [
            *[
                f"target_left_{x}"
                for x in LEFT_OFFSETS
            ],
            *[
                f"target_right_{x}"
                for x in RIGHT_OFFSETS
            ],
        ]:

            if event_rows[
                feature
            ].nunique(dropna=False) != 1:

                fail(
                    f"Gap {gap_length}, event "
                    f"{event['gap_id']}: "
                    f"context feature '{feature}' "
                    "changes between missing LPs."
                )

    info(
        "Context boundary check : PASSED"
    )


# =============================================================================
# SAVE DATASET
# =============================================================================

def save_supervised_dataset(
    supervised: pd.DataFrame,
    gap_length: int,
    metadata: pd.DataFrame,
) -> None:

    dataframe_path = (
        get_output_dataframe_path(
            gap_length
        )
    )

    metadata_path = (
        get_output_metadata_path(
            gap_length
        )
    )

    # -------------------------------------------------------------------------
    # Save parquet
    # -------------------------------------------------------------------------

    supervised.to_parquet(
        dataframe_path,
        index=False,
    )

    info(
        "Saved dataframe:\n"
        f"    {dataframe_path}"
    )

    # -------------------------------------------------------------------------
    # Dataset metadata
    # -------------------------------------------------------------------------

    feature_columns = get_feature_columns(
        supervised
    )

    split_counts = (
        supervised["split"]
        .value_counts()
        .to_dict()
    )

    event_counts = (
        metadata["split"]
        .value_counts()
        .to_dict()
    )

    metadata_output = {

        "project": {
            "name": CONFIG[
                "project"
            ]["name"],
            "version": CONFIG[
                "project"
            ]["version"],
        },

        "meter": {
            "meter_id": DATA_CONFIG[
                "selected_meter"
            ],
            "direction": DATA_CONFIG[
                "target_direction"
            ],
        },

        "gap": {
            "gap_length": gap_length,
            "gap_type": GAP_CONFIG[
                "type"
            ],
            "context_left_lp": CONTEXT_LEFT,
            "context_right_lp": CONTEXT_RIGHT,
        },

        "formulation": {
            "type": SUPERVISED_CONFIG[
                "formulation"
            ],
            "sliding_windows": False,
            "one_prediction_per_missing_lp": True,
            "same_context_for_event_targets": True,
        },

        "events": {
            "total": int(len(metadata)),
            "train": int(
                event_counts.get(
                    "train",
                    0
                )
            ),
            "validation": int(
                event_counts.get(
                    "validation",
                    0
                )
            ),
            "test": int(
                event_counts.get(
                    "test",
                    0
                )
            ),
        },

        "samples": {
            "total": int(
                len(supervised)
            ),
            "train": int(
                split_counts.get(
                    "train",
                    0
                )
            ),
            "validation": int(
                split_counts.get(
                    "validation",
                    0
                )
            ),
            "test": int(
                split_counts.get(
                    "test",
                    0
                )
            ),
        },

        "features": {
            "count": len(
                feature_columns
            ),
            "columns": feature_columns,
            "time_features": TIME_FEATURES,
            "left_context_offsets": LEFT_OFFSETS,
            "right_context_offsets": RIGHT_OFFSETS,
        },

        "target": {
            "column": "ground_truth",
            "target_derived_features": False,
            "normalization": False,
        },

        "source_modification": False,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata_output,
            file,
            indent=4,
        )

    info(
        "Saved metadata:\n"
        f"    {metadata_path}"
    )


# =============================================================================
# SAVED DATASET VERIFICATION
# =============================================================================

def verify_saved_dataset(
    dataframe_path: Path,
    expected_shape,
) -> None:

    section(
        "SAVED DATASET VERIFICATION"
    )

    if not dataframe_path.exists():
        fail(
            "Saved supervised dataframe does not exist."
        )

    reloaded = pd.read_parquet(
        dataframe_path
    )

    if reloaded.shape != expected_shape:
        fail(
            "Saved supervised dataframe shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {reloaded.shape}"
        )

    info(
        "Parquet reload          : PASSED"
    )

    info(
        f"Verified shape          : "
        f"{reloaded.shape}"
    )

    # -------------------------------------------------------------------------
    # Verify no NaN in predictors.
    # -------------------------------------------------------------------------

    features = get_feature_columns(
        reloaded
    )

    if reloaded[
        features
    ].isna().any().any():
        fail(
            "Reloaded supervised dataset "
            "contains NaN features."
        )

    info(
        "Data integrity          : PASSED"
    )


# =============================================================================
# PROCESS ONE GAP LENGTH
# =============================================================================

def process_gap_length(
    gap_length: int,
) -> None:

    section(
        f"GAP LENGTH: {gap_length} LP"
    )

    dataframe = load_gap_dataframe(
        gap_length
    )

    metadata = load_gap_metadata(
        gap_length
    )

    validate_gap_input(
        dataframe=dataframe,
        metadata=metadata,
        gap_length=gap_length,
    )

    section(
        f"CREATING FIXED-EVENT SUPERVISED DATASET "
        f"— {gap_length} LP"
    )

    supervised = create_supervised_dataset(
        dataframe=dataframe,
        metadata=metadata,
        gap_length=gap_length,
    )

    validate_supervised_dataset(
        supervised=supervised,
        metadata=metadata,
        gap_length=gap_length,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    section(
        f"SUPERVISED DATASET SUMMARY — "
        f"{gap_length} LP"
    )

    feature_columns = get_feature_columns(
        supervised
    )

    info(
        f"Gap length             : {gap_length} LP"
    )

    info(
        f"Total supervised rows  : "
        f"{len(supervised):,}"
    )

    info(
        f"Train samples          : "
        f"{(
            supervised['split'] == 'train'
        ).sum():,}"
    )

    info(
        f"Validation samples     : "
        f"{(
            supervised['split'] == 'validation'
        ).sum():,}"
    )

    info(
        f"Test samples           : "
        f"{(
            supervised['split'] == 'test'
        ).sum():,}"
    )

    info(
        f"Feature columns        : "
        f"{len(feature_columns):,}"
    )

    info(
        "Prediction formulation : "
        "one prediction per missing LP"
    )

    # -------------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------------

    print()
    print(
        "Dataset preview:"
    )

    preview_columns = [
        "event_id",
        "split",
        "target_index",
        "gap_position",
        "Time",
        "hour",
        "half_hour_slot",
        "day_of_week",
        "target_left_1",
        "target_left_2",
        "target_left_24",
        "target_right_1",
        "target_right_2",
        "target_right_24",
        "ground_truth",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in supervised.columns
    ]

    print(
        supervised[
            preview_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    section(
        f"SAVING SUPERVISED DATASET — "
        f"{gap_length} LP"
    )

    save_supervised_dataset(
        supervised=supervised,
        gap_length=gap_length,
        metadata=metadata,
    )

    verify_saved_dataset(
        dataframe_path=get_output_dataframe_path(
            gap_length
        ),
        expected_shape=supervised.shape,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print()
    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION — "
        "SUPERVISED DATASET STAGE"
    )
    print("=" * 80)

    info(
        "Stage: fixed gap events → supervised samples"
    )

    info(
        "Formulation: fixed event windows"
    )

    info(
        "Gap lengths: 1, 6, 24, 48 LP"
    )

    info(
        "96 LP gap: REMOVED"
    )

    info(
        f"Context: {CONTEXT_LEFT} LP left + "
        f"{CONTEXT_RIGHT} LP right"
    )

    info(
        "Sliding windows: DISABLED"
    )

    info(
        "One RF prediction per missing LP: ENABLED"
    )

    info(
        "Target-derived lag/lead features: DISABLED"
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

    # -------------------------------------------------------------------------
    # Configuration sanity checks
    # -------------------------------------------------------------------------

    if GAP_LENGTHS != [
        1,
        6,
        24,
        48,
    ]:
        fail(
            "Configuration gap lengths do not match "
            "the approved experiment design.\n"
            f"Configured: {GAP_LENGTHS}\n"
            "Expected  : [1, 6, 24, 48]"
        )

    if (
        SUPERVISED_CONFIG[
            "formulation"
        ]
        != "fixed_event_window"
    ):
        fail(
            "Configuration formulation must be "
            "'fixed_event_window'."
        )

    # -------------------------------------------------------------------------
    # Process each independent experiment.
    # -------------------------------------------------------------------------

    for gap_length in GAP_LENGTHS:

        process_gap_length(
            gap_length
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    section(
        "SUPERVISED DATASET STAGE COMPLETE"
    )

    info(
        "Generated independent supervised datasets:"
    )

    for gap_length in GAP_LENGTHS:

        path = get_output_dataframe_path(
            gap_length
        )

        info(
            f"    {gap_length:>2} LP : {path}"
        )

    print()
    info(
        "96 LP gap                : REMOVED"
    )

    info(
        "Sliding windows          : DISABLED"
    )

    info(
        "One prediction / LP      : ENABLED"
    )

    info(
        "Target leakage           : CHECKED"
    )

    info(
        "Context boundary         : CHECKED"
    )

    info(
        "Source CSV               : UNCHANGED"
    )

    info(
        "Feature dataframe        : UNCHANGED"
    )

    info(
        "Gap datasets             : UNCHANGED"
    )

    print()
    info(
        "NEXT STAGE:"
    )

    info(
        "Train Random Forest separately "
        "for the approved gap experiments."
    )

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()