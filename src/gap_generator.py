# =============================================================================
# src/gap_generator.py
# =============================================================================
"""
Random Forest CDP Load Profile Imputation
Fixed Event Gap Generation Stage

Purpose
-------
Create controlled artificial interior missing-LP events from the validated
feature dataframe.

IMPORTANT DESIGN
----------------
This module does NOT create sliding-window supervised samples.

Instead, it creates FIXED EVENTS.

For every gap experiment:

    1 LP
    6 LP
    24 LP
    48 LP

a controlled number of missing events is generated.

Each event contains:

    96 LP left context
    GAP
    96 LP right context

The gap is represented by:

    target = NaN

while:

    ground_truth = original target

is retained for later evaluation.

The original feature dataframe is NEVER modified.

No model training occurs here.
No Random Forest occurs here.
No normalization occurs here.
No supervised samples are created here.

Later:

    gap_generator.py
          ↓
    supervised_dataset.py
          ↓
    train.py
          ↓
    evaluation.py
          ↓
    impute.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# LOGGING
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
# CONFIGURATION HELPER
# =============================================================================

def get_config(
    config: dict,
    *keys: str,
    default=None
):
    """
    Retrieve nested configuration value.
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
# PATHS
# =============================================================================

def get_feature_dataframe_path(
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

    return (
        PROJECT_ROOT
        / processed_dir
        / filename
    )


def get_gap_output_directory(
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
        / "gaps"
    )


# =============================================================================
# LOAD FEATURE DATAFRAME
# =============================================================================

def load_feature_dataframe(
    config: dict
) -> pd.DataFrame:

    input_path = get_feature_dataframe_path(
        config
    )

    section(
        "LOADING FEATURE-ENGINEERED DATAFRAME"
    )

    info(
        f"Input file : {input_path}"
    )

    if not input_path.exists():

        fail(
            "Feature dataframe was not found:\n"
            f"    {input_path}\n\n"
            "Run feature engineering first:\n"
            "    python -m src.feature_engineering"
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

    section(
        "GAP GENERATOR INPUT VALIDATION"
    )

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
    # Timestamp
    # -------------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):

        fail(
            "Time column is not datetime."
        )

    if dataframe["Time"].duplicated().any():

        fail(
            "Duplicate timestamps detected."
        )

    if not dataframe[
        "Time"
    ].is_monotonic_increasing:

        fail(
            "Timestamps are not chronological."
        )

    info(
        "Timestamp validation   : PASSED"
    )

    # -------------------------------------------------------------------------
    # Interval
    # -------------------------------------------------------------------------

    expected_minutes = get_config(
        config,
        "data",
        "expected_interval_minutes",
        default=30
    )

    expected_delta = pd.Timedelta(
        minutes=expected_minutes
    )

    differences = (
        dataframe["Time"]
        .diff()
        .dropna()
    )

    if not (
        differences == expected_delta
    ).all():

        fail(
            "Timestamp continuity failed."
        )

    info(
        "30-minute continuity  : PASSED"
    )

    # -------------------------------------------------------------------------
    # Complete target
    # -------------------------------------------------------------------------

    if dataframe[
        "target"
    ].isna().any():

        fail(
            "Feature dataframe already contains "
            "missing target values."
        )

    if np.isinf(
        dataframe[
            "target"
        ].to_numpy(
            dtype=np.float64
        )
    ).any():

        fail(
            "Target contains infinite values."
        )

    info(
        "Complete target        : PASSED"
    )

    # -------------------------------------------------------------------------
    # Single meter
    # -------------------------------------------------------------------------

    if dataframe[
        "meter_id"
    ].nunique() != 1:

        fail(
            "More than one meter exists."
        )

    # -------------------------------------------------------------------------
    # Single direction
    # -------------------------------------------------------------------------

    if dataframe[
        "target_direction"
    ].nunique() != 1:

        fail(
            "More than one target direction exists."
        )

    info(
        "Single meter/direction : PASSED"
    )


# =============================================================================
# CHRONOLOGICAL SPLIT
# =============================================================================

def create_chronological_split(
    dataframe: pd.DataFrame,
    config: dict
) -> pd.DataFrame:

    section(
        "CHRONOLOGICAL SPLIT"
    )

    n_rows = len(dataframe)

    train_ratio = get_config(
        config,
        "split",
        "train_ratio",
        default=0.70
    )

    validation_ratio = get_config(
        config,
        "split",
        "validation_ratio",
        default=0.15
    )

    test_ratio = get_config(
        config,
        "split",
        "test_ratio",
        default=0.15
    )

    total_ratio = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if not np.isclose(
        total_ratio,
        1.0
    ):

        fail(
            "Train/validation/test ratios "
            "must sum to 1.0."
        )

    train_end = int(
        n_rows * train_ratio
    )

    validation_end = (
        train_end
        + int(
            n_rows * validation_ratio
        )
    )

    # Ensure the final split consumes every remaining row.
    split_labels = np.empty(
        n_rows,
        dtype=object
    )

    split_labels[
        :train_end
    ] = "train"

    split_labels[
        train_end:validation_end
    ] = "validation"

    split_labels[
        validation_end:
    ] = "test"

    split_dataframe = dataframe.copy()

    split_dataframe[
        "dataset_split"
    ] = split_labels

    info(
        f"Train      : rows 0 → {train_end - 1:,}"
    )

    info(
        f"Validation : rows {train_end:,} → "
        f"{validation_end - 1:,}"
    )

    info(
        f"Test       : rows {validation_end:,} → "
        f"{n_rows - 1:,}"
    )

    info(
        f"Train rows      : "
        f"{(split_labels == 'train').sum():,}"
    )

    info(
        f"Validation rows : "
        f"{(split_labels == 'validation').sum():,}"
    )

    info(
        f"Test rows       : "
        f"{(split_labels == 'test').sum():,}"
    )

    return split_dataframe


# =============================================================================
# EVENT CANDIDATES
# =============================================================================

def generate_candidate_starts(
    split_start: int,
    split_end: int,
    gap_length: int,
    context_left: int,
    context_right: int
) -> np.ndarray:
    """
    Return valid gap-start positions for a single split.

    The event must have:

        96 LP before the gap
        gap_length LP gap
        96 LP after the gap

    ALL of those positions must remain inside the same
    train/validation/test split.

    This prevents context from crossing split boundaries.
    """

    first_start = (
        split_start
        + context_left
    )

    last_start = (
        split_end
        - gap_length
        - context_right
    )

    if last_start < first_start:

        return np.array(
            [],
            dtype=np.int64
        )

    return np.arange(
        first_start,
        last_start + 1,
        dtype=np.int64
    )


# =============================================================================
# CONTROLLED EVENT SAMPLING
# =============================================================================

def select_event_starts(
    candidate_starts: np.ndarray,
    number_of_events: int,
    minimum_separation: int,
    random_seed: int
) -> list[int]:
    """
    Select a controlled number of non-overlapping event locations.

    minimum_separation is measured between gap START positions.

    Because each event has 96 LP context on both sides, the default
    separation of 288 LP gives substantial separation between events.
    """

    if number_of_events <= 0:

        return []

    if len(candidate_starts) == 0:

        fail(
            "No valid candidate event locations exist."
        )

    rng = np.random.default_rng(
        random_seed
    )

    shuffled = candidate_starts.copy()

    rng.shuffle(
        shuffled
    )

    selected: list[int] = []

    for candidate in shuffled:

        if all(
            abs(
                candidate - selected_start
            ) >= minimum_separation
            for selected_start in selected
        ):

            selected.append(
                int(candidate)
            )

            if len(selected) == number_of_events:

                break

    selected.sort()

    if len(selected) != number_of_events:

        fail(
            "Unable to generate the requested number "
            "of fixed events.\n"
            f"Requested       : {number_of_events}\n"
            f"Generated       : {len(selected)}\n"
            f"Minimum gap     : {minimum_separation} LP\n"
            f"Candidates      : {len(candidate_starts)}"
        )

    return selected


# =============================================================================
# COMMON TEST EVENTS
# =============================================================================

def generate_common_test_events(
    dataframe: pd.DataFrame,
    config: dict,
    maximum_gap_length: int
) -> list[int]:

    section(
        "GENERATING COMMON TEST EVENT LOCATIONS"
    )

    context_left = get_config(
        config,
        "gaps",
        "context_left",
        default=96
    )

    context_right = get_config(
        config,
        "gaps",
        "context_right",
        default=96
    )

    test_events = get_config(
        config,
        "supervised",
        "test_events",
        default=6
    )

    minimum_separation = get_config(
        config,
        "supervised",
        "minimum_event_separation",
        default=288
    )

    random_seed = get_config(
        config,
        "gaps",
        "random_seed",
        default=42
    )

    test_indices = np.where(
        dataframe[
            "dataset_split"
        ].to_numpy()
        == "test"
    )[0]

    if len(test_indices) == 0:

        fail(
            "No test rows found."
        )

    test_start = int(
        test_indices[0]
    )

    test_end = int(
        test_indices[-1]
        + 1
    )

    candidates = generate_candidate_starts(
        split_start=test_start,
        split_end=test_end,
        gap_length=maximum_gap_length,
        context_left=context_left,
        context_right=context_right
    )

    starts = select_event_starts(
        candidate_starts=candidates,
        number_of_events=test_events,
        minimum_separation=minimum_separation,
        random_seed=random_seed
    )

    info(
        f"Random seed              : {random_seed}"
    )

    info(
        f"Test events              : {len(starts)}"
    )

    info(
        "Common test event starts:"
    )

    for number, start in enumerate(
        starts,
        start=1
    ):

        timestamp = dataframe.loc[
            start,
            "Time"
        ]

        info(
            f"  Event {number:2d}: "
            f"row {start:,} | {timestamp}"
        )

    return starts


# =============================================================================
# TRAIN / VALIDATION EVENT LOCATIONS
# =============================================================================

def generate_train_validation_events(
    dataframe: pd.DataFrame,
    config: dict,
    gap_length: int
) -> dict:

    context_left = get_config(
        config,
        "gaps",
        "context_left",
        default=96
    )

    context_right = get_config(
        config,
        "gaps",
        "context_right",
        default=96
    )

    minimum_separation = get_config(
        config,
        "supervised",
        "minimum_event_separation",
        default=288
    )

    training_events = get_config(
        config,
        "supervised",
        "training_events",
        default=100
    )

    validation_events = get_config(
        config,
        "supervised",
        "validation_events",
        default=30
    )

    random_seed = get_config(
        config,
        "gaps",
        "random_seed",
        default=42
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    train_indices = np.where(
        dataframe[
            "dataset_split"
        ].to_numpy()
        == "train"
    )[0]

    train_start = int(
        train_indices[0]
    )

    train_end = int(
        train_indices[-1]
        + 1
    )

    train_candidates = generate_candidate_starts(
        split_start=train_start,
        split_end=train_end,
        gap_length=gap_length,
        context_left=context_left,
        context_right=context_right
    )

    train_starts = select_event_starts(
        candidate_starts=train_candidates,
        number_of_events=training_events,
        minimum_separation=minimum_separation,
        random_seed=random_seed + gap_length
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    validation_indices = np.where(
        dataframe[
            "dataset_split"
        ].to_numpy()
        == "validation"
    )[0]

    validation_start = int(
        validation_indices[0]
    )

    validation_end = int(
        validation_indices[-1]
        + 1
    )

    validation_candidates = generate_candidate_starts(
        split_start=validation_start,
        split_end=validation_end,
        gap_length=gap_length,
        context_left=context_left,
        context_right=context_right
    )

    validation_starts = select_event_starts(
        candidate_starts=validation_candidates,
        number_of_events=validation_events,
        minimum_separation=minimum_separation,
        random_seed=random_seed + 1000 + gap_length
    )

    return {
        "train": train_starts,
        "validation": validation_starts,
    }


# =============================================================================
# CREATE EVENT MASK
# =============================================================================

def apply_gap_events(
    dataframe: pd.DataFrame,
    event_starts: dict,
    gap_length: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = dataframe.copy()

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    df[
        "ground_truth"
    ] = df[
        "target"
    ].copy()

    # -------------------------------------------------------------------------
    # Event metadata columns
    # -------------------------------------------------------------------------

    df[
        "is_gap"
    ] = np.int8(0)

    df[
        "gap_id"
    ] = np.int32(-1)

    df[
        "event_position"
    ] = np.int32(-1)

    df[
        "event_split"
    ] = ""

    metadata_records = []

    global_event_id = 0

    # -------------------------------------------------------------------------
    # Process train / validation / test
    # -------------------------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        starts = event_starts.get(
            split_name,
            []
        )

        for start in starts:

            global_event_id += 1

            gap_start = int(
                start
            )

            gap_end = (
                gap_start
                + gap_length
                - 1
            )

            # -------------------------------------------------------------
            # Safety validation
            # -------------------------------------------------------------

            if gap_start < 0:

                fail(
                    "Negative gap start detected."
                )

            if gap_end >= len(df):

                fail(
                    "Gap extends beyond dataframe."
                )

            # -------------------------------------------------------------
            # Verify same split
            # -------------------------------------------------------------

            gap_splits = df.loc[
                gap_start:gap_end,
                "dataset_split"
            ].unique()

            if len(gap_splits) != 1:

                fail(
                    "Gap crosses a train/validation/test boundary."
                )

            if gap_splits[0] != split_name:

                fail(
                    "Event split does not match dataset split."
                )

            # -------------------------------------------------------------
            # Mask target
            # -------------------------------------------------------------

            positions = np.arange(
                gap_start,
                gap_end + 1
            )

            df.loc[
                gap_start:gap_end,
                "target"
            ] = np.nan

            df.loc[
                gap_start:gap_end,
                "is_gap"
            ] = 1

            df.loc[
                gap_start:gap_end,
                "gap_id"
            ] = global_event_id

            df.loc[
                gap_start:gap_end,
                "event_split"
            ] = split_name

            df.loc[
                gap_start:gap_end,
                "event_position"
            ] = np.arange(
                gap_length,
                dtype=np.int32
            )

            # -------------------------------------------------------------
            # Metadata
            # -------------------------------------------------------------

            context_left_start = (
                gap_start
                - get_config(
                    CONFIG,
                    "gaps",
                    "context_left",
                    default=96
                )
            )

            context_right_end = (
                gap_end
                + get_config(
                    CONFIG,
                    "gaps",
                    "context_right",
                    default=96
                )
            )

            metadata_records.append({

                "gap_id":
                    global_event_id,

                "split":
                    split_name,

                "gap_length":
                    gap_length,

                "gap_start_index":
                    gap_start,

                "gap_end_index":
                    gap_end,

                "gap_start_time":
                    str(
                        df.loc[
                            gap_start,
                            "Time"
                        ]
                    ),

                "gap_end_time":
                    str(
                        df.loc[
                            gap_end,
                            "Time"
                        ]
                    ),

                "left_context_start_index":
                    context_left_start,

                "left_context_end_index":
                    gap_start - 1,

                "right_context_start_index":
                    gap_end + 1,

                "right_context_end_index":
                    context_right_end,

                "left_context_length":
                    get_config(
                        CONFIG,
                        "gaps",
                        "context_left",
                        default=96
                    ),

                "right_context_length":
                    get_config(
                        CONFIG,
                        "gaps",
                        "context_right",
                        default=96
                    ),

                "target_values_masked":
                    gap_length,

            })

    metadata = pd.DataFrame(
        metadata_records
    )

    return df, metadata


# =============================================================================
# VALIDATE GAP EXPERIMENT
# =============================================================================

def validate_gap_experiment(
    original_dataframe: pd.DataFrame,
    masked_dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    gap_length: int
) -> None:

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if len(
        original_dataframe
    ) != len(
        masked_dataframe
    ):

        fail(
            f"Gap {gap_length}: row count changed."
        )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    if not original_dataframe[
        "Time"
    ].reset_index(
        drop=True
    ).equals(
        masked_dataframe[
            "Time"
        ].reset_index(
            drop=True
        )
    ):

        fail(
            f"Gap {gap_length}: timestamps changed."
        )

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    if not np.array_equal(
        original_dataframe[
            "target"
        ].to_numpy(),
        masked_dataframe[
            "ground_truth"
        ].to_numpy(),
        equal_nan=True
    ):

        fail(
            f"Gap {gap_length}: ground_truth "
            "does not match original target."
        )

    # -------------------------------------------------------------------------
    # Expected number of masked values
    # -------------------------------------------------------------------------

    expected_missing = (
        len(metadata)
        * gap_length
    )

    actual_missing = int(
        masked_dataframe[
            "target"
        ].isna().sum()
    )

    if actual_missing != expected_missing:

        fail(
            f"Gap {gap_length}: incorrect number "
            "of masked target values.\n"
            f"Expected: {expected_missing}\n"
            f"Actual  : {actual_missing}"
        )

    # -------------------------------------------------------------------------
    # Ground truth complete
    # -------------------------------------------------------------------------

    if masked_dataframe[
        "ground_truth"
    ].isna().any():

        fail(
            f"Gap {gap_length}: ground_truth "
            "contains NaN."
        )

    # -------------------------------------------------------------------------
    # is_gap count
    # -------------------------------------------------------------------------

    gap_rows = masked_dataframe[
        masked_dataframe[
            "is_gap"
        ] == 1
    ]

    if len(gap_rows) != expected_missing:

        fail(
            f"Gap {gap_length}: incorrect is_gap count.\n"
            f"Expected: {expected_missing}\n"
            f"Actual  : {len(gap_rows)}"
        )

    # -------------------------------------------------------------------------
    # No target missing outside gaps
    # -------------------------------------------------------------------------

    missing_outside_gap = (
        masked_dataframe[
            "target"
        ].isna()
        &
        (
            masked_dataframe[
                "is_gap"
            ] == 0
        )
    ).sum()

    if missing_outside_gap != 0:

        fail(
            f"Gap {gap_length}: target contains "
            "NaN outside artificial gaps."
        )

    # -------------------------------------------------------------------------
    # Every event has correct number of rows
    # -------------------------------------------------------------------------

    event_counts = (
        gap_rows
        .groupby(
            "gap_id"
        )
        .size()
    )

    if not (
        event_counts == gap_length
    ).all():

        fail(
            f"Gap {gap_length}: at least one event "
            "does not contain exactly the expected "
            "number of LPs."
        )

    # -------------------------------------------------------------------------
    # Gap IDs
    # -------------------------------------------------------------------------

    expected_event_ids = set(
        metadata[
            "gap_id"
        ].astype(int)
    )

    actual_event_ids = set(
        gap_rows[
            "gap_id"
        ].astype(int)
    )

    if expected_event_ids != actual_event_ids:

        fail(
            f"Gap {gap_length}: gap ID mismatch."
        )

    # -------------------------------------------------------------------------
    # Event positions
    # -------------------------------------------------------------------------

    for gap_id, group in gap_rows.groupby(
        "gap_id"
    ):

        positions = (
            group[
                "event_position"
            ]
            .astype(int)
            .to_numpy()
        )

        expected_positions = np.arange(
            gap_length
        )

        if not np.array_equal(
            positions,
            expected_positions
        ):

            fail(
                f"Gap {gap_length}: event {gap_id} "
                "has invalid event positions."
            )

    info(
        f"Gap {gap_length:3d} LP validation : PASSED"
    )


# =============================================================================
# SAVE EXPERIMENT
# =============================================================================

def save_gap_experiment(
    masked_dataframe: pd.DataFrame,
    metadata: pd.DataFrame,
    config: dict,
    gap_length: int
) -> None:

    output_root = get_gap_output_directory(
        config
    )

    experiment_directory = (
        output_root
        / f"gap_{gap_length}"
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True
    )

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

    dataframe_path = (
        experiment_directory
        /
        (
            f"{selected_meter}_"
            f"{target_direction}_"
            f"gap_{gap_length}.parquet"
        )
    )

    metadata_path = (
        experiment_directory
        / "gap_metadata.csv"
    )

    masked_dataframe.to_parquet(
        dataframe_path,
        index=False
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    info(
        "Saved dataframe:"
    )

    info(
        f"    {dataframe_path}"
    )

    info(
        "Saved metadata:"
    )

    info(
        f"    {metadata_path}"
    )


# =============================================================================
# CREATE MANIFEST
# =============================================================================

def create_manifest(
    config: dict,
    dataframe: pd.DataFrame,
    experiment_information: list[dict],
    common_test_starts: list[int]
) -> None:

    output_root = get_gap_output_directory(
        config
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

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

    manifest = {

        "project":
            get_config(
                config,
                "project",
                "name"
            ),

        "selected_meter":
            selected_meter,

        "target_direction":
            target_direction,

        "total_rows":
            len(dataframe),

        "timestamp_interval_minutes":
            get_config(
                config,
                "data",
                "expected_interval_minutes",
                default=30
            ),

        "gap_type":
            "interior",

        "context_left":
            get_config(
                config,
                "gaps",
                "context_left",
                default=96
            ),

        "context_right":
            get_config(
                config,
                "gaps",
                "context_right",
                default=96
            ),

        "gap_lengths":
            get_config(
                config,
                "gaps",
                "lengths",
                default=[1, 6, 24, 48]
            ),

        "formulation":
            "fixed_event_window",

        "training_events":
            get_config(
                config,
                "supervised",
                "training_events",
                default=100
            ),

        "validation_events":
            get_config(
                config,
                "supervised",
                "validation_events",
                default=30
            ),

        "test_events":
            get_config(
                config,
                "supervised",
                "test_events",
                default=6
            ),

        "minimum_event_separation":
            get_config(
                config,
                "supervised",
                "minimum_event_separation",
                default=288
            ),

        "random_seed":
            get_config(
                config,
                "gaps",
                "random_seed",
                default=42
            ),

        "common_test_locations":
            get_config(
                config,
                "gaps",
                "common_test_locations",
                default=True
            ),

        "common_test_start_indices":
            common_test_starts,

        "experiments":
            experiment_information,

        "source_modified":
            False,

        "feature_dataframe_modified":
            False,

        "model_training":
            False,

        "normalization":
            False,

    }

    manifest_path = (
        output_root
        / "gap_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    info(
        "Manifest saved:"
    )

    info(
        f"    {manifest_path}"
    )


# =============================================================================
# VERIFY SOURCE FEATURE FILE WAS NOT MODIFIED
# =============================================================================

def verify_source_unchanged(
    original_dataframe: pd.DataFrame,
    source_path: Path
) -> None:

    section(
        "VERIFYING FEATURE SOURCE REMAINS UNCHANGED"
    )

    reloaded = pd.read_parquet(
        source_path
    )

    if reloaded.shape != original_dataframe.shape:

        fail(
            "Feature source dataframe shape changed."
        )

    if list(
        reloaded.columns
    ) != list(
        original_dataframe.columns
    ):

        fail(
            "Feature source dataframe columns changed."
        )

    if not np.array_equal(
        reloaded[
            "target"
        ].to_numpy(),
        original_dataframe[
            "target"
        ].to_numpy(),
        equal_nan=True
    ):

        fail(
            "Feature source target values changed."
        )

    info(
        "Feature dataframe unchanged : PASSED"
    )


# =============================================================================
# GLOBAL CONFIGURATION REFERENCE
# =============================================================================

CONFIG: dict = {}


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    global CONFIG

    section(
        "RANDOM FOREST CDP IMPUTATION — "
        "FIXED EVENT GAP GENERATION STAGE"
    )

    CONFIG = load_config()

    info(
        "Stage: feature dataframe → "
        "independent fixed gap experiments"
    )

    info(
        "Gap lengths: 1, 6, 24, 48 LP"
    )

    info(
        "96 LP gap: REMOVED"
    )

    info(
        "Gap type: interior only"
    )

    info(
        "Context: 96 LP left + 96 LP right"
    )

    info(
        "Split: chronological 70 / 15 / 15"
    )

    info(
        "Supervised formulation: fixed event windows"
    )

    info(
        "Training events: 100"
    )

    info(
        "Validation events: 30"
    )

    info(
        "Test events: 6"
    )

    info(
        "Same test locations across experiments: YES"
    )

    info(
        "Sliding windows: DISABLED"
    )

    info(
        "One RF prediction per missing LP: "
        "DEFERRED TO SUPERVISED DATASET STAGE"
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
    # LOAD
    # =========================================================================

    dataframe = load_feature_dataframe(
        CONFIG
    )

    # =========================================================================
    # VALIDATE
    # =========================================================================

    validate_input_dataframe(
        dataframe,
        CONFIG
    )

    # =========================================================================
    # CHRONOLOGICAL SPLIT
    # =========================================================================

    split_dataframe = create_chronological_split(
        dataframe,
        CONFIG
    )

    # =========================================================================
    # CONFIG VALUES
    # =========================================================================

    gap_lengths = get_config(
        CONFIG,
        "gaps",
        "lengths",
        default=[1, 6, 24, 48]
    )

    context_left = get_config(
        CONFIG,
        "gaps",
        "context_left",
        default=96
    )

    context_right = get_config(
        CONFIG,
        "gaps",
        "context_right",
        default=96
    )

    # =========================================================================
    # VALIDATE GAP CONFIGURATION
    # =========================================================================

    if 96 in gap_lengths:

        fail(
            "Gap length 96 is no longer allowed.\n"
            "Remove 96 from config.yaml."
        )

    if context_left != 96:

        fail(
            "This pipeline currently requires "
            "context_left = 96."
        )

    if context_right != 96:

        fail(
            "This pipeline currently requires "
            "context_right = 96."
        )

    # =========================================================================
    # COMMON TEST LOCATIONS
    # =========================================================================

    maximum_gap_length = max(
        gap_lengths
    )

    common_test_starts = generate_common_test_events(
        dataframe=split_dataframe,
        config=CONFIG,
        maximum_gap_length=maximum_gap_length
    )

    # =========================================================================
    # CREATE EXPERIMENTS
    # =========================================================================

    section(
        "CREATING INDEPENDENT FIXED EVENT EXPERIMENTS"
    )

    experiment_information = []

    for gap_length in gap_lengths:

        print()
        print("-" * 80)

        info(
            f"GAP EXPERIMENT: {gap_length} LP"
        )

        print("-" * 80)

        # ---------------------------------------------------------------------
        # Train + validation events
        # ---------------------------------------------------------------------

        train_validation_events = (
            generate_train_validation_events(
                dataframe=split_dataframe,
                config=CONFIG,
                gap_length=gap_length
            )
        )

        # ---------------------------------------------------------------------
        # Test events are COMMON
        # ---------------------------------------------------------------------

        event_starts = {

            "train":
                train_validation_events[
                    "train"
                ],

            "validation":
                train_validation_events[
                    "validation"
                ],

            "test":
                common_test_starts,
        }

        # ---------------------------------------------------------------------
        # Apply gaps
        # ---------------------------------------------------------------------

        masked_dataframe, metadata = apply_gap_events(
            dataframe=split_dataframe,
            event_starts=event_starts,
            gap_length=gap_length
        )

        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------

        validate_gap_experiment(
            original_dataframe=split_dataframe,
            masked_dataframe=masked_dataframe,
            metadata=metadata,
            gap_length=gap_length
        )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        save_gap_experiment(
            masked_dataframe=masked_dataframe,
            metadata=metadata,
            config=CONFIG,
            gap_length=gap_length
        )

        # ---------------------------------------------------------------------
        # Experiment summary
        # ---------------------------------------------------------------------

        train_count = int(
            (
                metadata[
                    "split"
                ] == "train"
            ).sum()
        )

        validation_count = int(
            (
                metadata[
                    "split"
                ] == "validation"
            ).sum()
        )

        test_count = int(
            (
                metadata[
                    "split"
                ] == "test"
            ).sum()
        )

        experiment_information.append({

            "gap_length":
                gap_length,

            "train_events":
                train_count,

            "validation_events":
                validation_count,

            "test_events":
                test_count,

            "total_events":
                len(metadata),

            "masked_lps":
                int(
                    len(metadata)
                    * gap_length
                ),

        })

    # =========================================================================
    # MANIFEST
    # =========================================================================

    section(
        "SAVING COMMON GAP MANIFEST"
    )

    create_manifest(
        config=CONFIG,
        dataframe=split_dataframe,
        experiment_information=experiment_information,
        common_test_starts=common_test_starts
    )

    # =========================================================================
    # SOURCE VERIFICATION
    # =========================================================================

    verify_source_unchanged(
        original_dataframe=dataframe,
        source_path=get_feature_dataframe_path(
            CONFIG
        )
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    section(
        "GAP GENERATION COMPLETE"
    )

    info(
        f"Meter           : "
        f"{get_config(CONFIG, 'data', 'selected_meter')}"
    )

    info(
        f"Direction       : "
        f"{get_config(CONFIG, 'data', 'target_direction')}"
    )

    info(
        f"Total LPs       : {len(dataframe):,}"
    )

    info(
        "Gap experiments :"
    )

    for gap_length in gap_lengths:

        info(
            f"    {gap_length} LP"
        )

    info(
        f"Common test locations : "
        f"{len(common_test_starts)}"
    )

    info(
        "Gap type        : interior"
    )

    info(
        "Context         : 96 LP left + 96 LP right"
    )

    info(
        "Formulation     : fixed event windows"
    )

    info(
        "Sliding windows : DISABLED"
    )

    info(
        "96 LP gap       : REMOVED"
    )

    info(
        "Source CSV      : UNCHANGED"
    )

    info(
        "Feature parquet : UNCHANGED"
    )

    info(
        "Model training  : NOT PERFORMED"
    )

    info(
        "Evaluation      : NOT PERFORMED"
    )

    info(
        "Normalization   : NOT PERFORMED"
    )

    info(
        "\nOutput directory:"
    )

    info(
        f"    {get_gap_output_directory(CONFIG)}"
    )

    info(
        "\nNEXT STAGE:"
    )

    info(
        "Build fixed-event supervised datasets."
    )

    info(
        "Each missing LP will eventually become "
        "one supervised prediction target."
    )

    info(
        "The 96 LP left/right context will be used "
        "to construct the predictor information."
    )

    info(
        "Do NOT train Random Forest yet."
    )

    info(
        "=" * 80
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()