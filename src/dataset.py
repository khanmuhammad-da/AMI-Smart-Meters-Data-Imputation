# =============================================================================
# src/dataset.py
# =============================================================================
"""
Random Forest CDP Load Profile Imputation
------------------------------------------

DATASET STAGE ONLY

Responsibilities
----------------
1. Load configuration.
2. Load the validated source CSV.
3. Validate source structure.
4. Discover A+/A- meter pairs.
5. Audit meter directionality.
6. Select one configured unidirectional meter.
7. Extract a clean single-meter dataframe.
8. Validate the selected dataframe.
9. Save the selected dataframe for downstream stages.

This module DOES NOT:
    - perform feature engineering
    - generate artificial gaps
    - create supervised training samples
    - train Random Forest
    - perform evaluation
    - normalize data
    - modify the source CSV

The source CSV is NEVER modified.

Expected source structure
-------------------------
Time
CDP_xxxxx_P_01_A+ [kWh]
CDP_xxxxx_P_01_A- [kWh]
...

Current experiment
------------------
Meter     : CDP_00526_P_01
Direction : A-

The selected meter must be genuinely unidirectional:

    A+ = zero
    A- = actual target

Output
------
data/processed/CDP_00526_P_01_A-_selected.parquet
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT /
    "config" /
    "config.yaml"
)


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def section(title: str) -> None:
    """Print a standard section heading."""

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def info(message: str) -> None:
    """Print an informational message."""

    print(message)


def fail(message: str) -> None:
    """Raise a descriptive pipeline error."""

    raise RuntimeError(message)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(
    config_path: Path = CONFIG_PATH
) -> dict:
    """
    Load config/config.yaml.
    """

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
            "Configuration file did not load "
            "as a dictionary."
        )

    return config


def get_required_config(
    config: dict,
    *keys: str
):
    """
    Retrieve a required nested configuration value.

    Example:

        get_required_config(
            config,
            "data",
            "selected_meter"
        )
    """

    current = config

    for key in keys:

        if not isinstance(
            current,
            dict
        ):

            fail(
                "Invalid configuration structure "
                f"while looking for: {'.'.join(keys)}"
            )

        if key not in current:

            fail(
                "Required configuration key missing: "
                f"{'.'.join(keys)}"
            )

        current = current[key]

    return current


# =============================================================================
# COLUMN NAME PARSING
# =============================================================================

METER_COLUMN_PATTERN = re.compile(
    r"^(?P<meter>.+)_"
    r"(?P<direction>A\+|A-)"
    r" \[kWh\]$"
)


def parse_meter_column(
    column_name: str
) -> Tuple[str, str] | None:
    """
    Parse a meter energy column.

    Example
    -------
    CDP_00526_P_01_A- [kWh]

    Returns
    -------
    ("CDP_00526_P_01", "A-")
    """

    match = METER_COLUMN_PATTERN.match(
        column_name
    )

    if match is None:
        return None

    return (
        match.group("meter"),
        match.group("direction")
    )


def discover_meter_columns(
    dataframe: pd.DataFrame
) -> Dict[str, Dict[str, str]]:
    """
    Discover all A+/A- columns.

    Returns
    -------
    dict

        {
            "CDP_00526_P_01": {
                "A+": "...",
                "A-": "..."
            }
        }
    """

    meter_columns: Dict[
        str,
        Dict[str, str]
    ] = {}

    for column in dataframe.columns:

        parsed = parse_meter_column(
            str(column)
        )

        if parsed is None:
            continue

        meter_id, direction = parsed

        if meter_id not in meter_columns:

            meter_columns[meter_id] = {}

        if direction in meter_columns[meter_id]:

            fail(
                "Duplicate direction column detected:\n"
                f"Meter     : {meter_id}\n"
                f"Direction : {direction}\n"
                f"Column    : {column}"
            )

        meter_columns[
            meter_id
        ][direction] = column

    return meter_columns


# =============================================================================
# SOURCE CSV LOADING
# =============================================================================

def load_source_dataframe(
    config: dict
) -> pd.DataFrame:
    """
    Load the source CSV.
    """

    input_file = Path(
        get_required_config(
            config,
            "data",
            "input_file"
        )
    )

    if not input_file.is_absolute():

        input_file = (
            PROJECT_ROOT /
            input_file
        )

    delimiter = get_required_config(
        config,
        "data",
        "delimiter"
    )

    timestamp_column = get_required_config(
        config,
        "data",
        "timestamp_column"
    )

    timestamp_format = get_required_config(
        config,
        "data",
        "timestamp_format"
    )

    expected_interval = int(
        get_required_config(
            config,
            "data",
            "expected_interval_minutes"
        )
    )

    if not input_file.exists():

        fail(
            "Input CSV not found:\n"
            f"    {input_file}"
        )

    section(
        "LOADING SOURCE DATA"
    )

    info(
        f"Input file : {input_file}"
    )

    info(
        f"Delimiter  : {delimiter!r}"
    )

    dataframe = pd.read_csv(
        input_file,
        sep=delimiter,
        low_memory=False
    )

    info(
        f"Rows       : {len(dataframe):,}"
    )

    info(
        f"Columns    : {len(dataframe.columns):,}"
    )

    # -------------------------------------------------------------------------
    # Timestamp existence
    # -------------------------------------------------------------------------

    if timestamp_column not in dataframe.columns:

        fail(
            "Timestamp column not found:\n"
            f"    {timestamp_column}"
        )

    # -------------------------------------------------------------------------
    # Timestamp parsing
    # -------------------------------------------------------------------------

    parsed_time = pd.to_datetime(
        dataframe[timestamp_column],
        format=timestamp_format,
        errors="coerce"
    )

    # Some datasets may already be interpreted correctly by pandas despite
    # formatting differences. If the configured format produces invalid
    # values, attempt strict general parsing before failing.
    if parsed_time.isna().any():

        fallback_time = pd.to_datetime(
            dataframe[timestamp_column],
            errors="coerce",
            dayfirst=True
        )

        if fallback_time.isna().any():

            bad_count = int(
                fallback_time.isna().sum()
            )

            fail(
                "Timestamp parsing failed.\n"
                f"Invalid timestamp values: {bad_count}"
            )

        parsed_time = fallback_time

    dataframe[timestamp_column] = parsed_time

    return dataframe


# =============================================================================
# SOURCE STRUCTURE VALIDATION
# =============================================================================

def validate_source_structure(
    dataframe: pd.DataFrame,
    config: dict
) -> Dict[str, Dict[str, str]]:
    """
    Validate source CSV structure and discover meters.
    """

    section(
        "SOURCE STRUCTURE VALIDATION"
    )

    timestamp_column = get_required_config(
        config,
        "data",
        "timestamp_column"
    )

    expected_columns = int(
        get_required_config(
            config,
            "dataset",
            "expected_total_columns"
        )
    )

    expected_cdp_count = int(
        get_required_config(
            config,
            "dataset",
            "expected_cdp_count"
        )
    )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    if timestamp_column not in dataframe.columns:

        fail(
            f"Timestamp column '{timestamp_column}' "
            "is missing."
        )

    info(
        "Timestamp column       : PASSED"
    )

    # -------------------------------------------------------------------------
    # Total columns
    # -------------------------------------------------------------------------

    actual_columns = len(
        dataframe.columns
    )

    info(
        f"Expected total columns : {expected_columns}"
    )

    info(
        f"Actual total columns   : {actual_columns}"
    )

    if actual_columns != expected_columns:

        fail(
            "Unexpected total number of columns.\n"
            f"Expected: {expected_columns}\n"
            f"Actual  : {actual_columns}"
        )

    info(
        "Total column count     : PASSED"
    )

    # -------------------------------------------------------------------------
    # Discover CDPs
    # -------------------------------------------------------------------------

    meter_columns = discover_meter_columns(
        dataframe
    )

    info(
        f"Discovered CDPs        : "
        f"{len(meter_columns)}"
    )

    if len(meter_columns) != expected_cdp_count:

        fail(
            "Unexpected number of CDPs.\n"
            f"Expected: {expected_cdp_count}\n"
            f"Actual  : {len(meter_columns)}"
        )

    info(
        "CDP count              : PASSED"
    )

    # -------------------------------------------------------------------------
    # Verify A+/A- pairs
    # -------------------------------------------------------------------------

    incomplete_pairs: List[str] = []

    for meter_id, directions in meter_columns.items():

        if set(directions.keys()) != {
            "A+",
            "A-"
        }:

            incomplete_pairs.append(
                meter_id
            )

    if incomplete_pairs:

        fail(
            "Incomplete A+/A- meter pairs found:\n"
            + "\n".join(
                incomplete_pairs
            )
        )

    info(
        "A+/A- pairing          : PASSED"
    )

    return meter_columns


# =============================================================================
# SOURCE ASSUMPTION VALIDATION
# =============================================================================

def validate_source_assumptions(
    dataframe: pd.DataFrame,
    meter_columns: Dict[str, Dict[str, str]],
    config: dict
) -> None:
    """
    Validate assumptions about the clean source data.

    Expected:
        - unique timestamps
        - chronological timestamps
        - exactly 30-minute interval
        - numeric meter values
        - no NaN
        - no Inf
    """

    section(
        "SOURCE DATA ASSUMPTION VALIDATION"
    )

    timestamp_column = get_required_config(
        config,
        "data",
        "timestamp_column"
    )

    expected_interval = int(
        get_required_config(
            config,
            "data",
            "expected_interval_minutes"
        )
    )

    # -------------------------------------------------------------------------
    # Timestamp uniqueness
    # -------------------------------------------------------------------------

    if dataframe[
        timestamp_column
    ].duplicated().any():

        duplicate_count = int(
            dataframe[
                timestamp_column
            ].duplicated().sum()
        )

        fail(
            "Duplicate timestamps detected.\n"
            f"Duplicate count: {duplicate_count}"
        )

    info(
        "Timestamp uniqueness     : PASSED"
    )

    # -------------------------------------------------------------------------
    # Timestamp ordering
    # -------------------------------------------------------------------------

    timestamps = dataframe[
        timestamp_column
    ]

    if not timestamps.is_monotonic_increasing:

        fail(
            "Timestamp column is not "
            "chronologically ordered."
        )

    info(
        "Timestamp chronological  : PASSED"
    )

    # -------------------------------------------------------------------------
    # Timestamp interval
    # -------------------------------------------------------------------------

    intervals = (
        timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    unique_intervals = np.unique(
        intervals.to_numpy()
    )

    if len(unique_intervals) != 1:

        fail(
            "Timestamp intervals are not uniform.\n"
            f"Detected intervals: {unique_intervals}"
        )

    actual_interval = float(
        unique_intervals[0]
    )

    if actual_interval != expected_interval:

        fail(
            "Unexpected timestamp interval.\n"
            f"Expected: {expected_interval} minutes\n"
            f"Actual  : {actual_interval} minutes"
        )

    info(
        f"Timestamp interval     : "
        f"{actual_interval:g} minutes — PASSED"
    )

    # -------------------------------------------------------------------------
    # Meter numeric validation
    # -------------------------------------------------------------------------

    all_meter_columns: List[str] = []

    for directions in meter_columns.values():

        all_meter_columns.extend(
            directions.values()
        )

    for column in all_meter_columns:

        numeric = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

        if numeric.isna().any():

            fail(
                "Non-numeric or missing meter "
                "values detected.\n"
                f"Column: {column}"
            )

        if not np.isfinite(
            numeric.to_numpy()
        ).all():

            fail(
                "Infinite meter value detected.\n"
                f"Column: {column}"
            )

    info(
        "Meter numeric values   : PASSED"
    )


# =============================================================================
# UNIDIRECTIONAL METER AUDIT
# =============================================================================

def audit_meter_directionality(
    dataframe: pd.DataFrame,
    meter_columns: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    """
    Determine whether each meter is:

        A+
        A-
        BIDIRECTIONAL_OR_MIXED
    """

    section(
        "UNIDIRECTIONAL METER AUDIT"
    )

    records = []

    for meter_id in sorted(
        meter_columns.keys()
    ):

        aplus_column = meter_columns[
            meter_id
        ]["A+"]

        aminus_column = meter_columns[
            meter_id
        ]["A-"]

        aplus = dataframe[
            aplus_column
        ].astype(float)

        aminus = dataframe[
            aminus_column
        ].astype(float)

        aplus_nonzero = int(
            (aplus != 0).sum()
        )

        aminus_nonzero = int(
            (aminus != 0).sum()
        )

        aplus_max = float(
            aplus.max()
        )

        aminus_max = float(
            aminus.max()
        )

        if (
            aplus_nonzero > 0
            and
            aminus_nonzero == 0
        ):

            direction = "A+"

        elif (
            aminus_nonzero > 0
            and
            aplus_nonzero == 0
        ):

            direction = "A-"

        else:

            direction = (
                "BIDIRECTIONAL_OR_MIXED"
            )

        records.append({

            "meter_id": meter_id,

            "unidirectional_direction":
                direction,

            "Aplus_nonzero_count":
                aplus_nonzero,

            "Aminus_nonzero_count":
                aminus_nonzero,

            "Aplus_max":
                aplus_max,

            "Aminus_max":
                aminus_max

        })

    audit_df = pd.DataFrame(
        records
    )

    direction_counts = (
        audit_df[
            "unidirectional_direction"
        ]
        .value_counts()
    )

    for direction in [
        "A+",
        "A-",
        "BIDIRECTIONAL_OR_MIXED"
    ]:

        count = int(
            direction_counts.get(
                direction,
                0
            )
        )

        info(
            f"  {direction:<23}: {count}"
        )

    print()

    candidates = audit_df[
        audit_df[
            "unidirectional_direction"
        ].isin(
            ["A+", "A-"]
        )
    ]

    if len(candidates) > 0:

        print(
            "Unidirectional meter candidates:"
        )

        print(
            candidates.to_string(
                index=False
            )
        )

    else:

        info(
            "No unidirectional meters found."
        )

    return audit_df


# =============================================================================
# SAVE METER AUDIT
# =============================================================================

def save_meter_audit(
    audit_df: pd.DataFrame,
    config: dict
) -> Path:
    """
    Save meter direction audit.
    """

    processed_dir = Path(
        get_required_config(
            config,
            "outputs",
            "processed_data_dir"
        )
    )

    if not processed_dir.is_absolute():

        processed_dir = (
            PROJECT_ROOT /
            processed_dir
        )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        processed_dir /
        "meter_direction_audit.csv"
    )

    audit_df.to_csv(
        output_file,
        index=False
    )

    info(
        "Meter audit saved:\n"
        f"    {output_file}"
    )

    return output_file


# =============================================================================
# SELECT SINGLE METER
# =============================================================================

def select_meter(
    dataframe: pd.DataFrame,
    meter_columns: Dict[str, Dict[str, str]],
    audit_df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Select and validate the configured meter.
    """

    section(
        "SINGLE METER SELECTION"
    )

    meter_id = get_required_config(
        config,
        "data",
        "selected_meter"
    )

    target_direction = get_required_config(
        config,
        "data",
        "target_direction"
    )

    require_unidirectional = bool(
        get_required_config(
            config,
            "dataset",
            "require_unidirectional_meter"
        )
    )

    # -------------------------------------------------------------------------
    # Meter exists
    # -------------------------------------------------------------------------

    if meter_id not in meter_columns:

        fail(
            f"Configured meter '{meter_id}' "
            "was not discovered in the source CSV."
        )

    # -------------------------------------------------------------------------
    # Direction is valid
    # -------------------------------------------------------------------------

    if target_direction not in {
        "A+",
        "A-"
    }:

        fail(
            "target_direction must be "
            "'A+' or 'A-'.\n"
            f"Configured: {target_direction}"
        )

    # -------------------------------------------------------------------------
    # Check directionality
    # -------------------------------------------------------------------------

    audit_row = audit_df[
        audit_df["meter_id"] == meter_id
    ]

    if audit_row.empty:

        fail(
            f"No directionality audit found "
            f"for meter '{meter_id}'."
        )

    actual_direction = (
        audit_row.iloc[0][
            "unidirectional_direction"
        ]
    )

    if require_unidirectional:

        if actual_direction != target_direction:

            fail(
                f"Configured meter '{meter_id}' "
                "is not unidirectional in the "
                "requested direction.\n"
                f"Requested: {target_direction}\n"
                f"Detected : {actual_direction}"
            )

    # -------------------------------------------------------------------------
    # Column names
    # -------------------------------------------------------------------------

    target_column = meter_columns[
        meter_id
    ][target_direction]

    opposite_direction = (
        "A-"
        if target_direction == "A+"
        else "A+"
    )

    opposite_column = meter_columns[
        meter_id
    ][opposite_direction]

    timestamp_column = get_required_config(
        config,
        "data",
        "timestamp_column"
    )

    # -------------------------------------------------------------------------
    # Confirm expected source columns
    # -------------------------------------------------------------------------

    expected_target_column = (
        f"{meter_id}_{target_direction} [kWh]"
    )

    expected_opposite_column = (
        f"{meter_id}_{opposite_direction} [kWh]"
    )

    if target_column != expected_target_column:

        fail(
            "Discovered target column does not "
            "match expected structure.\n"
            f"Discovered: {target_column}\n"
            f"Expected  : {expected_target_column}"
        )

    if opposite_column != expected_opposite_column:

        fail(
            "Discovered opposite column does not "
            "match expected structure.\n"
            f"Discovered: {opposite_column}\n"
            f"Expected  : {expected_opposite_column}"
        )

    # -------------------------------------------------------------------------
    # Create selected dataframe
    # -------------------------------------------------------------------------

    selected_df = pd.DataFrame({

        "Time":
            dataframe[
                timestamp_column
            ].copy(),

        "target":
            pd.to_numeric(
                dataframe[
                    target_column
                ],
                errors="raise"
            ).astype(float),

        "opposite":
            pd.to_numeric(
                dataframe[
                    opposite_column
                ],
                errors="raise"
            ).astype(float),

        "meter_id":
            meter_id,

        "target_direction":
            target_direction

    })

    print(
        f"Selected meter       : {meter_id}"
    )

    print(
        f"Target direction     : "
        f"{target_direction}"
    )

    print(
        f"Target column        : "
        f"{target_column}"
    )

    print(
        f"Opposite column      : "
        f"{opposite_column}"
    )

    print(
        f"Rows                 : "
        f"{len(selected_df):,}"
    )

    return selected_df


# =============================================================================
# VALIDATE SELECTED METER
# =============================================================================

def validate_selected_meter(
    selected_df: pd.DataFrame,
    config: dict
) -> None:
    """
    Validate the single-meter dataframe.
    """

    section(
        "SELECTED METER VALIDATION"
    )

    required_columns = [
        "Time",
        "target",
        "opposite",
        "meter_id",
        "target_direction"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in selected_df.columns
    ]

    if missing_columns:

        fail(
            "Selected dataframe is missing columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(
        selected_df["Time"]
    ):

        fail(
            "Selected Time column is not datetime."
        )

    # -------------------------------------------------------------------------
    # Timestamp uniqueness
    # -------------------------------------------------------------------------

    if selected_df[
        "Time"
    ].duplicated().any():

        fail(
            "Selected dataframe contains "
            "duplicate timestamps."
        )

    # -------------------------------------------------------------------------
    # Timestamp order
    # -------------------------------------------------------------------------

    if not selected_df[
        "Time"
    ].is_monotonic_increasing:

        fail(
            "Selected dataframe is not "
            "chronologically ordered."
        )

    # -------------------------------------------------------------------------
    # Target numerical validation
    # -------------------------------------------------------------------------

    target = selected_df[
        "target"
    ].to_numpy(
        dtype=float
    )

    opposite = selected_df[
        "opposite"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        target
    ).all():

        fail(
            "Selected target contains NaN "
            "or Inf."
        )

    if not np.isfinite(
        opposite
    ).all():

        fail(
            "Selected opposite direction "
            "contains NaN or Inf."
        )

    # -------------------------------------------------------------------------
    # Selected meter consistency
    # -------------------------------------------------------------------------

    selected_meter = get_required_config(
        config,
        "data",
        "selected_meter"
    )

    target_direction = get_required_config(
        config,
        "data",
        "target_direction"
    )

    if (
        selected_df["meter_id"]
        .nunique()
        != 1
    ):

        fail(
            "Selected dataframe contains "
            "more than one meter."
        )

    if selected_df[
        "meter_id"
    ].iloc[0] != selected_meter:

        fail(
            "Selected meter ID mismatch."
        )

    if (
        selected_df["target_direction"]
        .nunique()
        != 1
    ):

        fail(
            "Selected dataframe contains "
            "more than one direction."
        )

    if selected_df[
        "target_direction"
    ].iloc[0] != target_direction:

        fail(
            "Selected target direction mismatch."
        )

    # -------------------------------------------------------------------------
    # Unidirectional validation
    # -------------------------------------------------------------------------

    opposite_nonzero = int(
        np.count_nonzero(
            opposite
        )
    )

    target_nonzero = int(
        np.count_nonzero(
            target
        )
    )

    target_zero = int(
        np.sum(
            target == 0
        )
    )

    target_min = float(
        target.min()
    )

    target_max = float(
        target.max()
    )

    target_mean = float(
        target.mean()
    )

    target_std = float(
        target.std()
    )

    print(
        f"Target observations     : "
        f"{len(target):,}"
    )

    print(
        f"Target non-zero        : "
        f"{target_nonzero:,}"
    )

    print(
        f"Target zero            : "
        f"{target_zero:,}"
    )

    print(
        f"Opposite non-zero      : "
        f"{opposite_nonzero:,}"
    )

    print(
        f"Target minimum         : "
        f"{target_min:.6f}"
    )

    print(
        f"Target maximum         : "
        f"{target_max:.6f}"
    )

    print(
        f"Target mean            : "
        f"{target_mean:.6f}"
    )

    print(
        f"Target std             : "
        f"{target_std:.6f}"
    )

    if require_unidirectional := bool(
        get_required_config(
            config,
            "dataset",
            "require_unidirectional_meter"
        )
    ):

        if opposite_nonzero != 0:

            fail(
                "Selected meter failed "
                "unidirectional validation.\n"
                f"Opposite non-zero values: "
                f"{opposite_nonzero}"
            )

    print(
        "Unidirectional check   : PASSED"
    )

    print(
        "Selected meter check   : PASSED"
    )


# =============================================================================
# SAVE SELECTED DATAFRAME
# =============================================================================

def save_selected_meter(
    selected_df: pd.DataFrame,
    config: dict
) -> Tuple[Path, Path]:
    """
    Save selected meter dataframe and metadata.
    """

    section(
        "SAVING SINGLE-METER DATAFRAME"
    )

    processed_dir = Path(
        get_required_config(
            config,
            "outputs",
            "processed_data_dir"
        )
    )

    if not processed_dir.is_absolute():

        processed_dir = (
            PROJECT_ROOT /
            processed_dir
        )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    meter_id = selected_df[
        "meter_id"
    ].iloc[0]

    direction = selected_df[
        "target_direction"
    ].iloc[0]

    # Replace + because it can be awkward in filenames.
    direction_filename = (
        "plus"
        if direction == "A+"
        else "minus"
    )

    dataframe_file = (
        processed_dir /
        f"{meter_id}_A-{'' if direction == 'A-' else '+'}_selected.parquet"
    )

    # The filename above is intentionally normalized below to preserve
    # the familiar A-/A+ naming convention.
    if direction == "A-":

        dataframe_file = (
            processed_dir /
            f"{meter_id}_A-_selected.parquet"
        )

    else:

        dataframe_file = (
            processed_dir /
            f"{meter_id}_A+_selected.parquet"
        )

    selected_df.to_parquet(
        dataframe_file,
        index=False,
        engine="pyarrow"
    )

    metadata = {

        "meter_id":
            meter_id,

        "target_direction":
            direction,

        "rows":
            int(len(selected_df)),

        "columns":
            list(selected_df.columns),

        "start_time":
            str(
                selected_df[
                    "Time"
                ].min()
            ),

        "end_time":
            str(
                selected_df[
                    "Time"
                ].max()
            ),

        "target_column":
            f"{meter_id}_{direction} [kWh]",

        "opposite_column":
            (
                f"{meter_id}_A+ [kWh]"
                if direction == "A-"
                else
                f"{meter_id}_A- [kWh]"
            ),

        "target_nonzero_count":
            int(
                np.count_nonzero(
                    selected_df[
                        "target"
                    ].to_numpy()
                )
            ),

        "target_zero_count":
            int(
                (
                    selected_df[
                        "target"
                    ] == 0
                ).sum()
            ),

        "opposite_nonzero_count":
            int(
                np.count_nonzero(
                    selected_df[
                        "opposite"
                    ].to_numpy()
                )
            ),

        "source_modified":
            False

    }

    metadata_file = (
        processed_dir /
        "selected_meter_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    info(
        "Saved dataframe:\n"
        f"    {dataframe_file}"
    )

    info(
        f"Shape: {selected_df.shape}"
    )

    info(
        "Selection metadata saved:\n"
        f"    {metadata_file}"
    )

    return (
        dataframe_file,
        metadata_file
    )


# =============================================================================
# FINAL VALIDATION OF SAVED DATAFRAME
# =============================================================================

def verify_saved_dataframe(
    selected_df: pd.DataFrame,
    dataframe_file: Path
) -> None:
    """
    Reload the saved Parquet and verify it matches the dataframe in memory.
    """

    section(
        "SAVED DATAFRAME VERIFICATION"
    )

    if not dataframe_file.exists():

        fail(
            "Saved dataframe does not exist:\n"
            f"    {dataframe_file}"
        )

    reloaded = pd.read_parquet(
        dataframe_file,
        engine="pyarrow"
    )

    if reloaded.shape != selected_df.shape:

        fail(
            "Reloaded dataframe shape mismatch.\n"
            f"Original: {selected_df.shape}\n"
            f"Reloaded: {reloaded.shape}"
        )

    # Timestamp
    if not selected_df[
        "Time"
    ].equals(
        reloaded["Time"]
    ):

        fail(
            "Reloaded Time column does not "
            "match original."
        )

    # Numeric columns
    for column in [
        "target",
        "opposite"
    ]:

        if not np.array_equal(
            selected_df[
                column
            ].to_numpy(),
            reloaded[
                column
            ].to_numpy()
        ):

            fail(
                f"Reloaded column '{column}' "
                "does not match original."
            )

    # Metadata columns
    for column in [
        "meter_id",
        "target_direction"
    ]:

        if not selected_df[
            column
        ].equals(
            reloaded[column]
        ):

            fail(
                f"Reloaded column '{column}' "
                "does not match original."
            )

    info(
        "Parquet reload          : PASSED"
    )

    info(
        f"Verified shape          : "
        f"{reloaded.shape}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Execute dataset stage.
    """

    print(
        "=" * 80
    )

    print(
        "RANDOM FOREST CDP IMPUTATION — "
        "DATASET STAGE"
    )

    print(
        "=" * 80
    )

    print(
        "Stage: source CSV → single validated meter"
    )

    print(
        "Feature engineering: DISABLED"
    )

    print(
        "Gap simulation: DISABLED"
    )

    print(
        "Supervised dataset creation: DISABLED"
    )

    print(
        "Model training: DISABLED"
    )

    print(
        "Evaluation: DISABLED"
    )

    print(
        "Normalization: DISABLED"
    )

    print(
        "Source modification: DISABLED"
    )

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------

    config = load_config()

    # -------------------------------------------------------------------------
    # Load source
    # -------------------------------------------------------------------------

    dataframe = load_source_dataframe(
        config
    )

    # -------------------------------------------------------------------------
    # Structure validation
    # -------------------------------------------------------------------------

    meter_columns = (
        validate_source_structure(
            dataframe,
            config
        )
    )

    # -------------------------------------------------------------------------
    # Source assumptions
    # -------------------------------------------------------------------------

    validate_source_assumptions(
        dataframe,
        meter_columns,
        config
    )

    # -------------------------------------------------------------------------
    # Directionality audit
    # -------------------------------------------------------------------------

    audit_df = (
        audit_meter_directionality(
            dataframe,
            meter_columns
        )
    )

    save_meter_audit(
        audit_df,
        config
    )

    # -------------------------------------------------------------------------
    # Select meter
    # -------------------------------------------------------------------------

    selected_df = select_meter(
        dataframe,
        meter_columns,
        audit_df,
        config
    )

    # -------------------------------------------------------------------------
    # Validate selected meter
    # -------------------------------------------------------------------------

    validate_selected_meter(
        selected_df,
        config
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    (
        dataframe_file,
        metadata_file
    ) = save_selected_meter(
        selected_df,
        config
    )

    # -------------------------------------------------------------------------
    # Reload verification
    # -------------------------------------------------------------------------

    verify_saved_dataframe(
        selected_df,
        dataframe_file
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    section(
        "DATASET STAGE COMPLETE"
    )

    print(
        "Selected meter : "
        f"{selected_df['meter_id'].iloc[0]}"
    )

    print(
        "Direction      : "
        f"{selected_df['target_direction'].iloc[0]}"
    )

    print(
        "Rows           : "
        f"{len(selected_df):,}"
    )

    print(
        "Start          : "
        f"{selected_df['Time'].min()}"
    )

    print(
        "End            : "
        f"{selected_df['Time'].max()}"
    )

    print(
        "Output         :"
    )

    print(
        f"    {dataframe_file}"
    )

    print()

    print(
        "NEXT STAGE:"
    )

    print(
        "Feature engineering will be performed "
        "separately."
    )

    print(
        "No target lags or target leads will be "
        "created."
    )

    print(
        "No gaps have been generated."
    )

    print(
        "No supervised samples have been created."
    )

    print(
        "Random Forest has NOT been trained."
    )

    print(
        "The source CSV remains unchanged."
    )

    print(
        "=" * 80
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()