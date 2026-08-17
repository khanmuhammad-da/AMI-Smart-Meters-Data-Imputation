"""
src/inference.py
================

Production-style inference for the V1 tuned Random Forest CDP imputation
pipeline.

Supported gap lengths
---------------------
1, 6, 24 and 48 half-hour LPs.

Training-time artifacts are used; CSV/XLSX input and JMR are NOT used during
training. They are used only here, at inference time.

Inference workflow
------------------
1. Read CSV/XLSX.
2. Identify Time and load-profile columns.
3. Detect contiguous missing LP gaps.
4. Validate that every gap is 1/6/24/48 LP.
5. Build the exact 28 V1 features used by the tuned models:
      - 22 calendar/cyclic features
      - 6 boundary/context features
6. Select the corresponding tuned model for each gap.
7. Predict only missing LPs.
8. Reconcile ONLY predicted/missing values proportionally to the supplied JMR.
9. Preserve all observed values exactly.
10. Write CSV + XLSX + PNG plot + JSON summary.

Example
-------
python -m src.inference --input "data/input/test_meter.xlsx" --jmr 69190000

Optional output directory
-------------------------
python -m src.inference ^
    --input "data/input/test_meter.xlsx" ^
    --jmr 69190000 ^
    --output-dir "outputs/inference"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

# Matplotlib is used only for the final diagnostic plot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = PROJECT_ROOT / "outputs" / "model_tuned" / "models"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "inference"

GAP_LENGTHS = (1, 6, 24, 48)
CONTEXT_LP = 96
LP_MINUTES = 30

# Exact V1 predictor feature order used by the tuned models.
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
    "target_left_1",
    "target_left_2",
    "target_left_24",
    "target_right_1",
    "target_right_2",
    "target_right_24",
]

# Common timestamp names.
TIME_CANDIDATES = [
    "Time",
    "time",
    "Timestamp",
    "timestamp",
    "DateTime",
    "Datetime",
    "datetime",
    "Date",
    "date",
]

# Standard load-profile names.
LP_CANDIDATES = [
    "LP",
    "lp",
    "Load",
    "load",
    "Load Profile",
    "load_profile",
    "Energy",
    "energy",
    "Value",
    "value",
]

# Common representations of missing cells in CSV/Excel exports.
MISSING_TOKENS = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "NULL",
    "none",
    "None",
    "NA",
    "N/A",
    "-",
    "--",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Gap:
    gap_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def target_indices(self) -> range:
        return range(self.start, self.end + 1)


# =============================================================================
# PRINTING / ERROR HANDLING
# =============================================================================

def line(char: str = "-", n: int = 80) -> str:
    return char * n


def info(message: str) -> None:
    print(message)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# INPUT READING
# =============================================================================

def read_input_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel input without changing the user's original file."""
    if not path.exists():
        fail(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    print()
    print(line("-", 80))
    print("READING INPUT FILE")
    print(line("-", 80))
    print(f"File       : {path}")
    print(f"Extension  : {suffix}")

    if suffix == ".csv":
        # utf-8-sig handles common Excel-generated CSVs with a BOM.
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")
    elif suffix in {".xlsx", ".xlsm"}:
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            fail(
                "Excel input requires openpyxl. Install it with:\n"
                "python -m pip install openpyxl"
            )
    elif suffix == ".xls":
        fail(
            "Legacy .xls input is not enabled. Convert the file to .xlsx or CSV, "
            "or install an appropriate legacy Excel reader."
        )
    else:
        fail(
            f"Unsupported input extension: {suffix}. "
            "Use .csv, .xlsx or .xlsm."
        )

    if df.empty:
        fail("Input file contains no rows.")

    # Strip accidental whitespace from headers while preserving their meaning.
    df.columns = [str(c).strip() for c in df.columns]

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Columns    : {df.columns.tolist()}")

    return df


def identify_time_column(df: pd.DataFrame) -> str:
    """Find timestamp column."""
    for candidate in TIME_CANDIDATES:
        if candidate in df.columns:
            return candidate

    # Conservative fallback: inspect column names.
    candidates = [
        c for c in df.columns
        if re.search(r"(time|timestamp|datetime|date)", str(c), re.I)
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        fail(
            "Could not identify timestamp column.\n"
            f"Expected one of: {TIME_CANDIDATES}\n"
            f"Found: {df.columns.tolist()}"
        )

    fail(
        "Multiple possible timestamp columns were found. "
        f"Please keep exactly one timestamp column.\nCandidates: {candidates}"
    )


def identify_lp_column(df: pd.DataFrame, time_column: str) -> str:
    """
    Identify the meter load-profile column.

    First uses standard names. If none match, it supports real meter register
    names such as 'A- [kWh]' by looking for a single kWh/energy-like column.
    """
    for candidate in LP_CANDIDATES:
        if candidate in df.columns and candidate != time_column:
            return candidate

    # Case-insensitive exact matching.
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in LP_CANDIDATES:
        found = lower_map.get(candidate.lower())
        if found is not None and found != time_column:
            return found

    # Robust fallback for real AMI exports such as:
    # A- [kWh], A+ [kWh], Active Energy [kWh], etc.
    non_time = [c for c in df.columns if c != time_column]

    kwh_candidates = [
        c for c in non_time
        if re.search(r"\bkwh\b", str(c), re.I)
    ]

    if len(kwh_candidates) == 1:
        return kwh_candidates[0]

    # A slightly broader energy-name fallback, still requiring uniqueness.
    energy_candidates = [
        c for c in non_time
        if re.search(r"(energy|load|active|a[-+])", str(c), re.I)
    ]

    if len(energy_candidates) == 1:
        return energy_candidates[0]

    fail(
        "Could not identify load-profile column.\n"
        f"Expected one of: {LP_CANDIDATES}\n"
        f"Found: {df.columns.tolist()}\n\n"
        "For meter exports such as 'A- [kWh]', the inference code supports "
        "a unique column containing 'kWh'. If multiple kWh columns exist, "
        "keep only the register to be imputed or rename/select it explicitly."
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def prepare_input(
    df: pd.DataFrame,
    time_column: str,
    lp_column: str,
) -> pd.DataFrame:
    """Create an internal normalized representation."""
    work = df.copy()

    # AMI exports commonly use DD/MM/YYYY HH:MM:SS.  Pandas versions with
    # strict datetime parsing can fail on a mixed-format column when called
    # without format="mixed".  Use an explicit day-first mixed parser so that
    # values such as "01/06/2026 00:00:00" are interpreted as 1 June 2026.
    print("Parsing timestamps...")
    print("Timestamp convention  : DD/MM/YYYY")

    raw_time = work[time_column]

    if pd.api.types.is_datetime64_any_dtype(raw_time):
        parsed_time = pd.to_datetime(raw_time, errors="coerce")
    else:
        try:
            parsed_time = pd.to_datetime(
                raw_time,
                errors="coerce",
                format="mixed",
                dayfirst=True,
            )
        except TypeError:
            # Compatibility fallback for older pandas versions that do not
            # support format="mixed".
            parsed_time = pd.to_datetime(
                raw_time,
                errors="coerce",
                dayfirst=True,
            )

    if parsed_time.isna().any():
        bad_mask = parsed_time.isna()
        bad = int(bad_mask.sum())
        examples = raw_time.loc[bad_mask].astype(str).head(10).tolist()
        fail(
            f"Timestamp parsing failed for {bad} row(s).\n"
            f"Timestamp parser      : mixed + dayfirst=True\n"
            f"Examples of unparsed values: {examples}"
        )

    print("Timestamp parser      : mixed + dayfirst=True")
    print(f"Timestamp parsing    : PASSED ({len(parsed_time):,} rows)")

    # Preserve original row order separately. The model needs chronological
    # ordering, but the final output should remain in the user's input order.
    work["_original_row_order"] = np.arange(len(work), dtype=int)
    work["_parsed_time"] = parsed_time

    # Convert load profile to numeric. Text missing tokens become NaN.
    raw = work[lp_column].copy()
    if raw.dtype == object:
        raw = raw.replace(list(MISSING_TOKENS), np.nan)

    work["_lp_value"] = pd.to_numeric(raw, errors="coerce")

    # Do not silently interpret invalid non-missing text as missing.
    original_nonmissing = raw.notna()
    invalid = original_nonmissing & work["_lp_value"].isna()

    if invalid.any():
        examples = raw[invalid].astype(str).head(5).tolist()
        fail(
            "The load-profile column contains non-numeric non-missing values.\n"
            f"Examples: {examples}"
        )

    # Chronological model view.
    work = work.sort_values("_parsed_time").reset_index(drop=True)

    if work["_parsed_time"].duplicated().any():
        duplicates = int(work["_parsed_time"].duplicated().sum())
        fail(f"Duplicate timestamps found: {duplicates}")

    # Require exactly 30-minute intervals.
    deltas = work["_parsed_time"].diff().dropna()
    expected = pd.Timedelta(value=LP_MINUTES, unit="min")

    if not deltas.eq(expected).all():
        bad = int((~deltas.eq(expected)).sum())
        examples = deltas[~deltas.eq(expected)].head(5).tolist()
        fail(
            "Timestamp grid is not continuous at 30-minute intervals.\n"
            f"Unexpected interval count: {bad}\n"
            f"Examples: {examples}"
        )

    if len(work) < (2 * CONTEXT_LP + 1):
        fail(
            f"Input contains only {len(work)} rows. At least "
            f"{2 * CONTEXT_LP + 1} rows are required to safely construct "
            "the 96-LP left/right context."
        )

    return work


# =============================================================================
# GAP DETECTION
# =============================================================================

def detect_gaps(values: pd.Series) -> list[Gap]:
    """Detect contiguous NaN blocks."""
    missing = values.isna().to_numpy(dtype=bool)

    gaps: list[Gap] = []
    i = 0
    gap_id = 1

    while i < len(missing):
        if not missing[i]:
            i += 1
            continue

        start = i
        while i + 1 < len(missing) and missing[i + 1]:
            i += 1

        end = i
        gaps.append(Gap(gap_id=gap_id, start=start, end=end))
        gap_id += 1
        i += 1

    return gaps


def validate_gaps(gaps: list[Gap], n_rows: int) -> None:
    if not gaps:
        fail("No missing LP values were detected.")

    unsupported = [g for g in gaps if g.length not in GAP_LENGTHS]

    if unsupported:
        details = ", ".join(
            f"Gap {g.gap_id}={g.length} LP"
            for g in unsupported
        )
        fail(
            "Unsupported missing-gap length detected.\n"
            f"{details}\n\n"
            f"Currently supported gap lengths: {list(GAP_LENGTHS)}"
        )

    for gap in gaps:
        if gap.start < CONTEXT_LP:
            fail(
                f"Gap {gap.gap_id} starts at row {gap.start}, but "
                f"{CONTEXT_LP} left-context LPs are required."
            )

        if gap.end + CONTEXT_LP >= n_rows:
            fail(
                f"Gap {gap.gap_id} ends at row {gap.end}, but "
                f"{CONTEXT_LP} right-context LPs are required."
            )


# =============================================================================
# V1 FEATURE ENGINEERING
# =============================================================================

def _bool_float(value: bool) -> float:
    return float(bool(value))


def calendar_features(timestamp: pd.Timestamp) -> dict[str, float]:
    """Create the exact 22 calendar/cyclic V1 features."""
    hour = int(timestamp.hour)
    minute = int(timestamp.minute)

    half_hour_slot = hour * 2 + (1 if minute >= 30 else 0)
    day_of_week = int(timestamp.dayofweek)
    day_of_month = int(timestamp.day)
    day_of_year = int(timestamp.dayofyear)
    week_of_year = int(timestamp.isocalendar().week)
    month = int(timestamp.month)
    quarter = int(timestamp.quarter)

    # These match the project V1 convention observed in the supervised data:
    # day-of-year cyclic encoding uses (day_of_year - 1) / 365.
    half_hour_angle = 2.0 * math.pi * half_hour_slot / 48.0
    hour_angle = 2.0 * math.pi * hour / 24.0
    dow_angle = 2.0 * math.pi * day_of_week / 7.0
    doy_angle = 2.0 * math.pi * (day_of_year - 1) / 365.0

    return {
        "hour": float(hour),
        "minute": float(minute),
        "half_hour_slot": float(half_hour_slot),
        "day_of_week": float(day_of_week),
        "day_of_month": float(day_of_month),
        "day_of_year": float(day_of_year),
        "week_of_year": float(week_of_year),
        "month": float(month),
        "quarter": float(quarter),
        "is_weekend": _bool_float(day_of_week >= 5),
        "is_month_start": _bool_float(timestamp.is_month_start),
        "is_month_end": _bool_float(timestamp.is_month_end),
        "is_day_start": _bool_float(hour == 0 and minute == 0),
        "is_day_end": _bool_float(hour == 23 and minute == 30),
        "half_hour_sin": math.sin(half_hour_angle),
        "half_hour_cos": math.cos(half_hour_angle),
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "day_of_week_sin": math.sin(dow_angle),
        "day_of_week_cos": math.cos(dow_angle),
        "day_of_year_sin": math.sin(doy_angle),
        "day_of_year_cos": math.cos(doy_angle),
    }


def build_feature_rows(
    work: pd.DataFrame,
    gap: Gap,
) -> pd.DataFrame:
    """
    Build one feature row per missing LP.

    IMPORTANT:
    The six context features are anchored to the gap boundaries, not to the
    current target position. This prevents the current gap from becoming a
    source of features and matches the fixed-event V1 supervised formulation.
    """
    values = work["_lp_value"].to_numpy(dtype=float)
    times = work["_parsed_time"]

    left_1_idx = gap.start - 1
    left_2_idx = gap.start - 2
    left_24_idx = gap.start - 24

    right_1_idx = gap.end + 1
    right_2_idx = gap.end + 2
    right_24_idx = gap.end + 24

    context_indices = [
        left_1_idx,
        left_2_idx,
        left_24_idx,
        right_1_idx,
        right_2_idx,
        right_24_idx,
    ]

    for idx in context_indices:
        if idx < 0 or idx >= len(values):
            fail(
                f"Gap {gap.gap_id}: context index {idx} is outside the "
                "input range."
            )
        if np.isnan(values[idx]):
            fail(
                f"Gap {gap.gap_id}: required observed context value at "
                f"row {idx} is missing."
            )

    boundary_features = {
        "target_left_1": values[left_1_idx],
        "target_left_2": values[left_2_idx],
        "target_left_24": values[left_24_idx],
        "target_right_1": values[right_1_idx],
        "target_right_2": values[right_2_idx],
        "target_right_24": values[right_24_idx],
    }

    rows = []

    for target_index in gap.target_indices:
        row = calendar_features(pd.Timestamp(times.iloc[target_index]))
        row.update(boundary_features)
        rows.append(row)

    features = pd.DataFrame(rows, columns=V1_FEATURES)

    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        fail(f"Generated V1 features contain NaN: {bad}")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        fail("Generated V1 features contain non-finite values.")

    return features


# =============================================================================
# MODEL LOADING
# =============================================================================

def model_path(gap_length: int) -> Path:
    return MODEL_ROOT / f"random_forest_v1_tuned_gap_{gap_length}.joblib"


def load_model(gap_length: int):
    path = model_path(gap_length)

    if not path.exists():
        fail(
            f"Tuned model for gap {gap_length} LP was not found:\n"
            f"{path}\n\n"
            "Run the V1 tuned training stage before inference."
        )

    model = joblib.load(path)

    if not hasattr(model, "predict"):
        fail(f"Loaded object is not a prediction model: {path}")

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(V1_FEATURES):
        fail(
            f"Model {path.name} expects {n_features} features, but the "
            f"inference feature layer creates {len(V1_FEATURES)}."
        )

    return model


# =============================================================================
# PREDICTION
# =============================================================================

def predict_gaps(
    work: pd.DataFrame,
    gaps: list[Gap],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Predict all missing cells.

    Returns
    -------
    work:
        Chronological working dataframe with raw predictions added.
    gap_records:
        Per-gap audit records.
    """
    work = work.copy()
    work["_raw_prediction"] = np.nan
    work["_model_gap_length"] = np.nan

    gap_records: list[dict] = []

    model_cache = {}

    for gap in gaps:
        print()
        print(line("-", 80))
        print(f"MODEL INFERENCE — GAP {gap.gap_id} ({gap.length} LP)")
        print(line("-", 80))

        if gap.length not in model_cache:
            print(f"Loading tuned model for {gap.length} LP...")
            model_cache[gap.length] = load_model(gap.length)

        model = model_cache[gap.length]
        features = build_feature_rows(work, gap)

        predictions = np.asarray(model.predict(features), dtype=float)

        if len(predictions) != gap.length:
            fail(
                f"Gap {gap.gap_id}: model returned {len(predictions)} "
                f"predictions for {gap.length} missing LPs."
            )

        if not np.isfinite(predictions).all():
            fail(f"Gap {gap.gap_id}: model produced NaN/Inf predictions.")

        # A load profile cannot physically be negative.
        predictions = np.maximum(predictions, 0.0)

        for row_index, prediction in zip(gap.target_indices, predictions):
            work.loc[row_index, "_raw_prediction"] = float(prediction)
            work.loc[row_index, "_model_gap_length"] = float(gap.length)

        gap_records.append(
            {
                "gap_id": gap.gap_id,
                "gap_length_lp": gap.length,
                "start_index": gap.start,
                "end_index": gap.end,
                "start_time": work.loc[gap.start, "_parsed_time"].isoformat(),
                "end_time": work.loc[gap.end, "_parsed_time"].isoformat(),
                "raw_prediction_sum_kwh": float(predictions.sum()),
                "raw_prediction_min_kwh": float(predictions.min()),
                "raw_prediction_max_kwh": float(predictions.max()),
                "model_file": str(model_path(gap.length)),
            }
        )

        print(f"Model                 : RandomForestRegressor")
        print(f"Predictions            : {len(predictions)}")
        print(f"Raw prediction sum    : {predictions.sum():,.0f} kWh")
        print(f"Raw prediction min    : {predictions.min():,.0f} kWh")
        print(f"Raw prediction max    : {predictions.max():,.0f} kWh")

    return work, gap_records


# =============================================================================
# JMR RECONCILIATION
# =============================================================================

def reconcile_to_jmr(
    work: pd.DataFrame,
    gaps: list[Gap],
    jmr_kwh: float,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """
    Reconcile ONLY the predicted/missing values.

    Reconciliation method
    ---------------------
    observed_total = sum(observed LPs)
    raw_predicted_total = sum(raw model predictions)
    remaining_for_missing = JMR - observed_total
    scale_factor = remaining_for_missing / raw_predicted_total
    scaled_prediction = raw_prediction * scale_factor

    The final missing-LP values are rounded to whole kWh.  Because rounding
    can introduce a small residual, +/-1 kWh corrections are distributed over
    the missing LPs so that the final total equals the supplied JMR exactly.

    Observed values are NEVER modified.
    """
    if not np.isfinite(jmr_kwh):
        fail("JMR must be a finite numeric value.")

    if jmr_kwh <= 0:
        fail("JMR must be greater than zero.")

    work = work.copy()

    observed_mask = work["_lp_value"].notna()
    predicted_mask = work["_raw_prediction"].notna()

    observed_total = float(work.loc[observed_mask, "_lp_value"].sum())
    raw_predicted_total = float(
        work.loc[predicted_mask, "_raw_prediction"].sum()
    )

    if raw_predicted_total <= 0:
        fail(
            "The total raw model prediction for missing LPs is zero. "
            "Proportional JMR reconciliation cannot be performed."
        )

    remaining_for_missing = float(jmr_kwh - observed_total)

    if remaining_for_missing < 0:
        fail(
            "JMR reconciliation is impossible because the observed LP total "
            f"({observed_total:,.0f} kWh) already exceeds the supplied JMR "
            f"({jmr_kwh:,.0f} kWh)."
        )

    # Whole-kWh final output requires the energy assigned to missing LPs to
    # itself be a whole number. This is normally true for integer meter data.
    if not np.isclose(remaining_for_missing, round(remaining_for_missing), atol=1e-6):
        fail(
            "Whole-kWh output cannot exactly reconcile to the supplied JMR "
            "because the required missing energy is fractional.\n"
            f"Required missing energy: {remaining_for_missing:.10f} kWh"
        )

    scale_factor = remaining_for_missing / raw_predicted_total

    if not np.isfinite(scale_factor) or scale_factor < 0:
        fail(f"Invalid JMR reconciliation scale factor: {scale_factor}")

    work["_reconciled_prediction"] = np.nan

    scaled_predictions = (
        work.loc[predicted_mask, "_raw_prediction"].to_numpy(dtype=float)
        * scale_factor
    )

    # Round all missing-LP values to whole kWh for the final deliverable.
    rounded_predictions = np.rint(scaled_predictions).astype(np.int64)

    required_missing_total = int(round(remaining_for_missing))
    rounded_missing_total = int(rounded_predictions.sum())
    residual = required_missing_total - rounded_missing_total

    # Distribute the integer rounding residual. Each correction is exactly
    # +/-1 kWh, so the model shape is preserved while the monthly JMR is exact.
    if residual != 0:
        step = 1 if residual > 0 else -1
        corrections = abs(residual)

        if corrections > len(rounded_predictions):
            fail(
                "Rounding correction is larger than the number of missing LPs. "
                "Cannot distribute integer reconciliation residual safely."
            )

        # Prefer values with the largest fractional parts when adding kWh and
        # the smallest fractional parts when removing kWh. This minimizes the
        # total rounding distortion.
        fractions = scaled_predictions - np.floor(scaled_predictions)

        if step > 0:
            order = np.argsort(-fractions)
        else:
            order = np.argsort(fractions)

        for idx in order[:corrections]:
            rounded_predictions[idx] += step

    # Store whole-kWh reconciled predictions.
    work.loc[predicted_mask, "_reconciled_prediction"] = rounded_predictions

    # Observed LPs remain exactly as supplied; only missing cells are filled.
    work["_final_value"] = work["_lp_value"]
    work.loc[predicted_mask, "_final_value"] = rounded_predictions

    final_total = float(work["_final_value"].sum())
    reconciliation_error = final_total - float(jmr_kwh)

    if abs(reconciliation_error) > 1e-6:
        fail(
            "Final JMR reconciliation failed after integer rounding.\n"
            f"JMR: {jmr_kwh:,.0f} kWh\n"
            f"Final total: {final_total:,.0f} kWh\n"
            f"Error: {reconciliation_error:,.6f} kWh"
        )

    gap_records: list[dict] = []

    for gap in gaps:
        indices = list(gap.target_indices)
        raw = work.loc[indices, "_raw_prediction"].to_numpy(dtype=float)
        rec = work.loc[indices, "_reconciled_prediction"].to_numpy(dtype=float)

        gap_records.append(
            {
                "gap_id": gap.gap_id,
                "gap_length_lp": gap.length,
                "raw_prediction_sum_kwh": int(round(float(np.sum(raw)))),
                "reconciled_prediction_sum_kwh": int(round(float(np.sum(rec)))),
                "scale_factor": float(scale_factor),
            }
        )

    summary = {
        "jmr_kwh": int(round(jmr_kwh)),
        "observed_total_kwh": int(round(observed_total)),
        "raw_predicted_total_kwh": int(round(raw_predicted_total)),
        "remaining_for_missing_kwh": int(round(remaining_for_missing)),
        "scale_factor": float(scale_factor),
        "rounding_residual_kwh": int(residual),
        "final_reconciled_total_kwh": int(round(final_total)),
        "reconciliation_error_kwh": 0,
        "missing_lp_count": int(predicted_mask.sum()),
        "observed_lp_count": int(observed_mask.sum()),
    }

    return work, summary, gap_records


# =============================================================================
# OUTPUT TABLE
# =============================================================================

def create_output_dataframe(
    original_df: pd.DataFrame,
    work: pd.DataFrame,
    time_column: str,
    lp_column: str,
) -> pd.DataFrame:
    """
    Create the final audit-friendly output dataframe.

    Energy values in the final CSV/XLSX are exported as whole kWh.  The
    original observed values are preserved exactly; missing values contain the
    integer JMR-reconciled predictions.
    """
    out = original_df.copy()

    chrono = work.sort_values("_original_row_order")
    chrono = chrono.set_index("_original_row_order").sort_index()

    original_values = chrono["_lp_value"]
    raw_prediction = chrono["_raw_prediction"]
    reconciled_prediction = chrono["_reconciled_prediction"]
    final_value = chrono["_final_value"]
    gap_length = chrono["_model_gap_length"]

    # Original numeric values. Integer input remains integer-valued in Excel;
    # missing values remain blank/NA in this audit column.
    out[f"{lp_column} - Original"] = (
        original_values.round().astype("Int64").to_numpy()
    )

    # Raw model output is diagnostic only and is rounded for presentation.
    out[f"{lp_column} - Raw Prediction"] = (
        raw_prediction.round().astype("Int64").to_numpy()
    )

    # Final reconciled predictions are whole kWh.
    out[f"{lp_column} - Reconciled"] = (
        reconciled_prediction.round().astype("Int64").to_numpy()
    )

    # Difference is meaningful only for originally missing rows. Observed rows
    # therefore remain blank in this column.
    difference = reconciled_prediction - original_values
    out[f"{lp_column} - Difference"] = (
        difference.round().astype("Int64").to_numpy()
    )

    out["Imputation Status"] = np.where(
        original_values.notna().to_numpy(),
        "Observed",
        "Imputed + JMR Reconciled",
    )

    out["Gap Length (LP)"] = (
        gap_length.round().astype("Int64").to_numpy()
    )

    # Replace the meter LP column with final values. This is the actual
    # reconciled deliverable column.
    out[lp_column] = (
        final_value.round().astype("Int64").to_numpy()
    )

    return out


# =============================================================================
# PLOT
# =============================================================================

def create_plot(
    work: pd.DataFrame,
    time_column: str,
    lp_column: str,
    jmr_kwh: float,
    output_path: Path,
) -> None:
    """Create one image showing original, imputed/reconciled and difference."""
    times = work["_parsed_time"]

    original = work["_lp_value"].to_numpy(dtype=float)
    final = work["_final_value"].to_numpy(dtype=float)

    # Difference is defined only for originally missing points.
    difference = np.full(len(work), np.nan, dtype=float)
    missing = np.isnan(original)
    difference[missing] = final[missing]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(
        times,
        original,
        label="Original observed LP",
        linewidth=1.2,
    )
    axes[0].plot(
        times,
        final,
        label="Imputed + JMR reconciled LP",
        linewidth=1.2,
    )

    axes[0].set_title(
        f"AMI Load Profile Imputation — {lp_column}\n"
        f"JMR = {jmr_kwh:,.0f} kWh"
    )
    axes[0].set_ylabel("kWh / 30 min")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        times,
        difference,
        label="Imputed value at missing LP",
        linewidth=1.2,
    )
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_ylabel("Imputed kWh")
    axes[1].set_xlabel(time_column)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_safe)


# =============================================================================
# OPTIONAL MLFLOW INFERENCE LOGGING
# =============================================================================

def log_inference_to_mlflow(
    input_path: Path,
    jmr_kwh: float,
    summary: dict,
    output_dir: Path,
) -> str | None:
    """
    Log an inference audit run to the existing local MLflow database.

    Failure to log to MLflow does NOT invalidate the actual inference result.
    """
    try:
        import mlflow
    except ImportError:
        return None

    try:
        mlflow_db = PROJECT_ROOT / "mlflow.db"
        if mlflow_db.exists():
            mlflow.set_tracking_uri(
                f"sqlite:///{mlflow_db.resolve().as_posix()}"
            )

        experiment_name = "AMI-Smart-Meters-V1-Tuned-Inference"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            mlflow.log_param("pipeline_version", "V1_TUNED")
            mlflow.log_param("input_file", str(input_path.name))
            mlflow.log_param("jmr_kwh", float(jmr_kwh))
            mlflow.log_param(
                "supported_gap_lengths",
                ",".join(map(str, GAP_LENGTHS)),
            )
            mlflow.log_metric(
                "observed_total_kwh",
                float(summary["observed_total_kwh"]),
            )
            mlflow.log_metric(
                "raw_predicted_total_kwh",
                float(summary["raw_predicted_total_kwh"]),
            )
            mlflow.log_metric(
                "reconciled_total_kwh",
                float(summary["final_reconciled_total_kwh"]),
            )
            mlflow.log_metric(
                "reconciliation_error_kwh",
                float(summary["reconciliation_error_kwh"]),
            )
            mlflow.log_metric(
                "jmr_scale_factor",
                float(summary["scale_factor"]),
            )

            metadata_path = output_dir / "inference_summary.json"
            if metadata_path.exists():
                mlflow.log_artifact(str(metadata_path))

            return run.info.run_id

    except Exception as exc:
        warnings.warn(
            f"MLflow inference logging skipped: {exc}",
            RuntimeWarning,
        )
        return None


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_inference(
    input_path: str | Path,
    jmr_kwh: float,
    output_dir: str | Path | None = None,
) -> dict:
    input_path = Path(input_path)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_ROOT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("AMI SMART METER — V1 TUNED INFERENCE + JMR RECONCILIATION")
    print("=" * 80)
    print()
    print("Training models          : V1 TUNED")
    print("Gap models               : 1, 6, 24, 48 LP")
    print("Prediction granularity   : 1 prediction / missing LP")
    print("JMR during training      : NOT USED")
    print("JMR during inference     : ENABLED")
    print("Reconciliation           : PREDICTED VALUES ONLY")
    print("Observed values          : UNCHANGED")
    print("MLflow inference logging : OPTIONAL")
    print()

    df = read_input_file(input_path)

    time_column = identify_time_column(df)
    lp_column = identify_lp_column(df, time_column)

    print()
    print(f"Timestamp column : {time_column}")
    print(f"LP column        : {lp_column}")

    work = prepare_input(df, time_column, lp_column)

    gaps = detect_gaps(work["_lp_value"])
    validate_gaps(gaps, len(work))

    print()
    print(line("-", 80))
    print("MISSING GAP DETECTION")
    print(line("-", 80))
    print(f"Total rows       : {len(work):,}")
    print(f"Missing LPs      : {sum(g.length for g in gaps):,}")
    print(f"Number of gaps   : {len(gaps)}")

    for gap in gaps:
        print(
            f"Gap {gap.gap_id:02d} | "
            f"{gap.length:>2d} LP | "
            f"{work.loc[gap.start, '_parsed_time']} -> "
            f"{work.loc[gap.end, '_parsed_time']}"
        )

    work, prediction_gap_records = predict_gaps(work, gaps)

    print()
    print(line("-", 80))
    print("JMR RECONCILIATION")
    print(line("-", 80))
    print(f"Supplied JMR      : {jmr_kwh:,.0f} kWh")

    work, jmr_summary, reconciliation_gap_records = reconcile_to_jmr(
        work,
        gaps,
        float(jmr_kwh),
    )

    print(f"Observed total    : {jmr_summary['observed_total_kwh']:,.0f} kWh")
    print(
        f"Raw predictions   : "
        f"{jmr_summary['raw_predicted_total_kwh']:,.0f} kWh"
    )
    print(
        f"Remaining for gaps: "
        f"{jmr_summary['remaining_for_missing_kwh']:,.0f} kWh"
    )
    print(f"Scale factor      : {jmr_summary['scale_factor']:.10f}")
    print(
        f"Final total       : "
        f"{jmr_summary['final_reconciled_total_kwh']:,.0f} kWh"
    )
    print(
        f"Reconciliation err: "
        f"{jmr_summary['reconciliation_error_kwh']:,.0f} kWh"
    )

    # Merge gap-level audit records.
    gap_map = {}
    for rec in prediction_gap_records:
        gap_map.setdefault(rec["gap_id"], {}).update(rec)
    for rec in reconciliation_gap_records:
        gap_map.setdefault(rec["gap_id"], {}).update(rec)

    gap_records = [
        gap_map[gap.gap_id]
        for gap in gaps
    ]

    output_df = create_output_dataframe(
        df,
        work,
        time_column,
        lp_column,
    )

    # Output names.
    stem = input_path.stem
    csv_path = output_dir / f"{stem}_jmr_reconciled.csv"
    xlsx_path = output_dir / f"{stem}_jmr_reconciled.xlsx"
    plot_path = output_dir / f"{stem}_imputation_plot.png"
    summary_path = output_dir / "inference_summary.json"

    # CSV.
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # XLSX.
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            output_df.to_excel(
                writer,
                sheet_name="Reconciled LP",
                index=False,
            )

            # Compact audit sheet.
            audit_df = pd.DataFrame(
                [
                    {
                        "Item": "Input file",
                        "Value": str(input_path),
                    },
                    {
                        "Item": "Timestamp column",
                        "Value": time_column,
                    },
                    {
                        "Item": "Load-profile column",
                        "Value": lp_column,
                    },
                    {
                        "Item": "JMR kWh",
                        "Value": int(round(jmr_kwh)),
                    },
                    {
                        "Item": "Observed total kWh",
                        "Value": jmr_summary["observed_total_kwh"],
                    },
                    {
                        "Item": "Raw predicted total kWh",
                        "Value": jmr_summary["raw_predicted_total_kwh"],
                    },
                    {
                        "Item": "JMR scale factor",
                        "Value": jmr_summary["scale_factor"],
                    },
                    {
                        "Item": "Final reconciled total kWh",
                        "Value": jmr_summary[
                            "final_reconciled_total_kwh"
                        ],
                    },
                    {
                        "Item": "Reconciliation error kWh",
                        "Value": jmr_summary[
                            "reconciliation_error_kwh"
                        ],
                    },
                ]
            )
            audit_df.to_excel(
                writer,
                sheet_name="Audit",
                index=False,
            )

            pd.DataFrame(gap_records).to_excel(
                writer,
                sheet_name="Gap Audit",
                index=False,
            )

    except ImportError:
        fail(
            "Excel output requires openpyxl. Install it with:\n"
            "python -m pip install openpyxl"
        )

    create_plot(
        work,
        time_column,
        lp_column,
        float(jmr_kwh),
        plot_path,
    )

    summary = {
        "pipeline": "AMI Smart Meter V1 Tuned Inference",
        "pipeline_version": "V1_TUNED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "timestamp_column": time_column,
        "load_profile_column": lp_column,
        "model_directory": str(MODEL_ROOT),
        "gap_lengths_supported": list(GAP_LENGTHS),
        "feature_count": len(V1_FEATURES),
        "features": V1_FEATURES,
        "jmr_reconciliation_method": "proportional_scaling_of_predicted_values_only",
        "jmr": jmr_summary,
        "gaps": gap_records,
        "outputs": {
            "csv": str(csv_path),
            "xlsx": str(xlsx_path),
            "plot": str(plot_path),
            "summary": str(summary_path),
        },
    }

    write_json(summary_path, summary)

    # Log after summary exists.
    mlflow_run_id = log_inference_to_mlflow(
        input_path=input_path,
        jmr_kwh=float(jmr_kwh),
        summary=jmr_summary,
        output_dir=output_dir,
    )

    summary["mlflow_run_id"] = mlflow_run_id
    write_json(summary_path, summary)

    print()
    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print()
    print(f"CSV output     : {csv_path}")
    print(f"Excel output   : {xlsx_path}")
    print(f"Plot           : {plot_path}")
    print(f"Summary        : {summary_path}")
    print(
        f"Final JMR total: "
        f"{jmr_summary['final_reconciled_total_kwh']:,.0f} kWh"
    )

    if mlflow_run_id:
        print(f"MLflow run     : {mlflow_run_id}")
    else:
        print("MLflow run     : NOT LOGGED")

    print()

    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run V1 tuned Random Forest inference and proportional "
            "JMR reconciliation."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV/XLSX/XLSM file.",
    )

    parser.add_argument(
        "--jmr",
        required=True,
        type=float,
        help="Full-month JMR total in kWh.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_inference(
            input_path=args.input,
            jmr_kwh=args.jmr,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print()
        print("=" * 80)
        print("INFERENCE FAILED")
        print("=" * 80)
        print()
        print(f"{type(exc).__name__}: {exc}")
        print()

        # Keep the CLI failure visible to scripts/CI.
        raise SystemExit(1)


if __name__ == "__main__":
    main()