"""
RANDOM FOREST CDP IMPUTATION
DIAGNOSTICS STAGE

Purpose
-------
Analyze already-generated Random Forest predictions.

This module DOES NOT:
    - train models
    - modify source CSV
    - modify feature data
    - generate gaps
    - create supervised datasets
    - modify predictions

It reads:
    outputs/model/predictions/predictions_gap_*.parquet

and produces diagnostic reports under:
    outputs/diagnostics/

Gap lengths:
    1, 6, 24, 48 LP

96 LP:
    REMOVED

Formulation:
    fixed event windows

Prediction:
    one RF prediction per missing LP
"""

from pathlib import Path
from typing import List

import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import CONFIG


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    CONFIG.get(
        "project_root",
        Path(__file__).resolve().parents[1],
    )
)

GAP_LENGTHS = CONFIG.get(
    "gaps",
    {}
).get(
    "lengths",
    [1, 6, 24, 48],
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / CONFIG["outputs"].get(
        "model",
        {}
    ).get(
        "predictions_dir",
        "outputs/model/predictions",
    )
)

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / CONFIG["outputs"].get(
        "diagnostics_dir",
        "outputs/diagnostics",
    )
)

SUMMARY_DIR = DIAGNOSTICS_DIR / "summary"
EVENT_DIR = DIAGNOSTICS_DIR / "event"
POSITION_DIR = DIAGNOSTICS_DIR / "position"
TIME_DIR = DIAGNOSTICS_DIR / "time"
RESIDUAL_DIR = DIAGNOSTICS_DIR / "residual"


# =============================================================================
# LOGGING
# =============================================================================

def info(message: str) -> None:
    print(message)


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def create_output_directories() -> None:

    for directory in [
        DIAGNOSTICS_DIR,
        SUMMARY_DIR,
        EVENT_DIR,
        POSITION_DIR,
        TIME_DIR,
        RESIDUAL_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# FILE PATH
# =============================================================================

def prediction_file(
    gap_length: int,
) -> Path:

    return (
        PREDICTIONS_DIR
        / f"predictions_gap_{gap_length}.parquet"
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

REQUIRED_COLUMNS = [
    "event_id",
    "gap_length",
    "split",
    "target_index",
    "gap_position",
    "Time",
    "ground_truth",
    "prediction",
]


def validate_prediction_dataframe(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:

    subsection(
        f"GAP {gap_length} INPUT VALIDATION"
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        fail(
            f"Gap {gap_length}: missing required columns:\n"
            f"{missing}"
        )

    info(
        "Required columns       : PASSED"
    )

    if dataframe.empty:
        fail(
            f"Gap {gap_length}: prediction dataframe is empty."
        )

    if dataframe["prediction"].isna().any():
        fail(
            f"Gap {gap_length}: prediction contains NaN."
        )

    if dataframe["ground_truth"].isna().any():
        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    if not np.isfinite(
        dataframe["prediction"].to_numpy(
            dtype=float
        )
    ).all():
        fail(
            f"Gap {gap_length}: prediction contains Inf."
        )

    if not np.isfinite(
        dataframe["ground_truth"].to_numpy(
            dtype=float
        )
    ).all():
        fail(
            f"Gap {gap_length}: ground_truth contains Inf."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):
        dataframe["Time"] = pd.to_datetime(
            dataframe["Time"]
        )

    unique_gap_lengths = (
        dataframe["gap_length"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(unique_gap_lengths) != 1:
        fail(
            f"Gap {gap_length}: multiple gap lengths found:\n"
            f"{unique_gap_lengths}"
        )

    if int(unique_gap_lengths[0]) != gap_length:
        fail(
            f"Gap {gap_length}: dataframe contains "
            f"gap length {unique_gap_lengths[0]}."
        )

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        dataframe["split"]
        .astype(str)
        .unique()
    )

    if not actual_splits.issubset(
        allowed_splits
    ):
        fail(
            f"Gap {gap_length}: unexpected split labels:\n"
            f"{actual_splits}"
        )

    if dataframe["target_index"].duplicated().any():
        fail(
            f"Gap {gap_length}: duplicate target_index values."
        )

    info(
        "Prediction column     : PASSED"
    )

    info(
        "Ground truth          : PASSED"
    )

    info(
        "Prediction values     : PASSED"
    )

    info(
        "Split labels          : PASSED"
    )

    info(
        "Timestamp              : PASSED"
    )

    info(
        "Target indices        : PASSED"
    )

    info(
        f"Rows                   : {len(dataframe):,}"
    )


# =============================================================================
# ADD ERROR COLUMNS
# =============================================================================

def add_error_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["error"] = (
        dataframe["prediction"]
        - dataframe["ground_truth"]
    )

    dataframe["absolute_error"] = (
        dataframe["error"].abs()
    )

    dataframe["squared_error"] = (
        dataframe["error"] ** 2
    )

    dataframe["percentage_error"] = np.where(
        dataframe["ground_truth"] != 0,
        (
            dataframe["absolute_error"]
            /
            dataframe["ground_truth"].abs()
            *
            100.0
        ),
        np.nan,
    )

    dataframe["hour"] = (
        pd.to_datetime(
            dataframe["Time"]
        ).dt.hour
    )

    dataframe["day_of_week"] = (
        pd.to_datetime(
            dataframe["Time"]
        ).dt.dayofweek
    )

    dataframe["month"] = (
        pd.to_datetime(
            dataframe["Time"]
        ).dt.month
    )

    return dataframe


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> dict:

    y_true = dataframe[
        "ground_truth"
    ].to_numpy(
        dtype=float
    )

    y_pred = dataframe[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    if len(
        np.unique(y_true)
    ) > 1:

        r2 = r2_score(
            y_true,
            y_pred,
        )

    else:
        r2 = np.nan

    nonzero = (
        y_true != 0
    )

    if nonzero.any():

        mape = np.mean(
            np.abs(
                (
                    y_true[nonzero]
                    -
                    y_pred[nonzero]
                )
                /
                y_true[nonzero]
            )
        ) * 100.0

    else:

        mape = np.nan

    mean_true = np.mean(
        np.abs(y_true)
    )

    if mean_true != 0:

        mae_percent = (
            mae
            /
            mean_true
            *
            100.0
        )

    else:

        mae_percent = np.nan

    return {
        "samples": len(dataframe),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "MAE_percent_of_mean": mae_percent,
        "mean_error_bias": np.mean(
            y_pred - y_true
        ),
        "mean_absolute_error": np.mean(
            np.abs(
                y_pred - y_true
            )
        ),
        "max_absolute_error": np.max(
            np.abs(
                y_pred - y_true
            )
        ),
    }


# =============================================================================
# SPLIT DIAGNOSTICS
# =============================================================================

def analyze_splits(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    records = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataframe[
            dataframe["split"].astype(str)
            == split
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(
            subset
        )

        metrics["gap_length"] = gap_length
        metrics["split"] = split

        records.append(
            metrics
        )

    result = pd.DataFrame(
        records
    )

    result = result[
        [
            "gap_length",
            "split",
            "samples",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
            "MAE_percent_of_mean",
            "mean_error_bias",
            "max_absolute_error",
        ]
    ]

    return result


# =============================================================================
# EVENT DIAGNOSTICS
# =============================================================================

def analyze_events(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    test = dataframe[
        dataframe["split"].astype(str)
        == "test"
    ]

    records = []

    for event_id, group in test.groupby(
        "event_id"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics["gap_length"] = gap_length
        metrics["event_id"] = int(
            event_id
        )

        records.append(
            metrics
        )

    result = pd.DataFrame(
        records
    )

    if result.empty:
        return result

    return result[
        [
            "gap_length",
            "event_id",
            "samples",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
            "mean_error_bias",
            "max_absolute_error",
        ]
    ].sort_values(
        "MAE",
        ascending=False,
    )


# =============================================================================
# GAP POSITION DIAGNOSTICS
# =============================================================================

def analyze_positions(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    test = dataframe[
        dataframe["split"].astype(str)
        == "test"
    ]

    records = []

    for position, group in test.groupby(
        "gap_position"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics["gap_length"] = gap_length
        metrics["gap_position"] = int(
            position
        )

        records.append(
            metrics
        )

    result = pd.DataFrame(
        records
    )

    if result.empty:
        return result

    return result[
        [
            "gap_length",
            "gap_position",
            "samples",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
            "mean_error_bias",
            "max_absolute_error",
        ]
    ].sort_values(
        "gap_position"
    )


# =============================================================================
# HOUR DIAGNOSTICS
# =============================================================================

def analyze_hours(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    test = dataframe[
        dataframe["split"].astype(str)
        == "test"
    ]

    records = []

    for hour, group in test.groupby(
        "hour"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics["gap_length"] = gap_length
        metrics["hour"] = int(
            hour
        )

        records.append(
            metrics
        )

    result = pd.DataFrame(
        records
    )

    if result.empty:
        return result

    return result[
        [
            "gap_length",
            "hour",
            "samples",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
            "mean_error_bias",
            "max_absolute_error",
        ]
    ].sort_values(
        "hour"
    )


# =============================================================================
# DAY-OF-WEEK DIAGNOSTICS
# =============================================================================

def analyze_days(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    test = dataframe[
        dataframe["split"].astype(str)
        == "test"
    ]

    records = []

    for day, group in test.groupby(
        "day_of_week"
    ):

        metrics = calculate_metrics(
            group
        )

        metrics["gap_length"] = gap_length
        metrics["day_of_week"] = int(
            day
        )

        records.append(
            metrics
        )

    result = pd.DataFrame(
        records
    )

    if result.empty:
        return result

    return result[
        [
            "gap_length",
            "day_of_week",
            "samples",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
            "mean_error_bias",
            "max_absolute_error",
        ]
    ].sort_values(
        "day_of_week"
    )


# =============================================================================
# RESIDUAL DIAGNOSTICS
# =============================================================================

def residual_summary(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    records = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataframe[
            dataframe["split"].astype(str)
            == split
        ]

        if subset.empty:
            continue

        residuals = subset[
            "error"
        ].to_numpy(
            dtype=float
        )

        records.append(
            {
                "gap_length": gap_length,
                "split": split,
                "mean_residual": np.mean(
                    residuals
                ),
                "median_residual": np.median(
                    residuals
                ),
                "std_residual": np.std(
                    residuals
                ),
                "min_residual": np.min(
                    residuals
                ),
                "max_residual": np.max(
                    residuals
                ),
                "p05_absolute_error": np.percentile(
                    subset["absolute_error"],
                    5,
                ),
                "p50_absolute_error": np.percentile(
                    subset["absolute_error"],
                    50,
                ),
                "p95_absolute_error": np.percentile(
                    subset["absolute_error"],
                    95,
                ),
                "p99_absolute_error": np.percentile(
                    subset["absolute_error"],
                    99,
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# PREDICTION DISTRIBUTION
# =============================================================================

def prediction_distribution(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:

    records = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = dataframe[
            dataframe["split"].astype(str)
            == split
        ]

        if subset.empty:
            continue

        records.append(
            {
                "gap_length": gap_length,
                "split": split,
                "true_mean": subset[
                    "ground_truth"
                ].mean(),
                "true_std": subset[
                    "ground_truth"
                ].std(),
                "true_min": subset[
                    "ground_truth"
                ].min(),
                "true_max": subset[
                    "ground_truth"
                ].max(),
                "prediction_mean": subset[
                    "prediction"
                ].mean(),
                "prediction_std": subset[
                    "prediction"
                ].std(),
                "prediction_min": subset[
                    "prediction"
                ].min(),
                "prediction_max": subset[
                    "prediction"
                ].max(),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    section(
        "RANDOM FOREST CDP IMPUTATION — DIAGNOSTICS STAGE"
    )

    info(
        "Stage: saved predictions → model diagnostics"
    )

    info(
        "Gap lengths: 1, 6, 24, 48 LP"
    )

    info(
        "96 LP gap: REMOVED"
    )

    info(
        "Sliding windows: DISABLED"
    )

    info(
        "One RF prediction per missing LP: ENABLED"
    )

    info(
        "Model retraining: DISABLED"
    )

    info(
        "Source modification: DISABLED"
    )

    create_output_directories()

    all_split_results = []
    all_event_results = []
    all_position_results = []
    all_hour_results = []
    all_day_results = []
    all_residual_results = []
    all_distribution_results = []
    all_prediction_errors = []

    for gap_length in GAP_LENGTHS:

        subsection(
            f"GAP LENGTH: {gap_length} LP"
        )

        file_path = prediction_file(
            gap_length
        )

        info(
            "Input prediction file:"
        )

        info(
            f"    {file_path}"
        )

        if not file_path.exists():

            fail(
                f"Prediction file not found:\n"
                f"{file_path}"
            )

        dataframe = pd.read_parquet(
            file_path
        )

        info(
            f"Rows                 : {len(dataframe):,}"
        )

        info(
            f"Columns              : {len(dataframe.columns)}"
        )

        validate_prediction_dataframe(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        dataframe = add_error_columns(
            dataframe
        )

        # -------------------------------------------------------------
        # Split diagnostics
        # -------------------------------------------------------------

        split_metrics = analyze_splits(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        split_metrics.to_csv(
            SUMMARY_DIR
            / f"split_diagnostics_gap_{gap_length}.csv",
            index=False,
        )

        all_split_results.append(
            split_metrics
        )

        # -------------------------------------------------------------
        # Event diagnostics
        # -------------------------------------------------------------

        event_metrics = analyze_events(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        event_metrics.to_csv(
            EVENT_DIR
            / f"event_diagnostics_gap_{gap_length}.csv",
            index=False,
        )

        all_event_results.append(
            event_metrics
        )

        # -------------------------------------------------------------
        # Position diagnostics
        # -------------------------------------------------------------

        position_metrics = analyze_positions(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        position_metrics.to_csv(
            POSITION_DIR
            / f"position_diagnostics_gap_{gap_length}.csv",
            index=False,
        )

        all_position_results.append(
            position_metrics
        )

        # -------------------------------------------------------------
        # Hour diagnostics
        # -------------------------------------------------------------

        hour_metrics = analyze_hours(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        hour_metrics.to_csv(
            TIME_DIR
            / f"hour_diagnostics_gap_{gap_length}.csv",
            index=False,
        )

        all_hour_results.append(
            hour_metrics
        )

        # -------------------------------------------------------------
        # Day diagnostics
        # -------------------------------------------------------------

        day_metrics = analyze_days(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        day_metrics.to_csv(
            TIME_DIR
            / f"day_of_week_diagnostics_gap_{gap_length}.csv",
            index=False,
        )

        all_day_results.append(
            day_metrics
        )

        # -------------------------------------------------------------
        # Residual diagnostics
        # -------------------------------------------------------------

        residuals = residual_summary(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        residuals.to_csv(
            RESIDUAL_DIR
            / f"residual_diagnostics_gap_{gap_length}.csv",
            index=False,
        )

        all_residual_results.append(
            residuals
        )

        # -------------------------------------------------------------
        # Prediction distribution
        # -------------------------------------------------------------

        distributions = prediction_distribution(
            dataframe=dataframe,
            gap_length=gap_length,
        )

        distributions.to_csv(
            SUMMARY_DIR
            / f"prediction_distribution_gap_{gap_length}.csv",
            index=False,
        )

        all_distribution_results.append(
            distributions
        )

        # -------------------------------------------------------------
        # Save detailed errors
        # -------------------------------------------------------------

        error_columns = [
            "event_id",
            "gap_length",
            "split",
            "target_index",
            "gap_position",
            "Time",
            "ground_truth",
            "prediction",
            "error",
            "absolute_error",
            "squared_error",
            "percentage_error",
            "hour",
            "day_of_week",
            "month",
        ]

        dataframe[
            error_columns
        ].to_parquet(
            RESIDUAL_DIR
            / f"prediction_errors_gap_{gap_length}.parquet",
            index=False,
        )

        all_prediction_errors.append(
            dataframe[
                error_columns
            ]
        )

        # -------------------------------------------------------------
        # Console summary
        # -------------------------------------------------------------

        test = split_metrics[
            split_metrics["split"]
            == "test"
        ]

        if not test.empty:

            row = test.iloc[0]

            subsection(
                f"TEST DIAGNOSTICS — {gap_length} LP"
            )

            info(
                f"Samples               : {int(row['samples']):,}"
            )

            info(
                f"MAE                   : {row['MAE']:.4f}"
            )

            info(
                f"RMSE                  : {row['RMSE']:.4f}"
            )

            info(
                f"R2                    : {row['R2']:.4f}"
            )

            info(
                f"MAPE                  : {row['MAPE_percent']:.4f}%"
            )

            info(
                f"Mean error bias       : "
                f"{row['mean_error_bias']:.4f}"
            )

            info(
                f"Maximum absolute error: "
                f"{row['max_absolute_error']:.4f}"
            )

    # =========================================================================
    # CONSOLIDATED RESULTS
    # =========================================================================

    section(
        "CREATING CONSOLIDATED DIAGNOSTIC RESULTS"
    )

    if all_split_results:

        consolidated_split = pd.concat(
            all_split_results,
            ignore_index=True,
        )

        consolidated_split.to_csv(
            SUMMARY_DIR
            / "all_split_diagnostics.csv",
            index=False,
        )

    if all_event_results:

        consolidated_events = pd.concat(
            all_event_results,
            ignore_index=True,
        )

        consolidated_events.to_csv(
            EVENT_DIR
            / "all_event_diagnostics.csv",
            index=False,
        )

    if all_position_results:

        consolidated_positions = pd.concat(
            all_position_results,
            ignore_index=True,
        )

        consolidated_positions.to_csv(
            POSITION_DIR
            / "all_position_diagnostics.csv",
            index=False,
        )

    if all_hour_results:

        consolidated_hours = pd.concat(
            all_hour_results,
            ignore_index=True,
        )

        consolidated_hours.to_csv(
            TIME_DIR
            / "all_hour_diagnostics.csv",
            index=False,
        )

    if all_day_results:

        consolidated_days = pd.concat(
            all_day_results,
            ignore_index=True,
        )

        consolidated_days.to_csv(
            TIME_DIR
            / "all_day_of_week_diagnostics.csv",
            index=False,
        )

    if all_residual_results:

        consolidated_residuals = pd.concat(
            all_residual_results,
            ignore_index=True,
        )

        consolidated_residuals.to_csv(
            RESIDUAL_DIR
            / "all_residual_diagnostics.csv",
            index=False,
        )

    if all_distribution_results:

        consolidated_distribution = pd.concat(
            all_distribution_results,
            ignore_index=True,
        )

        consolidated_distribution.to_csv(
            SUMMARY_DIR
            / "all_prediction_distributions.csv",
            index=False,
        )

    if all_prediction_errors:

        consolidated_errors = pd.concat(
            all_prediction_errors,
            ignore_index=True,
        )

        consolidated_errors.to_parquet(
            SUMMARY_DIR
            / "all_prediction_errors.parquet",
            index=False,
        )

    # =========================================================================
    # FIND BEST / WORST TEST RESULTS
    # =========================================================================

    subsection(
        "TEST PERFORMANCE DIAGNOSTIC SUMMARY"
    )

    if all_split_results:

        test_summary = pd.concat(
            all_split_results,
            ignore_index=True,
        )

        test_summary = test_summary[
            test_summary["split"]
            == "test"
        ].copy()

        if not test_summary.empty:

            print()

            print(
                test_summary[
                    [
                        "gap_length",
                        "samples",
                        "MAE",
                        "RMSE",
                        "R2",
                        "MAPE_percent",
                        "mean_error_bias",
                    ]
                ].to_string(
                    index=False
                )
            )

            best_r2 = test_summary.loc[
                test_summary["R2"].idxmax()
            ]

            best_mae = test_summary.loc[
                test_summary["MAE"].idxmin()
            ]

            info(
                f"\nBest R2 gap       : "
                f"{int(best_r2['gap_length'])} LP "
                f"({best_r2['R2']:.6f})"
            )

            info(
                f"Best MAE gap      : "
                f"{int(best_mae['gap_length'])} LP "
                f"({best_mae['MAE']:.4f})"
            )

    # =========================================================================
    # WORST EVENTS
    # =========================================================================

    subsection(
        "WORST TEST EVENTS"
    )

    if all_event_results:

        events = pd.concat(
            all_event_results,
            ignore_index=True,
        )

        if not events.empty:

            worst = events.sort_values(
                "MAE",
                ascending=False,
            ).head(10)

            print(
                worst[
                    [
                        "gap_length",
                        "event_id",
                        "samples",
                        "MAE",
                        "RMSE",
                        "R2",
                    ]
                ].to_string(
                    index=False
                )
            )

    # =========================================================================
    # WORST POSITIONS
    # =========================================================================

    subsection(
        "WORST TEST GAP POSITIONS"
    )

    if all_position_results:

        positions = pd.concat(
            all_position_results,
            ignore_index=True,
        )

        if not positions.empty:

            worst_positions = positions.sort_values(
                "MAE",
                ascending=False,
            ).head(10)

            print(
                worst_positions[
                    [
                        "gap_length",
                        "gap_position",
                        "samples",
                        "MAE",
                        "RMSE",
                        "R2",
                    ]
                ].to_string(
                    index=False
                )
            )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "stage": "diagnostics",
        "gap_lengths": [
            int(x)
            for x in GAP_LENGTHS
        ],
        "removed_gap_length": 96,
        "sliding_windows": False,
        "one_prediction_per_missing_lp": True,
        "model_retraining": False,
        "source_modification": False,
        "prediction_directory": str(
            PREDICTIONS_DIR
        ),
        "diagnostics_directory": str(
            DIAGNOSTICS_DIR
        ),
    }

    with open(
        DIAGNOSTICS_DIR
        / "diagnostics_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    section(
        "DIAGNOSTICS STAGE COMPLETE"
    )

    info(
        "Models retrained     : NO"
    )

    info(
        "Predictions modified : NO"
    )

    info(
        "Gap generation       : NO"
    )

    info(
        "Supervised datasets  : NO"
    )

    info(
        "Source CSV           : UNCHANGED"
    )

    info(
        "Feature dataframe    : UNCHANGED"
    )

    info(
        "Diagnostics outputs:"
    )

    info(
        f"    {DIAGNOSTICS_DIR}"
    )

    info(
        "Key outputs:"
    )

    info(
        f"    {SUMMARY_DIR / 'all_split_diagnostics.csv'}"
    )

    info(
        f"    {EVENT_DIR / 'all_event_diagnostics.csv'}"
    )

    info(
        f"    {POSITION_DIR / 'all_position_diagnostics.csv'}"
    )

    info(
        f"    {TIME_DIR / 'all_hour_diagnostics.csv'}"
    )

    info(
        f"    {RESIDUAL_DIR / 'all_residual_diagnostics.csv'}"
    )

    info(
        f"    {SUMMARY_DIR / 'all_prediction_errors.parquet'}"
    )


if __name__ == "__main__":
    main()