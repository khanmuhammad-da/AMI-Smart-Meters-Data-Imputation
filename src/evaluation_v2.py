"""
RANDOM FOREST CDP IMPUTATION — V2 EVALUATION STAGE

Purpose
-------
Compare the frozen V1 Random Forest baseline against V2 Random Forest.

V1:
    22 features
    outputs/model/predictions/

V2:
    43 features
    outputs/model_v2/predictions/

This module:
    - does NOT retrain models
    - does NOT modify V1 predictions
    - does NOT modify V2 predictions
    - does NOT modify source data
    - compares TEST performance
    - produces consolidated V1 vs V2 results

Gap lengths:
    1, 6, 24, 48 LP

Metrics:
    MAE
    RMSE
    R2
    MAPE
    Mean error bias
    Maximum absolute error
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import CONFIG


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(
    CONFIG["project_root"]
)


# =============================================================================
# GAP CONFIGURATION
# =============================================================================

GAP_LENGTHS = [
    int(x)
    for x in CONFIG["gaps"]["lengths"]
]


# =============================================================================
# V1 / V2 OUTPUT DIRECTORIES
# =============================================================================

V1_PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model"
    / "predictions"
)

V2_PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_v2"
    / "predictions"
)


# =============================================================================
# EVALUATION OUTPUT
# =============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_v2"
)

SUMMARY_DIR = (
    OUTPUT_DIR
    / "summary"
)

DETAIL_DIR = (
    OUTPUT_DIR
    / "detail"
)


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
# FILE PATHS
# =============================================================================

def v1_prediction_path(
    gap_length: int,
) -> Path:

    return (
        V1_PREDICTION_DIR
        / f"predictions_gap_{gap_length}.parquet"
    )


def v2_prediction_path(
    gap_length: int,
) -> Path:

    return (
        V2_PREDICTION_DIR
        / f"predictions_gap_{gap_length}_v2.parquet"
    )


# =============================================================================
# VALIDATION
# =============================================================================

REQUIRED_COLUMNS = {
    "event_id",
    "gap_length",
    "split",
    "target_index",
    "gap_position",
    "Time",
    "ground_truth",
    "prediction",
}


def validate_prediction_dataframe(
    dataframe: pd.DataFrame,
    gap_length: int,
    version: str,
) -> None:

    missing = sorted(
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing:

        fail(
            f"{version} gap {gap_length}: "
            f"missing required columns: {missing}"
        )

    if dataframe.empty:

        fail(
            f"{version} gap {gap_length}: "
            "prediction dataframe is empty."
        )

    # -------------------------------------------------------------------------
    # Gap length
    # -------------------------------------------------------------------------

    if not (
        dataframe["gap_length"]
        == gap_length
    ).all():

        fail(
            f"{version} gap {gap_length}: "
            "unexpected gap_length values."
        )

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    splits = set(
        dataframe["split"]
        .astype(str)
        .unique()
    )

    if not splits.issubset(
        {"train", "validation", "test"}
    ):

        fail(
            f"{version} gap {gap_length}: "
            f"unexpected split labels: {splits}"
        )

    # -------------------------------------------------------------------------
    # Numeric values
    # -------------------------------------------------------------------------

    for column in [
        "ground_truth",
        "prediction",
    ]:

        values = dataframe[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            fail(
                f"{version} gap {gap_length}: "
                f"{column} contains NaN/Inf."
            )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    timestamps = pd.to_datetime(
        dataframe["Time"],
        errors="coerce",
    )

    if timestamps.isna().any():

        fail(
            f"{version} gap {gap_length}: "
            "invalid timestamps."
        )


# =============================================================================
# ALIGN V1 AND V2
# =============================================================================

def align_predictions(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    gap_length: int,
) -> pd.DataFrame:
    """
    Align V1 and V2 predictions using the actual prediction identity:

        split
        event_id
        target_index
        gap_position
        Time
        ground_truth

    This prevents accidental comparison of differently ordered rows.
    """

    keys = [
        "split",
        "event_id",
        "target_index",
        "gap_position",
        "Time",
    ]

    v1_columns = keys + [
        "ground_truth",
        "prediction",
    ]

    v2_columns = keys + [
        "ground_truth",
        "prediction",
    ]

    left = v1[
        v1_columns
    ].copy()

    right = v2[
        v2_columns
    ].copy()

    left = left.rename(
        columns={
            "ground_truth": "ground_truth_v1",
            "prediction": "prediction_v1",
        }
    )

    right = right.rename(
        columns={
            "ground_truth": "ground_truth_v2",
            "prediction": "prediction_v2",
        }
    )

    merged = left.merge(
        right,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    # -------------------------------------------------------------------------
    # Identity validation.
    # -------------------------------------------------------------------------

    if not (
        merged["_merge"] == "both"
    ).all():

        unmatched = merged[
            merged["_merge"] != "both"
        ]

        fail(
            f"Gap {gap_length}: V1/V2 prediction "
            f"identity mismatch.\n"
            f"{unmatched[keys + ['_merge']].to_string(index=False)}"
        )

    merged = merged.drop(
        columns="_merge"
    )

    # -------------------------------------------------------------------------
    # Ground truth must match.
    # -------------------------------------------------------------------------

    if not np.allclose(
        merged["ground_truth_v1"].to_numpy(
            dtype=float
        ),
        merged["ground_truth_v2"].to_numpy(
            dtype=float
        ),
    ):

        fail(
            f"Gap {gap_length}: "
            "V1 and V2 ground-truth values differ."
        )

    merged[
        "ground_truth"
    ] = merged[
        "ground_truth_v1"
    ]

    merged = merged.drop(
        columns=[
            "ground_truth_v1",
            "ground_truth_v2",
        ]
    )

    return merged


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

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

    r2 = r2_score(
        y_true,
        y_pred,
    )

    nonzero = (
        np.abs(y_true) > 0
    )

    if nonzero.any():

        mape = float(
            np.mean(
                np.abs(
                    (
                        y_pred[nonzero]
                        - y_true[nonzero]
                    )
                    /
                    y_true[nonzero]
                )
            )
            * 100.0
        )

    else:

        mape = float("nan")

    bias = float(
        np.mean(
            y_pred - y_true
        )
    )

    max_abs_error = float(
        np.max(
            np.abs(
                y_pred - y_true
            )
        )
    )

    return {
        "samples": int(
            len(y_true)
        ),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": mape,
        "mean_error_bias": bias,
        "max_absolute_error": max_abs_error,
    }


# =============================================================================
# COMPARE ONE GAP
# =============================================================================

def evaluate_gap(
    gap_length: int,
) -> tuple[Dict, pd.DataFrame]:

    section(
        f"V1 vs V2 EVALUATION — GAP {gap_length} LP"
    )

    v1_path = v1_prediction_path(
        gap_length
    )

    v2_path = v2_prediction_path(
        gap_length
    )

    # -------------------------------------------------------------------------
    # Check files.
    # -------------------------------------------------------------------------

    if not v1_path.exists():

        fail(
            f"V1 prediction file not found:\n"
            f"{v1_path}"
        )

    if not v2_path.exists():

        fail(
            f"V2 prediction file not found:\n"
            f"{v2_path}"
        )

    info(
        f"V1 predictions:\n    {v1_path}"
    )

    info(
        f"V2 predictions:\n    {v2_path}"
    )

    # -------------------------------------------------------------------------
    # Read.
    # -------------------------------------------------------------------------

    v1 = pd.read_parquet(
        v1_path
    )

    v2 = pd.read_parquet(
        v2_path
    )

    info(
        f"V1 rows              : {len(v1):,}"
    )

    info(
        f"V2 rows              : {len(v2):,}"
    )

    # -------------------------------------------------------------------------
    # Validate.
    # -------------------------------------------------------------------------

    validate_prediction_dataframe(
        dataframe=v1,
        gap_length=gap_length,
        version="V1",
    )

    validate_prediction_dataframe(
        dataframe=v2,
        gap_length=gap_length,
        version="V2",
    )

    info(
        "Prediction validation: PASSED"
    )

    # -------------------------------------------------------------------------
    # Align.
    # -------------------------------------------------------------------------

    comparison = align_predictions(
        v1=v1,
        v2=v2,
        gap_length=gap_length,
    )

    info(
        "V1/V2 alignment      : PASSED"
    )

    # -------------------------------------------------------------------------
    # Test set only.
    # -------------------------------------------------------------------------

    test = comparison[
        comparison["split"] == "test"
    ].copy()

    if test.empty:

        fail(
            f"Gap {gap_length}: "
            "no test samples found."
        )

    y_true = test[
        "ground_truth"
    ].to_numpy(
        dtype=float
    )

    y_v1 = test[
        "prediction_v1"
    ].to_numpy(
        dtype=float
    )

    y_v2 = test[
        "prediction_v2"
    ].to_numpy(
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Metrics.
    # -------------------------------------------------------------------------

    v1_metrics = calculate_metrics(
        y_true,
        y_v1,
    )

    v2_metrics = calculate_metrics(
        y_true,
        y_v2,
    )

    # -------------------------------------------------------------------------
    # Percentage change.
    #
    # For error metrics:
    #     negative = V2 improved
    #     positive = V2 worsened
    #
    # For R2:
    #     positive = V2 improved
    #     negative = V2 worsened
    # -------------------------------------------------------------------------

    def error_change(
        v1_value: float,
        v2_value: float,
    ) -> float:

        if v1_value == 0:
            return float("nan")

        return (
            (v2_value - v1_value)
            / abs(v1_value)
        ) * 100.0

    def r2_change(
        v1_value: float,
        v2_value: float,
    ) -> float:

        return (
            v2_value - v1_value
        )

    result = {

        "gap_length": gap_length,

        "samples": v1_metrics["samples"],

        # ---------------------------------------------------------------------
        # MAE
        # ---------------------------------------------------------------------

        "V1_MAE": v1_metrics["MAE"],
        "V2_MAE": v2_metrics["MAE"],

        "MAE_change_percent": error_change(
            v1_metrics["MAE"],
            v2_metrics["MAE"],
        ),

        # ---------------------------------------------------------------------
        # RMSE
        # ---------------------------------------------------------------------

        "V1_RMSE": v1_metrics["RMSE"],
        "V2_RMSE": v2_metrics["RMSE"],

        "RMSE_change_percent": error_change(
            v1_metrics["RMSE"],
            v2_metrics["RMSE"],
        ),

        # ---------------------------------------------------------------------
        # R2
        # ---------------------------------------------------------------------

        "V1_R2": v1_metrics["R2"],
        "V2_R2": v2_metrics["R2"],

        "R2_change": r2_change(
            v1_metrics["R2"],
            v2_metrics["R2"],
        ),

        # ---------------------------------------------------------------------
        # MAPE
        # ---------------------------------------------------------------------

        "V1_MAPE_percent": (
            v1_metrics["MAPE_percent"]
        ),

        "V2_MAPE_percent": (
            v2_metrics["MAPE_percent"]
        ),

        "MAPE_change_percent": error_change(
            v1_metrics["MAPE_percent"],
            v2_metrics["MAPE_percent"],
        ),

        # ---------------------------------------------------------------------
        # Bias
        # ---------------------------------------------------------------------

        "V1_mean_error_bias": (
            v1_metrics["mean_error_bias"]
        ),

        "V2_mean_error_bias": (
            v2_metrics["mean_error_bias"]
        ),

        # ---------------------------------------------------------------------
        # Maximum error
        # ---------------------------------------------------------------------

        "V1_max_absolute_error": (
            v1_metrics["max_absolute_error"]
        ),

        "V2_max_absolute_error": (
            v2_metrics["max_absolute_error"]
        ),

        "max_error_change_percent": error_change(
            v1_metrics["max_absolute_error"],
            v2_metrics["max_absolute_error"],
        ),
    }

    # -------------------------------------------------------------------------
    # Determine winner.
    # -------------------------------------------------------------------------

    if (
        result["V2_MAE"]
        < result["V1_MAE"]
    ):

        result["MAE_winner"] = "V2"

    elif (
        result["V2_MAE"]
        > result["V1_MAE"]
    ):

        result["MAE_winner"] = "V1"

    else:

        result["MAE_winner"] = "Tie"

    if (
        result["V2_RMSE"]
        < result["V1_RMSE"]
    ):

        result["RMSE_winner"] = "V2"

    elif (
        result["V2_RMSE"]
        > result["V1_RMSE"]
    ):

        result["RMSE_winner"] = "V1"

    else:

        result["RMSE_winner"] = "Tie"

    if (
        result["V2_R2"]
        > result["V1_R2"]
    ):

        result["R2_winner"] = "V2"

    elif (
        result["V2_R2"]
        < result["V1_R2"]
    ):

        result["R2_winner"] = "V1"

    else:

        result["R2_winner"] = "Tie"

    if (
        result["V2_MAPE_percent"]
        < result["V1_MAPE_percent"]
    ):

        result["MAPE_winner"] = "V2"

    elif (
        result["V2_MAPE_percent"]
        > result["V1_MAPE_percent"]
    ):

        result["MAPE_winner"] = "V1"

    else:

        result["MAPE_winner"] = "Tie"

    # -------------------------------------------------------------------------
    # Detailed row-level comparison.
    # -------------------------------------------------------------------------

    test[
        "error_v1"
    ] = (
        test["prediction_v1"]
        - test["ground_truth"]
    )

    test[
        "absolute_error_v1"
    ] = test[
        "error_v1"
    ].abs()

    test[
        "error_v2"
    ] = (
        test["prediction_v2"]
        - test["ground_truth"]
    )

    test[
        "absolute_error_v2"
    ] = test[
        "error_v2"
    ].abs()

    test[
        "V2_minus_V1_prediction"
    ] = (
        test["prediction_v2"]
        - test["prediction_v1"]
    )

    test[
        "V2_minus_V1_absolute_error"
    ] = (
        test["absolute_error_v2"]
        - test["absolute_error_v1"]
    )

    test[
        "better_model"
    ] = np.where(
        test["absolute_error_v2"]
        < test["absolute_error_v1"],
        "V2",
        np.where(
            test["absolute_error_v2"]
            > test["absolute_error_v1"],
            "V1",
            "Tie",
        ),
    )

    test[
        "gap_length"
    ] = gap_length

    info()
    info(
        f"TEST SAMPLES         : "
        f"{result['samples']}"
    )

    info()
    info(
        "                 V1              V2"
    )

    info(
        f"MAE       {result['V1_MAE']:14.4f}"
        f"    {result['V2_MAE']:14.4f}"
    )

    info(
        f"RMSE      {result['V1_RMSE']:14.4f}"
        f"    {result['V2_RMSE']:14.4f}"
    )

    info(
        f"R2        {result['V1_R2']:14.4f}"
        f"    {result['V2_R2']:14.4f}"
    )

    info(
        f"MAPE      {result['V1_MAPE_percent']:13.4f}%"
        f"    {result['V2_MAPE_percent']:13.4f}%"
    )

    info(
        f"Bias      {result['V1_mean_error_bias']:14.4f}"
        f"    {result['V2_mean_error_bias']:14.4f}"
    )

    info()
    info(
        f"MAE winner            : "
        f"{result['MAE_winner']}"
    )

    info(
        f"RMSE winner           : "
        f"{result['RMSE_winner']}"
    )

    info(
        f"R2 winner             : "
        f"{result['R2_winner']}"
    )

    info(
        f"MAPE winner           : "
        f"{result['MAPE_winner']}"
    )

    return result, test


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION — V2 EVALUATION"
    )
    print("=" * 80)

    info()
    info(
        "V1 model                 : FROZEN"
    )
    info(
        "V2 model                 : FROZEN"
    )
    info(
        "V1 features              : 22"
    )
    info(
        "V2 features              : 43"
    )
    info(
        "Gap lengths              : 1, 6, 24, 48 LP"
    )
    info(
        "Evaluation split         : TEST ONLY"
    )
    info(
        "Retraining               : DISABLED"
    )
    info(
        "Prediction modification  : DISABLED"
    )
    info(
        "Source modification      : DISABLED"
    )
    info(
        "MLflow                   : NOT YET"
    )

    # -------------------------------------------------------------------------
    # Create directories.
    # -------------------------------------------------------------------------

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DETAIL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Evaluate each gap.
    # -------------------------------------------------------------------------

    all_results: List[Dict] = []

    all_details: List[pd.DataFrame] = []

    for gap_length in GAP_LENGTHS:

        result, details = evaluate_gap(
            gap_length
        )

        all_results.append(
            result
        )

        all_details.append(
            details
        )

    # -------------------------------------------------------------------------
    # Consolidated summary.
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        all_results
    )

    summary_path = (
        SUMMARY_DIR
        / "v1_vs_v2_test_comparison.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Detailed comparison.
    # -------------------------------------------------------------------------

    details = pd.concat(
        all_details,
        ignore_index=True,
    )

    details_path = (
        DETAIL_DIR
        / "v1_vs_v2_prediction_comparison.parquet"
    )

    details.to_parquet(
        details_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print consolidated result.
    # -------------------------------------------------------------------------

    section(
        "V1 vs V2 TEST PERFORMANCE SUMMARY"
    )

    display_columns = [
        "gap_length",
        "samples",
        "V1_MAE",
        "V2_MAE",
        "MAE_change_percent",
        "V1_RMSE",
        "V2_RMSE",
        "RMSE_change_percent",
        "V1_R2",
        "V2_R2",
        "R2_change",
        "V1_MAPE_percent",
        "V2_MAPE_percent",
        "MAPE_change_percent",
        "MAE_winner",
        "RMSE_winner",
        "R2_winner",
        "MAPE_winner",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Overall winner count.
    # -------------------------------------------------------------------------

    section(
        "METRIC WINNER COUNT"
    )

    winner_summary = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "R2",
                "MAPE",
            ],

            "V1_wins": [
                int(
                    (
                        summary["MAE_winner"]
                        == "V1"
                    ).sum()
                ),

                int(
                    (
                        summary["RMSE_winner"]
                        == "V1"
                    ).sum()
                ),

                int(
                    (
                        summary["R2_winner"]
                        == "V1"
                    ).sum()
                ),

                int(
                    (
                        summary["MAPE_winner"]
                        == "V1"
                    ).sum()
                ),
            ],

            "V2_wins": [
                int(
                    (
                        summary["MAE_winner"]
                        == "V2"
                    ).sum()
                ),

                int(
                    (
                        summary["RMSE_winner"]
                        == "V2"
                    ).sum()
                ),

                int(
                    (
                        summary["R2_winner"]
                        == "V2"
                    ).sum()
                ),

                int(
                    (
                        summary["MAPE_winner"]
                        == "V2"
                    ).sum()
                ),
            ],
        }
    )

    print(
        winner_summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Best gap for V1 and V2.
    # -------------------------------------------------------------------------

    section(
        "BEST GAP BY MAE"
    )

    best_v1_mae = summary.loc[
        summary["V1_MAE"].idxmin()
    ]

    best_v2_mae = summary.loc[
        summary["V2_MAE"].idxmin()
    ]

    info(
        f"V1 best gap : "
        f"{int(best_v1_mae['gap_length'])} LP "
        f"(MAE = {best_v1_mae['V1_MAE']:.4f})"
    )

    info(
        f"V2 best gap : "
        f"{int(best_v2_mae['gap_length'])} LP "
        f"(MAE = {best_v2_mae['V2_MAE']:.4f})"
    )

    # -------------------------------------------------------------------------
    # Overall conclusion.
    # -------------------------------------------------------------------------

    section(
        "V2 OVERALL CONCLUSION"
    )

    mae_v2_wins = int(
        (
            summary["MAE_winner"]
            == "V2"
        ).sum()
    )

    mae_v1_wins = int(
        (
            summary["MAE_winner"]
            == "V1"
        ).sum()
    )

    r2_v2_wins = int(
        (
            summary["R2_winner"]
            == "V2"
        ).sum()
    )

    r2_v1_wins = int(
        (
            summary["R2_winner"]
            == "V1"
        ).sum()
    )

    info(
        f"MAE wins — V1: {mae_v1_wins}, "
        f"V2: {mae_v2_wins}"
    )

    info(
        f"R2 wins  — V1: {r2_v1_wins}, "
        f"V2: {r2_v2_wins}"
    )

    if (
        mae_v2_wins
        > mae_v1_wins
        and
        r2_v2_wins
        >= r2_v1_wins
    ):

        conclusion = (
            "V2 shows overall improvement "
            "over V1 under the tested gaps."
        )

    elif (
        mae_v1_wins
        > mae_v2_wins
        and
        r2_v1_wins
        >= r2_v2_wins
    ):

        conclusion = (
            "V1 outperforms V2 overall under "
            "the tested gaps."
        )

    else:

        conclusion = (
            "V1 and V2 show mixed performance; "
            "further feature analysis is required."
        )

    info(
        conclusion
    )

    # -------------------------------------------------------------------------
    # Save machine-readable metadata.
    # -------------------------------------------------------------------------

    metadata = {
        "evaluation_version": "v2",
        "comparison": "V1_vs_V2",
        "evaluation_split": "test",
        "v1_feature_count": 22,
        "v2_feature_count": 43,
        "gap_lengths": GAP_LENGTHS,
        "retraining": False,
        "prediction_modification": False,
        "source_modification": False,
        "conclusion": conclusion,
        "summary_file": str(
            summary_path
        ),
        "detail_file": str(
            details_path
        ),
    }

    metadata_path = (
        SUMMARY_DIR
        / "evaluation_metadata_v2.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Final verification.
    # -------------------------------------------------------------------------

    reloaded_summary = pd.read_csv(
        summary_path
    )

    reloaded_details = pd.read_parquet(
        details_path
    )

    if len(
        reloaded_summary
    ) != len(GAP_LENGTHS):

        fail(
            "Summary reload verification failed."
        )

    if len(
        reloaded_details
    ) != sum(
        int(
            (
                pd.read_parquet(
                    v2_prediction_path(g)
                )["split"]
                == "test"
            ).sum()
        )
        for g in GAP_LENGTHS
    ):

        fail(
            "Detailed comparison reload verification failed."
        )

    info()
    info(
        "Summary reload         : PASSED"
    )

    info(
        "Detail reload          : PASSED"
    )

    # -------------------------------------------------------------------------
    # Complete.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "V1 vs V2 EVALUATION COMPLETE"
    )
    print("=" * 80)

    info()
    info(
        f"Summary:\n    {summary_path}"
    )

    info(
        f"Detailed comparison:\n    "
        f"{details_path}"
    )

    info(
        f"Metadata:\n    {metadata_path}"
    )

    info()
    info(
        "V1 predictions          : UNCHANGED"
    )

    info(
        "V2 predictions          : UNCHANGED"
    )

    info(
        "Models                  : NOT RETRAINED"
    )

    info(
        "Source data             : UNCHANGED"
    )


if __name__ == "__main__":
    main()