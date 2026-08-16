"""
RANDOM FOREST CDP IMPUTATION
V1 vs V2 vs V2.1 Evaluation

Purpose
-------
Compare the frozen V1, V2, and V2.1 predictions on the TEST split.

V1:
    22 predictor features

V2:
    43 predictor features

V2.1:
    29 predictor features
        - 22 V1 original
        - 3 historical
        - 4 recent-context

Important
---------
This script:
    - DOES NOT retrain models
    - DOES NOT modify predictions
    - DOES NOT modify source data
    - DOES NOT modify V1/V2/V2.1 datasets
    - evaluates TEST split only

Outputs
-------
outputs/
└── evaluation_v21/
    ├── summary/
    │   ├── v1_v2_v21_test_comparison.csv
    │   ├── v21_vs_v1_improvement.csv
    │   └── evaluation_metadata_v21.json
    │
    └── detail/
        └── v1_v2_v21_prediction_comparison.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model"
    / "predictions"
)

V2_PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_v2"
    / "predictions"
)

V21_PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_v21"
    / "predictions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_v21"
)

SUMMARY_DIR = OUTPUT_DIR / "summary"
DETAIL_DIR = OUTPUT_DIR / "detail"


# =============================================================================
# EXPERIMENT DEFINITION
# =============================================================================

GAP_LENGTHS = [1, 6, 24, 48]

V1_FEATURE_COUNT = 22
V2_FEATURE_COUNT = 43
V21_FEATURE_COUNT = 29

TEST_SPLIT = "test"

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


# =============================================================================
# HELPERS
# =============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure_directories() -> None:
    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DETAIL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def v1_prediction_path(gap_length: int) -> Path:
    return (
        V1_PREDICTIONS_DIR
        / f"predictions_gap_{gap_length}.parquet"
    )


def v2_prediction_path(gap_length: int) -> Path:
    return (
        V2_PREDICTIONS_DIR
        / f"predictions_gap_{gap_length}_v2.parquet"
    )


def v21_prediction_path(gap_length: int) -> Path:
    return (
        V21_PREDICTIONS_DIR
        / f"predictions_gap_{gap_length}_v21.parquet"
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_prediction_dataframe(
    dataframe: pd.DataFrame,
    model_name: str,
    gap_length: int,
) -> None:

    missing = sorted(
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        fail(
            f"{model_name} gap {gap_length}: "
            f"missing required columns: {missing}"
        )

    if dataframe.empty:
        fail(
            f"{model_name} gap {gap_length}: "
            "prediction dataframe is empty."
        )

    if dataframe["prediction"].isna().any():
        fail(
            f"{model_name} gap {gap_length}: "
            "prediction contains NaN values."
        )

    if dataframe["ground_truth"].isna().any():
        fail(
            f"{model_name} gap {gap_length}: "
            "ground_truth contains NaN values."
        )

    invalid_splits = set(
        dataframe["split"].dropna().astype(str)
    ) - {
        "train",
        "validation",
        "test",
    }

    if invalid_splits:
        fail(
            f"{model_name} gap {gap_length}: "
            f"invalid split labels: {sorted(invalid_splits)}"
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe["prediction"]
    ):
        fail(
            f"{model_name} gap {gap_length}: "
            "prediction is not numeric."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe["ground_truth"]
    ):
        fail(
            f"{model_name} gap {gap_length}: "
            "ground_truth is not numeric."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):
        try:
            pd.to_datetime(
                dataframe["Time"],
                errors="raise",
            )
        except Exception as exc:
            fail(
                f"{model_name} gap {gap_length}: "
                f"Time column cannot be parsed: {exc}"
            )


# =============================================================================
# ALIGNMENT
# =============================================================================

ALIGNMENT_COLUMNS = [
    "event_id",
    "gap_length",
    "split",
    "target_index",
    "gap_position",
    "Time",
    "ground_truth",
]


def prepare_for_alignment(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    result["Time"] = pd.to_datetime(
        result["Time"]
    )

    return result[
        ALIGNMENT_COLUMNS + ["prediction"]
    ].copy()


def validate_v1_v2_v21_alignment(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    v21: pd.DataFrame,
    gap_length: int,
) -> None:

    if len(v1) != len(v2):
        fail(
            f"Gap {gap_length}: V1/V2 row count mismatch: "
            f"{len(v1)} vs {len(v2)}"
        )

    if len(v1) != len(v21):
        fail(
            f"Gap {gap_length}: V1/V2.1 row count mismatch: "
            f"{len(v1)} vs {len(v21)}"
        )

    v1_key = v1[ALIGNMENT_COLUMNS].reset_index(drop=True)
    v2_key = v2[ALIGNMENT_COLUMNS].reset_index(drop=True)
    v21_key = v21[ALIGNMENT_COLUMNS].reset_index(drop=True)

    if not v1_key.equals(v2_key):
        fail(
            f"Gap {gap_length}: "
            "V1 and V2 sample alignment failed."
        )

    if not v1_key.equals(v21_key):
        fail(
            f"Gap {gap_length}: "
            "V1 and V2.1 sample alignment failed."
        )

    if not np.allclose(
        v1["ground_truth"].to_numpy(dtype=float),
        v2["ground_truth"].to_numpy(dtype=float),
        equal_nan=False,
    ):
        fail(
            f"Gap {gap_length}: "
            "V1 and V2 ground truth values differ."
        )

    if not np.allclose(
        v1["ground_truth"].to_numpy(dtype=float),
        v21["ground_truth"].to_numpy(dtype=float),
        equal_nan=False,
    ):
        fail(
            f"Gap {gap_length}: "
            "V1 and V2.1 ground truth values differ."
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> Dict[str, float]:

    y_true = dataframe[
        "ground_truth"
    ].to_numpy(dtype=float)

    y_pred = dataframe[
        "prediction"
    ].to_numpy(dtype=float)

    error = y_pred - y_true

    absolute_error = np.abs(error)

    squared_error = error ** 2

    mae = float(
        np.mean(absolute_error)
    )

    rmse = float(
        np.sqrt(
            np.mean(squared_error)
        )
    )

    denominator = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if denominator == 0:
        r2 = float("nan")
    else:
        r2 = float(
            1.0
            - np.sum(squared_error)
            / denominator
        )

    nonzero_mask = y_true != 0

    if np.any(nonzero_mask):
        mape = float(
            np.mean(
                np.abs(
                    error[nonzero_mask]
                    / y_true[nonzero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = float("nan")

    bias = float(
        np.mean(error)
    )

    max_absolute_error = float(
        np.max(absolute_error)
    )

    return {
        "samples": int(len(dataframe)),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "mean_error_bias": bias,
        "max_absolute_error": max_absolute_error,
    }


# =============================================================================
# IMPROVEMENT CALCULATION
# =============================================================================

def percentage_improvement(
    baseline: float,
    candidate: float,
) -> float:

    if baseline == 0:
        return float("nan")

    return float(
        (baseline - candidate)
        / abs(baseline)
        * 100.0
    )


def r2_change(
    baseline: float,
    candidate: float,
) -> float:

    return float(
        candidate - baseline
    )


# =============================================================================
# LOAD MODEL PREDICTIONS
# =============================================================================

def load_predictions(
    path: Path,
    model_name: str,
    gap_length: int,
) -> pd.DataFrame:

    if not path.exists():
        fail(
            f"{model_name} prediction file not found:\n"
            f"{path}"
        )

    dataframe = pd.read_parquet(
        path
    )

    validate_prediction_dataframe(
        dataframe=dataframe,
        model_name=model_name,
        gap_length=gap_length,
    )

    return prepare_for_alignment(
        dataframe
    )


# =============================================================================
# DETAIL DATAFRAME
# =============================================================================

def create_detail_dataframe(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    v21: pd.DataFrame,
) -> pd.DataFrame:

    detail = v1[
        ALIGNMENT_COLUMNS
    ].copy()

    detail["V1_prediction"] = (
        v1["prediction"].to_numpy()
    )

    detail["V2_prediction"] = (
        v2["prediction"].to_numpy()
    )

    detail["V21_prediction"] = (
        v21["prediction"].to_numpy()
    )

    detail["V1_error"] = (
        detail["V1_prediction"]
        - detail["ground_truth"]
    )

    detail["V2_error"] = (
        detail["V2_prediction"]
        - detail["ground_truth"]
    )

    detail["V21_error"] = (
        detail["V21_prediction"]
        - detail["ground_truth"]
    )

    detail["V1_absolute_error"] = (
        np.abs(detail["V1_error"])
    )

    detail["V2_absolute_error"] = (
        np.abs(detail["V2_error"])
    )

    detail["V21_absolute_error"] = (
        np.abs(detail["V21_error"])
    )

    detail["V21_MAE_improvement_vs_V1_percent"] = (
        np.nan
    )

    detail["V21_MAE_improvement_vs_V2_percent"] = (
        np.nan
    )

    for idx in detail.index:

        v1_error = (
            detail.loc[idx, "V1_absolute_error"]
        )

        v2_error = (
            detail.loc[idx, "V2_absolute_error"]
        )

        v21_error = (
            detail.loc[idx, "V21_absolute_error"]
        )

        if v1_error != 0:
            detail.loc[
                idx,
                "V21_MAE_improvement_vs_V1_percent",
            ] = (
                (v1_error - v21_error)
                / v1_error
                * 100.0
            )

        if v2_error != 0:
            detail.loc[
                idx,
                "V21_MAE_improvement_vs_V2_percent",
            ] = (
                (v2_error - v21_error)
                / v2_error
                * 100.0
            )

    return detail


# =============================================================================
# GAP EVALUATION
# =============================================================================

def evaluate_gap(
    gap_length: int,
) -> tuple[pd.DataFrame, Dict[str, object]]:

    print()
    print("-" * 80)
    print(
        f"V1 vs V2 vs V2.1 EVALUATION — "
        f"GAP {gap_length} LP"
    )
    print("-" * 80)

    v1_path = v1_prediction_path(
        gap_length
    )

    v2_path = v2_prediction_path(
        gap_length
    )

    v21_path = v21_prediction_path(
        gap_length
    )

    print("V1 predictions:")
    print(f"    {v1_path}")

    print("V2 predictions:")
    print(f"    {v2_path}")

    print("V2.1 predictions:")
    print(f"    {v21_path}")

    v1 = load_predictions(
        v1_path,
        "V1",
        gap_length,
    )

    v2 = load_predictions(
        v2_path,
        "V2",
        gap_length,
    )

    v21 = load_predictions(
        v21_path,
        "V2.1",
        gap_length,
    )

    print(
        f"V1 rows              : {len(v1)}"
    )

    print(
        f"V2 rows              : {len(v2)}"
    )

    print(
        f"V2.1 rows            : {len(v21)}"
    )

    validate_v1_v2_v21_alignment(
        v1=v1,
        v2=v2,
        v21=v21,
        gap_length=gap_length,
    )

    print(
        "Prediction validation: PASSED"
    )

    print(
        "V1/V2/V2.1 alignment : PASSED"
    )

    v1_test = v1[
        v1["split"] == TEST_SPLIT
    ].copy()

    v2_test = v2[
        v2["split"] == TEST_SPLIT
    ].copy()

    v21_test = v21[
        v21["split"] == TEST_SPLIT
    ].copy()

    if len(v1_test) != len(v2_test):
        fail(
            f"Gap {gap_length}: "
            "V1/V2 TEST row count mismatch."
        )

    if len(v1_test) != len(v21_test):
        fail(
            f"Gap {gap_length}: "
            "V1/V2.1 TEST row count mismatch."
        )

    v1_metrics = calculate_metrics(
        v1_test
    )

    v2_metrics = calculate_metrics(
        v2_test
    )

    v21_metrics = calculate_metrics(
        v21_test
    )

    print()
    print(
        f"TEST SAMPLES         : "
        f"{len(v1_test)}"
    )

    print()
    print(
        f"{'Metric':<20}"
        f"{'V1':>15}"
        f"{'V2':>15}"
        f"{'V2.1':>15}"
    )

    print(
        f"{'MAE':<20}"
        f"{v1_metrics['MAE']:>15.4f}"
        f"{v2_metrics['MAE']:>15.4f}"
        f"{v21_metrics['MAE']:>15.4f}"
    )

    print(
        f"{'RMSE':<20}"
        f"{v1_metrics['RMSE']:>15.4f}"
        f"{v2_metrics['RMSE']:>15.4f}"
        f"{v21_metrics['RMSE']:>15.4f}"
    )

    print(
        f"{'R2':<20}"
        f"{v1_metrics['R2']:>15.4f}"
        f"{v2_metrics['R2']:>15.4f}"
        f"{v21_metrics['R2']:>15.4f}"
    )

    print(
        f"{'MAPE':<20}"
        f"{v1_metrics['MAPE_percent']:>14.4f}%"
        f"{v2_metrics['MAPE_percent']:>14.4f}%"
        f"{v21_metrics['MAPE_percent']:>14.4f}%"
    )

    print(
        f"{'Bias':<20}"
        f"{v1_metrics['mean_error_bias']:>15.4f}"
        f"{v2_metrics['mean_error_bias']:>15.4f}"
        f"{v21_metrics['mean_error_bias']:>15.4f}"
    )

    # -------------------------------------------------------------------------
    # Winner determination
    # -------------------------------------------------------------------------

    model_names = [
        "V1",
        "V2",
        "V2.1",
    ]

    metric_values = {
        "MAE": [
            v1_metrics["MAE"],
            v2_metrics["MAE"],
            v21_metrics["MAE"],
        ],
        "RMSE": [
            v1_metrics["RMSE"],
            v2_metrics["RMSE"],
            v21_metrics["RMSE"],
        ],
        "R2": [
            v1_metrics["R2"],
            v2_metrics["R2"],
            v21_metrics["R2"],
        ],
        "MAPE": [
            v1_metrics["MAPE_percent"],
            v2_metrics["MAPE_percent"],
            v21_metrics["MAPE_percent"],
        ],
    }

    winners = {}

    for metric, values in metric_values.items():

        if metric == "R2":
            winner = model_names[
                int(np.nanargmax(values))
            ]
        else:
            winner = model_names[
                int(np.nanargmin(values))
            ]

        winners[metric] = winner

        print(
            f"{metric} winner".ljust(25)
            + f": {winner}"
        )

    # -------------------------------------------------------------------------
    # Comparison row
    # -------------------------------------------------------------------------

    row = {
        "gap_length": gap_length,
        "samples": v1_metrics["samples"],

        "V1_MAE": v1_metrics["MAE"],
        "V2_MAE": v2_metrics["MAE"],
        "V21_MAE": v21_metrics["MAE"],

        "V1_RMSE": v1_metrics["RMSE"],
        "V2_RMSE": v2_metrics["RMSE"],
        "V21_RMSE": v21_metrics["RMSE"],

        "V1_R2": v1_metrics["R2"],
        "V2_R2": v2_metrics["R2"],
        "V21_R2": v21_metrics["R2"],

        "V1_MAPE_percent": v1_metrics[
            "MAPE_percent"
        ],
        "V2_MAPE_percent": v2_metrics[
            "MAPE_percent"
        ],
        "V21_MAPE_percent": v21_metrics[
            "MAPE_percent"
        ],

        "V1_mean_error_bias": v1_metrics[
            "mean_error_bias"
        ],
        "V2_mean_error_bias": v2_metrics[
            "mean_error_bias"
        ],
        "V21_mean_error_bias": v21_metrics[
            "mean_error_bias"
        ],

        "V1_max_absolute_error": v1_metrics[
            "max_absolute_error"
        ],
        "V2_max_absolute_error": v2_metrics[
            "max_absolute_error"
        ],
        "V21_max_absolute_error": v21_metrics[
            "max_absolute_error"
        ],

        # V2.1 vs V1
        "V21_MAE_change_vs_V1_percent":
            percentage_improvement(
                v1_metrics["MAE"],
                v21_metrics["MAE"],
            ),

        "V21_RMSE_change_vs_V1_percent":
            percentage_improvement(
                v1_metrics["RMSE"],
                v21_metrics["RMSE"],
            ),

        "V21_R2_change_vs_V1":
            r2_change(
                v1_metrics["R2"],
                v21_metrics["R2"],
            ),

        "V21_MAPE_change_vs_V1_percent":
            percentage_improvement(
                v1_metrics["MAPE_percent"],
                v21_metrics["MAPE_percent"],
            ),

        # V2.1 vs V2
        "V21_MAE_change_vs_V2_percent":
            percentage_improvement(
                v2_metrics["MAE"],
                v21_metrics["MAE"],
            ),

        "V21_RMSE_change_vs_V2_percent":
            percentage_improvement(
                v2_metrics["RMSE"],
                v21_metrics["RMSE"],
            ),

        "V21_R2_change_vs_V2":
            r2_change(
                v2_metrics["R2"],
                v21_metrics["R2"],
            ),

        "V21_MAPE_change_vs_V2_percent":
            percentage_improvement(
                v2_metrics["MAPE_percent"],
                v21_metrics["MAPE_percent"],
            ),

        "MAE_winner": winners["MAE"],
        "RMSE_winner": winners["RMSE"],
        "R2_winner": winners["R2"],
        "MAPE_winner": winners["MAPE"],
    }

    detail = create_detail_dataframe(
        v1=v1_test,
        v2=v2_test,
        v21=v21_test,
    )

    return detail, row


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    ensure_directories()

    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION — "
        "V1 vs V2 vs V2.1 EVALUATION"
    )
    print("=" * 80)

    print()
    print("V1 model                 : FROZEN")
    print("V2 model                 : FROZEN")
    print("V2.1 model               : FROZEN")
    print(
        f"V1 features              : "
        f"{V1_FEATURE_COUNT}"
    )
    print(
        f"V2 features              : "
        f"{V2_FEATURE_COUNT}"
    )
    print(
        f"V2.1 features            : "
        f"{V21_FEATURE_COUNT}"
    )
    print(
        "Gap lengths              : "
        "1, 6, 24, 48 LP"
    )
    print(
        "Evaluation split         : TEST ONLY"
    )
    print(
        "Retraining               : DISABLED"
    )
    print(
        "Prediction modification  : DISABLED"
    )
    print(
        "Source modification      : DISABLED"
    )
    print(
        "MLflow                   : NOT YET"
    )

    all_rows: List[Dict[str, object]] = []
    all_details: List[pd.DataFrame] = []

    for gap_length in GAP_LENGTHS:

        detail, row = evaluate_gap(
            gap_length
        )

        all_details.append(
            detail
        )

        all_rows.append(
            row
        )

    # =========================================================================
    # CONSOLIDATED SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "V1 vs V2 vs V2.1 TEST PERFORMANCE SUMMARY"
    )
    print("=" * 80)

    summary = pd.DataFrame(
        all_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary_path = (
        SUMMARY_DIR
        / "v1_v2_v21_test_comparison.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # =========================================================================
    # V2.1 IMPROVEMENT SUMMARY
    # =========================================================================

    improvement_columns = [
        "gap_length",
        "samples",

        "V1_MAE",
        "V21_MAE",
        "V21_MAE_change_vs_V1_percent",

        "V1_RMSE",
        "V21_RMSE",
        "V21_RMSE_change_vs_V1_percent",

        "V1_R2",
        "V21_R2",
        "V21_R2_change_vs_V1",

        "V1_MAPE_percent",
        "V21_MAPE_percent",
        "V21_MAPE_change_vs_V1_percent",

        "V2_MAE",
        "V21_MAE_change_vs_V2_percent",

        "V2_RMSE",
        "V21_RMSE_change_vs_V2_percent",

        "V2_R2",
        "V21_R2_change_vs_V2",

        "V2_MAPE_percent",
        "V21_MAPE_change_vs_V2_percent",
    ]

    improvement = summary[
        improvement_columns
    ].copy()

    improvement_path = (
        SUMMARY_DIR
        / "v21_vs_v1_improvement.csv"
    )

    improvement.to_csv(
        improvement_path,
        index=False,
    )

    # =========================================================================
    # WIN COUNTS
    # =========================================================================

    print()
    print("-" * 80)
    print("METRIC WINNER COUNT")
    print("-" * 80)

    winner_counts = {}

    for metric in [
        "MAE_winner",
        "RMSE_winner",
        "R2_winner",
        "MAPE_winner",
    ]:

        counts = (
            summary[metric]
            .value_counts()
        )

        winner_counts[
            metric.replace("_winner", "")
        ] = {
            "V1": int(
                counts.get("V1", 0)
            ),
            "V2": int(
                counts.get("V2", 0)
            ),
            "V2.1": int(
                counts.get("V2.1", 0)
            ),
        }

    winner_dataframe = pd.DataFrame(
        winner_counts
    ).T

    winner_dataframe.index.name = "Metric"

    print(
        winner_dataframe.to_string()
    )

    winner_path = (
        SUMMARY_DIR
        / "metric_winner_counts_v21.csv"
    )

    winner_dataframe.to_csv(
        winner_path
    )

    # =========================================================================
    # BEST GAP
    # =========================================================================

    print()
    print("-" * 80)
    print("BEST GAP BY MAE")
    print("-" * 80)

    best_v1 = summary.loc[
        summary["V1_MAE"].idxmin()
    ]

    best_v2 = summary.loc[
        summary["V2_MAE"].idxmin()
    ]

    best_v21 = summary.loc[
        summary["V21_MAE"].idxmin()
    ]

    print(
        "V1 best gap   : "
        f"{int(best_v1['gap_length'])} LP "
        f"(MAE = {best_v1['V1_MAE']:.4f})"
    )

    print(
        "V2 best gap   : "
        f"{int(best_v2['gap_length'])} LP "
        f"(MAE = {best_v2['V2_MAE']:.4f})"
    )

    print(
        "V2.1 best gap : "
        f"{int(best_v21['gap_length'])} LP "
        f"(MAE = {best_v21['V21_MAE']:.4f})"
    )

    # =========================================================================
    # V2.1 CONCLUSION
    # =========================================================================

    v21_mae_wins = int(
        (
            summary["MAE_winner"]
            == "V2.1"
        ).sum()
    )

    v21_r2_wins = int(
        (
            summary["R2_winner"]
            == "V2.1"
        ).sum()
    )

    v21_mape_wins = int(
        (
            summary["MAPE_winner"]
            == "V2.1"
        ).sum()
    )

    v21_rmse_wins = int(
        (
            summary["RMSE_winner"]
            == "V2.1"
        ).sum()
    )

    print()
    print("-" * 80)
    print("V2.1 CONCLUSION")
    print("-" * 80)

    print(
        f"MAE wins  — V2.1: "
        f"{v21_mae_wins} / {len(summary)}"
    )

    print(
        f"RMSE wins — V2.1: "
        f"{v21_rmse_wins} / {len(summary)}"
    )

    print(
        f"R2 wins   — V2.1: "
        f"{v21_r2_wins} / {len(summary)}"
    )

    print(
        f"MAPE wins — V2.1: "
        f"{v21_mape_wins} / {len(summary)}"
    )

    # =========================================================================
    # DETAIL DATASET
    # =========================================================================

    details = pd.concat(
        all_details,
        ignore_index=True,
    )

    detail_path = (
        DETAIL_DIR
        / "v1_v2_v21_prediction_comparison.parquet"
    )

    details.to_parquet(
        detail_path,
        index=False,
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "evaluation_version": "v2.1",
        "comparison": [
            "V1",
            "V2",
            "V2.1",
        ],
        "features": {
            "V1": V1_FEATURE_COUNT,
            "V2": V2_FEATURE_COUNT,
            "V2.1": V21_FEATURE_COUNT,
        },
        "gap_lengths": GAP_LENGTHS,
        "evaluation_split": TEST_SPLIT,
        "retraining": False,
        "prediction_modification": False,
        "source_modification": False,
        "mlflow": False,
        "metric_winner_counts": winner_counts,
        "files": {
            "summary": str(
                summary_path
            ),
            "improvement": str(
                improvement_path
            ),
            "winner_counts": str(
                winner_path
            ),
            "detail": str(
                detail_path
            ),
        },
    }

    metadata_path = (
        SUMMARY_DIR
        / "evaluation_metadata_v21.json"
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

    # =========================================================================
    # RELOAD VALIDATION
    # =========================================================================

    summary_reload = pd.read_csv(
        summary_path
    )

    if len(summary_reload) != len(
        GAP_LENGTHS
    ):
        fail(
            "Summary reload validation failed."
        )

    detail_reload = pd.read_parquet(
        detail_path
    )

    if len(detail_reload) != len(
        details
    ):
        fail(
            "Detail reload validation failed."
        )

    print()
    print(
        "Summary reload         : PASSED"
    )

    print(
        "Detail reload          : PASSED"
    )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "V1 vs V2 vs V2.1 EVALUATION COMPLETE"
    )
    print("=" * 80)

    print()
    print("Summary:")
    print(
        f"    {summary_path}"
    )

    print()
    print("Improvement:")
    print(
        f"    {improvement_path}"
    )

    print()
    print("Winner counts:")
    print(
        f"    {winner_path}"
    )

    print()
    print("Detailed comparison:")
    print(
        f"    {detail_path}"
    )

    print()
    print("Metadata:")
    print(
        f"    {metadata_path}"
    )

    print()
    print(
        "V1 predictions          : UNCHANGED"
    )

    print(
        "V2 predictions          : UNCHANGED"
    )

    print(
        "V2.1 predictions        : UNCHANGED"
    )

    print(
        "Models                  : NOT RETRAINED"
    )

    print(
        "Source data             : UNCHANGED"
    )


if __name__ == "__main__":
    main()