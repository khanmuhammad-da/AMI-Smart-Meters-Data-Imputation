"""
RANDOM FOREST CDP IMPUTATION — V2 MODEL STAGE

V2 model = V2 supervised feature layer + Random Forest.

V1 is NOT modified.

V2 features:
    22 V1 deterministic/calendar features
    21 additional V2 features
    = 43 predictor features

Gap lengths:
    1, 6, 24, 48 LP

Prediction:
    One prediction per missing LP.

Data split:
    Chronological split already established by gap_generator.py.
    Existing split labels are preserved.

Leakage prevention:
    ground_truth is never used as a predictor.
    target is never used as a predictor.
    prediction is never used as a predictor.
    event metadata is not used as a predictor.
    artificial-gap values are not used as predictors.

Outputs:
    outputs/model_v2/
        models/
        predictions/
        metadata/

V1 model.py remains untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import CONFIG


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    CONFIG["project_root"]
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / CONFIG["outputs"]["processed_data_dir"]
)

V2_SUPERVISED_DIR = (
    PROCESSED_DIR
    / "supervised_v2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_v2"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "models"
)

PREDICTION_DIR = (
    OUTPUT_DIR
    / "predictions"
)

METADATA_DIR = (
    OUTPUT_DIR
    / "metadata"
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

GAP_LENGTHS = [
    int(x)
    for x in CONFIG["gaps"]["lengths"]
]

METER_ID = CONFIG["data"]["selected_meter"]

TARGET_DIRECTION = CONFIG["data"]["target_direction"]


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
# V2 FEATURES
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


FEATURES = (
    V1_FEATURES
    + V2_FEATURES
)


# =============================================================================
# NON-PREDICTOR COLUMNS
# =============================================================================

NON_FEATURE_COLUMNS = {
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
# PATH HELPERS
# =============================================================================

def supervised_path(
    gap_length: int,
) -> Path:

    return (
        V2_SUPERVISED_DIR
        / f"gap_{gap_length}"
        / (
            f"{METER_ID}_{TARGET_DIRECTION}"
            f"_supervised_gap_{gap_length}_v2.parquet"
        )
    )


def model_path(
    gap_length: int,
) -> Path:

    return (
        MODEL_DIR
        / (
            f"random_forest_v2_gap_{gap_length}.joblib"
        )
    )


def prediction_path(
    gap_length: int,
) -> Path:

    return (
        PREDICTION_DIR
        / (
            f"predictions_gap_{gap_length}_v2.parquet"
        )
    )


def metadata_path(
    gap_length: int,
) -> Path:

    return (
        METADATA_DIR
        / (
            f"model_metadata_gap_{gap_length}_v2.json"
        )
    )


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

def get_model_parameters() -> Dict:
    """
    Read Random Forest parameters from config/config.yaml.

    The function intentionally accepts the parameters already
    defined in the project's model configuration.

    Parameters not recognized by RandomForestRegressor are ignored.
    """

    if "model" not in CONFIG:
        fail(
            "config.yaml does not contain a 'model' section."
        )

    model_config = CONFIG["model"]

    if not isinstance(
        model_config,
        dict,
    ):
        fail(
            "CONFIG['model'] must be a dictionary."
        )

    algorithm = str(
        model_config.get(
            "algorithm",
            "RandomForestRegressor",
        )
    )

    normalized_algorithm = (
        algorithm
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    if normalized_algorithm not in {
        "randomforestregressor",
        "randomforest",
    }:
        fail(
            "V2 requires RandomForestRegressor. "
            f"Configured algorithm: {algorithm}"
        )

    valid_parameters = {
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "min_weight_fraction_leaf",
        "max_features",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "bootstrap",
        "oob_score",
        "n_jobs",
        "random_state",
        "verbose",
        "warm_start",
        "ccp_alpha",
        "max_samples",
    }

    parameters = {}

    for key, value in model_config.items():

        if key in valid_parameters:
            parameters[key] = value

    # Ensure reproducibility if not explicitly specified.
    parameters.setdefault(
        "random_state",
        42,
    )

    # Use all available CPU cores unless explicitly configured.
    parameters.setdefault(
        "n_jobs",
        -1,
    )

    return parameters


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_input(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:

    required_columns = set(
        FEATURES
        + [
            "event_id",
            "gap_length",
            "split",
            "target_index",
            "gap_position",
            "Time",
            "ground_truth",
        ]
    )

    missing = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        fail(
            f"Gap {gap_length}: missing required columns:\n"
            f"{missing}"
        )

    if dataframe.empty:
        fail(
            f"Gap {gap_length}: dataset is empty."
        )

    # -------------------------------------------------------------------------
    # Gap length
    # -------------------------------------------------------------------------

    if not (
        dataframe["gap_length"]
        == gap_length
    ).all():

        fail(
            f"Gap {gap_length}: gap_length column contains "
            "unexpected values."
        )

    # -------------------------------------------------------------------------
    # Split labels
    # -------------------------------------------------------------------------

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
            f"Gap {gap_length}: unexpected split labels: "
            f"{actual_splits}"
        )

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    if dataframe[
        "ground_truth"
    ].isna().any():

        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    ground_truth = dataframe[
        "ground_truth"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        ground_truth
    ).all():

        fail(
            f"Gap {gap_length}: ground_truth contains "
            "non-finite values."
        )

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    for feature in FEATURES:

        values = dataframe[
            feature
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            fail(
                f"Gap {gap_length}: feature "
                f"'{feature}' contains NaN/Inf."
            )

    # -------------------------------------------------------------------------
    # Previous-week availability
    # -------------------------------------------------------------------------

    availability = set(
        dataframe[
            "previous_week_available"
        ]
        .astype(int)
        .unique()
    )

    if not availability.issubset(
        {0, 1}
    ):

        fail(
            f"Gap {gap_length}: "
            "previous_week_available must contain "
            "only 0 and 1."
        )

    # -------------------------------------------------------------------------
    # Feature version
    # -------------------------------------------------------------------------

    if "feature_version" in dataframe.columns:

        versions = set(
            dataframe[
                "feature_version"
            ]
            .astype(str)
            .unique()
        )

        if versions != {"v2"}:

            fail(
                f"Gap {gap_length}: unexpected feature "
                f"versions: {versions}"
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
            f"Gap {gap_length}: invalid timestamps."
        )

    # -------------------------------------------------------------------------
    # One sample per missing LP
    # -------------------------------------------------------------------------

    expected_count = (
        dataframe
        .groupby("event_id")
        .size()
    )

    invalid_events = expected_count[
        expected_count != gap_length
    ]

    if not invalid_events.empty:

        fail(
            f"Gap {gap_length}: events do not contain "
            f"exactly {gap_length} samples:\n"
            f"{invalid_events}"
        )


# =============================================================================
# FEATURE LEAKAGE VALIDATION
# =============================================================================

def validate_features() -> None:

    if len(FEATURES) != 43:

        fail(
            "V2 feature count must be 43. "
            f"Found {len(FEATURES)}."
        )

    duplicates = (
        pd.Series(FEATURES)
        .duplicated()
    )

    if duplicates.any():

        duplicate_features = (
            pd.Series(FEATURES)[
                duplicates
            ].tolist()
        )

        fail(
            "Duplicate model features detected: "
            f"{duplicate_features}"
        )

    forbidden = {
        "target",
        "ground_truth",
        "prediction",
        "Time",
        "split",
        "event_id",
        "gap_length",
        "target_index",
    }

    leakage = (
        set(FEATURES)
        & forbidden
    )

    if leakage:

        fail(
            f"Forbidden predictor columns detected: "
            f"{leakage}"
        )


# =============================================================================
# TRAIN RANDOM FOREST
# =============================================================================

def train_model(
    train_dataframe: pd.DataFrame,
    parameters: Dict,
) -> RandomForestRegressor:

    X_train = train_dataframe[
        FEATURES
    ]

    y_train = train_dataframe[
        "ground_truth"
    ]

    model = RandomForestRegressor(
        **parameters
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =============================================================================
# PREDICTION
# =============================================================================

def create_predictions(
    model: RandomForestRegressor,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    X = dataframe[
        FEATURES
    ]

    predictions = model.predict(
        X
    )

    if not np.isfinite(
        predictions
    ).all():

        fail(
            "Model produced NaN/Inf predictions."
        )

    result = dataframe[
        [
            "event_id",
            "gap_length",
            "split",
            "target_index",
            "gap_position",
            "Time",
            "ground_truth",
        ]
    ].copy()

    result[
        "prediction"
    ] = predictions.astype(float)

    result[
        "error"
    ] = (
        result["prediction"]
        - result["ground_truth"]
    )

    result[
        "absolute_error"
    ] = result[
        "error"
    ].abs()

    result[
        "squared_error"
    ] = result[
        "error"
    ] ** 2

    result[
        "percentage_error"
    ] = np.where(
        result[
            "ground_truth"
        ].abs() > 0,

        (
            result[
                "absolute_error"
            ]
            /
            result[
                "ground_truth"
            ].abs()
        )
        * 100.0,

        np.nan,
    )

    result[
        "feature_version"
    ] = "v2"

    return result


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> Dict:

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
                        y_true[nonzero]
                        - y_pred[nonzero]
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
            len(dataframe)
        ),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": mape,
        "mean_error_bias": bias,
        "max_absolute_error": max_abs_error,
    }


# =============================================================================
# SAVE MODEL METADATA
# =============================================================================

def save_metadata(
    gap_length: int,
    parameters: Dict,
    model: RandomForestRegressor,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    test_metrics: Dict,
) -> None:

    payload = {

        "model_version": "v2",

        "feature_version": "v2",

        "algorithm": (
            "RandomForestRegressor"
        ),

        "meter_id": METER_ID,

        "direction": TARGET_DIRECTION,

        "gap_length": gap_length,

        "feature_count": len(
            FEATURES
        ),

        "v1_feature_count": len(
            V1_FEATURES
        ),

        "v2_feature_count": len(
            V2_FEATURES
        ),

        "features": FEATURES,

        "v1_features": V1_FEATURES,

        "v2_features": V2_FEATURES,

        "model_parameters": parameters,

        "n_features_in": int(
            model.n_features_in_
        ),

        "n_estimators_fitted": int(
            len(model.estimators_)
        ),

        "train_samples": int(
            len(train_dataframe)
        ),

        "validation_samples": int(
            len(validation_dataframe)
        ),

        "test_samples": int(
            len(test_dataframe)
        ),

        "test_metrics": test_metrics,

        "ground_truth_used_as_feature": False,

        "target_used_as_feature": False,

        "prediction_used_as_feature": False,

        "source_modified": False,

        "mlflow_enabled": False,
    }

    path = metadata_path(
        gap_length
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
            default=str,
        )


# =============================================================================
# PROCESS ONE GAP
# =============================================================================

def process_gap(
    gap_length: int,
    parameters: Dict,
) -> Dict:

    section(
        f"V2 RANDOM FOREST — GAP {gap_length} LP"
    )

    input_path = supervised_path(
        gap_length
    )

    if not input_path.exists():

        fail(
            f"V2 supervised dataset not found:\n"
            f"{input_path}"
        )

    info(
        f"Input dataset:\n    {input_path}"
    )

    dataframe = pd.read_parquet(
        input_path
    )

    info(
        f"Rows                 : "
        f"{len(dataframe):,}"
    )

    info(
        f"Columns              : "
        f"{len(dataframe.columns)}"
    )

    # -------------------------------------------------------------------------
    # Validate.
    # -------------------------------------------------------------------------

    validate_input(
        dataframe,
        gap_length,
    )

    info(
        "Input validation     : PASSED"
    )

    # -------------------------------------------------------------------------
    # Separate splits.
    # -------------------------------------------------------------------------

    train_dataframe = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_dataframe = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test_dataframe = dataframe[
        dataframe["split"] == "test"
    ].copy()

    if train_dataframe.empty:
        fail(
            f"Gap {gap_length}: training dataset is empty."
        )

    if validation_dataframe.empty:
        fail(
            f"Gap {gap_length}: validation dataset is empty."
        )

    if test_dataframe.empty:
        fail(
            f"Gap {gap_length}: test dataset is empty."
        )

    info(
        f"Train samples        : "
        f"{len(train_dataframe):,}"
    )

    info(
        f"Validation samples   : "
        f"{len(validation_dataframe):,}"
    )

    info(
        f"Test samples         : "
        f"{len(test_dataframe):,}"
    )

    # -------------------------------------------------------------------------
    # Train.
    # -------------------------------------------------------------------------

    info()
    info(
        "Training Random Forest..."
    )

    model = train_model(
        train_dataframe=train_dataframe,
        parameters=parameters,
    )

    info(
        "Model training       : PASSED"
    )

    # -------------------------------------------------------------------------
    # Predictions.
    #
    # We predict all samples so that train/validation/test predictions
    # are available for diagnostics.
    # -------------------------------------------------------------------------

    predictions = create_predictions(
        model=model,
        dataframe=dataframe,
    )

    # -------------------------------------------------------------------------
    # Test metrics.
    # -------------------------------------------------------------------------

    test_predictions = predictions[
        predictions["split"] == "test"
    ].copy()

    test_metrics = calculate_metrics(
        test_predictions
    )

    info()
    info(
        f"V2 TEST PERFORMANCE — "
        f"{gap_length} LP"
    )

    info(
        f"Samples              : "
        f"{test_metrics['samples']}"
    )

    info(
        f"MAE                  : "
        f"{test_metrics['MAE']:.4f}"
    )

    info(
        f"RMSE                 : "
        f"{test_metrics['RMSE']:.4f}"
    )

    info(
        f"R2                   : "
        f"{test_metrics['R2']:.4f}"
    )

    info(
        f"MAPE                 : "
        f"{test_metrics['MAPE_percent']:.4f}%"
    )

    info(
        f"Mean error bias      : "
        f"{test_metrics['mean_error_bias']:.4f}"
    )

    info(
        f"Maximum abs error    : "
        f"{test_metrics['max_absolute_error']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Save model.
    # -------------------------------------------------------------------------

    model_file = model_path(
        gap_length
    )

    model_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_file,
    )

    info()
    info(
        f"Saved model:\n    {model_file}"
    )

    # -------------------------------------------------------------------------
    # Save predictions.
    # -------------------------------------------------------------------------

    prediction_file = prediction_path(
        gap_length
    )

    prediction_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        prediction_file,
        index=False,
    )

    info(
        f"Saved predictions:\n    "
        f"{prediction_file}"
    )

    # -------------------------------------------------------------------------
    # Verify prediction file.
    # -------------------------------------------------------------------------

    reloaded = pd.read_parquet(
        prediction_file
    )

    if len(reloaded) != len(
        predictions
    ):

        fail(
            f"Gap {gap_length}: "
            "prediction reload row count mismatch."
        )

    if not np.allclose(
        reloaded["prediction"].to_numpy(
            dtype=float
        ),
        predictions["prediction"].to_numpy(
            dtype=float
        ),
    ):

        fail(
            f"Gap {gap_length}: "
            "prediction reload mismatch."
        )

    info(
        "Prediction reload    : PASSED"
    )

    # -------------------------------------------------------------------------
    # Save metadata.
    # -------------------------------------------------------------------------

    save_metadata(
        gap_length=gap_length,
        parameters=parameters,
        model=model,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
        test_metrics=test_metrics,
    )

    info(
        f"Saved metadata:\n    "
        f"{metadata_path(gap_length)}"
    )

    return {
        "gap_length": gap_length,
        **test_metrics,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)

    print(
        "RANDOM FOREST CDP IMPUTATION — V2 MODEL STAGE"
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
        "Algorithm                   : RandomForestRegressor"
    )
    info(
        "Gap lengths                 : 1, 6, 24, 48 LP"
    )
    info(
        "96 LP gap                   : REMOVED"
    )
    info(
        "One prediction / missing LP : ENABLED"
    )
    info(
        "V2 predictor features       : 43"
    )
    info(
        "V1 predictor features       : 22"
    )
    info(
        "New V2 features             : 21"
    )
    info(
        "Ground-truth leakage        : PROHIBITED"
    )
    info(
        "Current-gap feature use     : PROHIBITED"
    )
    info(
        "MLflow                      : NOT YET"
    )
    info(
        "Source modification         : DISABLED"
    )

    # -------------------------------------------------------------------------
    # Validate feature definition.
    # -------------------------------------------------------------------------

    validate_features()

    info()
    info(
        "Feature definition          : PASSED"
    )

    # -------------------------------------------------------------------------
    # Load model parameters.
    # -------------------------------------------------------------------------

    parameters = get_model_parameters()

    info()
    info(
        "RANDOM FOREST PARAMETERS"
    )

    for key, value in parameters.items():

        info(
            f"    {key}: {value}"
        )

    # -------------------------------------------------------------------------
    # Create output directories.
    # -------------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Train each gap independently.
    # -------------------------------------------------------------------------

    results = []

    for gap_length in GAP_LENGTHS:

        result = process_gap(
            gap_length=gap_length,
            parameters=parameters,
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # Consolidated summary.
    # -------------------------------------------------------------------------

    section(
        "V2 MODEL TEST PERFORMANCE SUMMARY"
    )

    summary = pd.DataFrame(
        results
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary_path = (
        OUTPUT_DIR
        / "v2_test_performance_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    info()
    info(
        f"Summary saved:\n    "
        f"{summary_path}"
    )

    # -------------------------------------------------------------------------
    # Final.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "V2 RANDOM FOREST MODEL STAGE COMPLETE"
    )
    print("=" * 80)

    info()
    info(
        "V1 model.py              : UNCHANGED"
    )
    info(
        "V2 supervised datasets   : USED"
    )
    info(
        "V2 Random Forest         : TRAINED"
    )
    info(
        "V1 predictions           : UNCHANGED"
    )
    info(
        "V2 predictions           : CREATED"
    )
    info(
        "MLflow                   : NOT YET"
    )
    info(
        "Source CSV               : UNCHANGED"
    )

    info()
    info(
        f"Model directory:\n    {MODEL_DIR}"
    )

    info(
        f"Prediction directory:\n    "
        f"{PREDICTION_DIR}"
    )


if __name__ == "__main__":
    main()