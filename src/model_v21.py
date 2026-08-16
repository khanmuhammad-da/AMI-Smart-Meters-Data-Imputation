"""
RANDOM FOREST CDP IMPUTATION — V2.1 MODEL STAGE

Purpose
-------
Train and evaluate Random Forest models using the V2.1 supervised
feature datasets.

Experiment lineage
------------------
V1 : FROZEN
V2 : FROZEN
V2.1 : CURRENT

V2.1 feature set
----------------
22 V1 original features
+ 3 historical features
+ 4 recent-context features
= 29 predictor features

Important constraints
---------------------
- V1 source files are NOT modified.
- V2 source files are NOT modified.
- V1 models/predictions are NOT modified.
- V2 models/predictions are NOT modified.
- Source CSV is NOT modified.
- 96 LP gap remains removed.
- Sliding windows remain disabled.
- One prediction is generated per missing LP.
- Ground-truth leakage is prohibited.
- Current-gap values are not used as predictors.
- TEST set is used only for final performance reporting.
- Random Forest configuration is identical to V1/V2.
- MLflow is intentionally not included yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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
# CONSTANTS
# =============================================================================

VERSION = "v2.1"

GAP_LENGTHS = [1, 6, 24, 48]

EXPECTED_V1_FEATURE_COUNT = 22
EXPECTED_HISTORICAL_FEATURE_COUNT = 3
EXPECTED_RECENT_CONTEXT_FEATURE_COUNT = 4
EXPECTED_FEATURE_COUNT = 29

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

FEATURES = (
    V1_FEATURES
    + HISTORICAL_FEATURES
    + RECENT_CONTEXT_FEATURES
)

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
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    CONFIG["project_root"]
)

SUPERVISED_V21_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "supervised_v21"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model_v21"
)

MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METADATA_DIR = OUTPUT_DIR / "metadata"


# =============================================================================
# RANDOM FOREST PARAMETERS
# =============================================================================

RF_PARAMS = {
    "random_state": 42,
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def line(char: str = "-", length: int = 80) -> None:
    print(char * length)


def fail(message: str) -> None:
    raise RuntimeError(message)


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

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    # R2 is mathematically undefined for fewer than
    # two samples or constant targets.
    if len(y_true) < 2:
        r2 = float("nan")
    elif np.isclose(
        np.var(y_true),
        0.0,
    ):
        r2 = float("nan")
    else:
        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

    nonzero = np.abs(y_true) > 1e-12

    if np.any(nonzero):
        mape = float(
            np.mean(
                np.abs(
                    (
                        y_true[nonzero]
                        - y_pred[nonzero]
                    )
                    / y_true[nonzero]
                )
            )
            * 100.0
        )
    else:
        mape = float("nan")

    error = y_pred - y_true

    bias = float(
        np.mean(error)
    )

    max_absolute_error = float(
        np.max(
            np.abs(error)
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "mean_error_bias": bias,
        "max_absolute_error": max_absolute_error,
    }


# =============================================================================
# FEATURE VALIDATION
# =============================================================================

def validate_feature_definition() -> None:

    if len(V1_FEATURES) != EXPECTED_V1_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_V1_FEATURE_COUNT} V1 features, "
            f"found {len(V1_FEATURES)}."
        )

    if len(HISTORICAL_FEATURES) != (
        EXPECTED_HISTORICAL_FEATURE_COUNT
    ):
        fail(
            "Historical feature count mismatch."
        )

    if len(RECENT_CONTEXT_FEATURES) != (
        EXPECTED_RECENT_CONTEXT_FEATURE_COUNT
    ):
        fail(
            "Recent-context feature count mismatch."
        )

    if len(FEATURES) != EXPECTED_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_FEATURE_COUNT} V2.1 features, "
            f"found {len(FEATURES)}."
        )

    if len(set(FEATURES)) != len(FEATURES):
        fail(
            "Duplicate feature names detected."
        )

    prohibited = {
        "ground_truth",
        "prediction",
        "target",
    }

    leakage = [
        feature
        for feature in FEATURES
        if feature.lower() in prohibited
    ]

    if leakage:
        fail(
            f"Potential target leakage detected: {leakage}"
        )

    print(
        "Feature definition          : PASSED"
    )


# =============================================================================
# DATASET PATH
# =============================================================================

def supervised_path(
    gap_length: int,
) -> Path:

    return (
        SUPERVISED_V21_DIR
        / f"gap_{gap_length}"
        / (
            "CDP_00526_P_01_A-_"
            f"supervised_gap_{gap_length}_v21.parquet"
        )
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_input_dataset(
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
            "gap_position_fraction",
            "Time",
            "ground_truth",
            "feature_version",
        ]
    )

    missing = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        fail(
            f"Gap {gap_length}: missing required columns: "
            f"{missing}"
        )

    if len(dataframe) == 0:
        fail(
            f"Gap {gap_length}: dataset is empty."
        )

    if dataframe["feature_version"].astype(str).nunique() != 1:
        fail(
            f"Gap {gap_length}: multiple feature versions found."
        )

    feature_version = str(
        dataframe["feature_version"].iloc[0]
    )

    if feature_version != VERSION:
        fail(
            f"Gap {gap_length}: expected feature version "
            f"{VERSION}, found {feature_version}."
        )

    actual_gap_lengths = (
        dataframe["gap_length"]
        .astype(int)
        .unique()
        .tolist()
    )

    if actual_gap_lengths != [gap_length]:
        fail(
            f"Gap {gap_length}: unexpected gap lengths "
            f"{actual_gap_lengths}."
        )

    # -----------------------------------------------------------------
    # Feature NaN validation
    # -----------------------------------------------------------------

    for feature in FEATURES:

        if dataframe[feature].isna().any():
            count = int(
                dataframe[feature].isna().sum()
            )

            fail(
                f"Gap {gap_length}: feature "
                f"{feature} contains {count} NaN values."
            )

        if np.isinf(
            dataframe[feature].astype(float)
        ).any():
            fail(
                f"Gap {gap_length}: feature "
                f"{feature} contains infinite values."
            )

    # -----------------------------------------------------------------
    # Ground truth
    # -----------------------------------------------------------------

    if dataframe["ground_truth"].isna().any():
        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    # -----------------------------------------------------------------
    # Split validation
    # -----------------------------------------------------------------

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        dataframe["split"].astype(str).unique()
    )

    if not actual_splits.issubset(
        valid_splits
    ):
        fail(
            f"Gap {gap_length}: invalid split labels: "
            f"{actual_splits - valid_splits}"
        )

    for split in [
        "train",
        "validation",
        "test",
    ]:
        count = int(
            (
                dataframe["split"]
                == split
            ).sum()
        )

        if count == 0:
            fail(
                f"Gap {gap_length}: split '{split}' is empty."
            )

    # -----------------------------------------------------------------
    # Target-index uniqueness
    # -----------------------------------------------------------------

    if dataframe["target_index"].duplicated().any():
        fail(
            f"Gap {gap_length}: duplicate target_index values."
        )

    # -----------------------------------------------------------------
    # Expected number of samples
    # -----------------------------------------------------------------

    event_count = (
        dataframe["event_id"]
        .nunique()
    )

    expected_rows = (
        event_count
        * gap_length
    )

    if len(dataframe) != expected_rows:
        fail(
            f"Gap {gap_length}: expected "
            f"{expected_rows} rows from "
            f"{event_count} events × {gap_length} LP, "
            f"found {len(dataframe)}."
        )


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset(
    gap_length: int,
) -> pd.DataFrame:

    path = supervised_path(
        gap_length
    )

    if not path.exists():
        fail(
            f"V2.1 supervised dataset not found:\n{path}"
        )

    print("Input dataset:")
    print(f"    {path}")
    print(
        f"Rows                 : "
        f"{pd.read_parquet(path).shape[0]}"
    )

    dataframe = pd.read_parquet(
        path
    )

    print(
        f"Columns              : "
        f"{dataframe.shape[1]}"
    )

    validate_input_dataset(
        dataframe=dataframe,
        gap_length=gap_length,
    )

    print(
        "Input validation     : PASSED"
    )

    return dataframe


# =============================================================================
# SPLIT DATA
# =============================================================================

def prepare_split(
    dataframe: pd.DataFrame,
    split: str,
) -> Tuple[pd.DataFrame, pd.Series]:

    subset = dataframe[
        dataframe["split"].astype(str)
        == split
    ].copy()

    if subset.empty:
        fail(
            f"Split '{split}' contains no samples."
        )

    X = subset[
        FEATURES
    ].copy()

    y = subset[
        "ground_truth"
    ].astype(float)

    return X, y


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:

    model = RandomForestRegressor(
        **RF_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =============================================================================
# PREDICTION DATAFRAME
# =============================================================================

def create_prediction_dataframe(
    dataframe: pd.DataFrame,
    model: RandomForestRegressor,
) -> pd.DataFrame:

    X = dataframe[
        FEATURES
    ]

    predictions = model.predict(
        X
    )

    prediction_dataframe = dataframe[
        [
            "event_id",
            "gap_length",
            "split",
            "target_index",
            "gap_position",
            "gap_position_fraction",
            "Time",
            "ground_truth",
        ]
    ].copy()

    prediction_dataframe[
        "prediction"
    ] = predictions.astype(float)

    prediction_dataframe[
        "error"
    ] = (
        prediction_dataframe["prediction"]
        - prediction_dataframe["ground_truth"]
    )

    prediction_dataframe[
        "absolute_error"
    ] = np.abs(
        prediction_dataframe["error"]
    )

    prediction_dataframe[
        "squared_error"
    ] = (
        prediction_dataframe["error"]
        ** 2
    )

    denominator = (
        prediction_dataframe["ground_truth"]
        .abs()
    )

    prediction_dataframe[
        "percentage_error"
    ] = np.where(
        denominator > 1e-12,
        (
            prediction_dataframe[
                "absolute_error"
            ]
            / denominator
        )
        * 100.0,
        np.nan,
    )

    prediction_dataframe[
        "feature_version"
    ] = VERSION

    return prediction_dataframe


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(
    model: RandomForestRegressor,
    gap_length: int,
) -> Path:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        MODEL_DIR
        / f"random_forest_v21_gap_{gap_length}.joblib"
    )

    joblib.dump(
        model,
        path,
    )

    return path


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    prediction_dataframe: pd.DataFrame,
    gap_length: int,
) -> Path:

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        PREDICTION_DIR
        / f"predictions_gap_{gap_length}_v21.parquet"
    )

    prediction_dataframe.to_parquet(
        path,
        index=False,
    )

    return path


# =============================================================================
# SAVE METADATA
# =============================================================================

def save_metadata(
    gap_length: int,
    dataframe: pd.DataFrame,
    metrics: Dict[str, float],
    model_path: Path,
    prediction_path: Path,
) -> Path:

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        METADATA_DIR
        / f"model_metadata_gap_{gap_length}_v21.json"
    )

    split_counts = (
        dataframe["split"]
        .value_counts()
        .to_dict()
    )

    metadata = {
        "version": VERSION,
        "gap_length": gap_length,
        "algorithm": "RandomForestRegressor",
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "v1_feature_count": len(V1_FEATURES),
        "historical_feature_count": len(
            HISTORICAL_FEATURES
        ),
        "recent_context_feature_count": len(
            RECENT_CONTEXT_FEATURES
        ),
        "random_forest_parameters": RF_PARAMS,
        "samples": {
            "total": int(len(dataframe)),
            "train": int(
                split_counts.get(
                    "train",
                    0,
                )
            ),
            "validation": int(
                split_counts.get(
                    "validation",
                    0,
                )
            ),
            "test": int(
                split_counts.get(
                    "test",
                    0,
                )
            ),
        },
        "test_metrics": {
            key: (
                None
                if pd.isna(value)
                else float(value)
            )
            for key, value in metrics.items()
        },
        "ground_truth_leakage": False,
        "current_gap_feature_use": False,
        "sliding_windows": False,
        "one_prediction_per_missing_lp": True,
        "mlflow": False,
        "source_modified": False,
        "model_path": str(
            model_path
        ),
        "prediction_path": str(
            prediction_path
        ),
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    return path


# =============================================================================
# TRAIN ONE GAP
# =============================================================================

def process_gap(
    gap_length: int,
) -> Dict[str, float]:

    print()
    line()
    print(
        f"V2.1 RANDOM FOREST — GAP "
        f"{gap_length} LP"
    )
    line()

    dataframe = load_dataset(
        gap_length
    )

    # -----------------------------------------------------------------
    # Split preparation
    # -----------------------------------------------------------------

    X_train, y_train = prepare_split(
        dataframe,
        "train",
    )

    X_validation, y_validation = prepare_split(
        dataframe,
        "validation",
    )

    X_test, y_test = prepare_split(
        dataframe,
        "test",
    )

    print(
        f"Train samples        : "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples   : "
        f"{len(X_validation)}"
    )

    print(
        f"Test samples         : "
        f"{len(X_test)}"
    )

    # -----------------------------------------------------------------
    # Explicit feature-shape validation
    # -----------------------------------------------------------------

    if X_train.shape[1] != EXPECTED_FEATURE_COUNT:
        fail(
            f"Gap {gap_length}: expected "
            f"{EXPECTED_FEATURE_COUNT} predictors, "
            f"found {X_train.shape[1]}."
        )

    if list(X_train.columns) != FEATURES:
        fail(
            f"Gap {gap_length}: feature ordering mismatch."
        )

    # Validation data is intentionally NOT used to tune
    # the model in this stage. It is retained for experiment
    # lineage and future hyperparameter selection.
    _ = X_validation
    _ = y_validation

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    print()
    print(
        "Training Random Forest..."
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    print(
        "Model training       : PASSED"
    )

    # -----------------------------------------------------------------
    # Test prediction
    # -----------------------------------------------------------------

    y_test_pred = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_test_pred,
    )

    print()
    print(
        f"V2.1 TEST PERFORMANCE — "
        f"{gap_length} LP"
    )

    print(
        f"Samples              : "
        f"{len(y_test)}"
    )

    print(
        f"MAE                  : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"RMSE                 : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"R2                   : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"MAPE                 : "
        f"{metrics['MAPE_percent']:.4f}%"
    )

    print(
        f"Mean error bias      : "
        f"{metrics['mean_error_bias']:.4f}"
    )

    print(
        f"Maximum abs error    : "
        f"{metrics['max_absolute_error']:.4f}"
    )

    # -----------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------

    model_path = save_model(
        model=model,
        gap_length=gap_length,
    )

    print(
        "Saved model:"
    )
    print(
        f"    {model_path}"
    )

    # -----------------------------------------------------------------
    # Generate predictions for ALL samples
    # -----------------------------------------------------------------

    prediction_dataframe = (
        create_prediction_dataframe(
            dataframe=dataframe,
            model=model,
        )
    )

    # Ensure test predictions exactly match
    # the metrics calculated above.
    test_prediction_dataframe = (
        prediction_dataframe[
            prediction_dataframe["split"]
            == "test"
        ]
        .sort_values(
            "target_index"
        )
        .reset_index(drop=True)
    )

    expected_test = pd.DataFrame(
        {
            "target_index": dataframe.loc[
                dataframe["split"] == "test",
                "target_index",
            ].astype(int).to_numpy(),
            "prediction": y_test_pred,
        }
    ).sort_values(
        "target_index"
    ).reset_index(drop=True)

    if not np.allclose(
        test_prediction_dataframe[
            "prediction"
        ].to_numpy(),
        expected_test[
            "prediction"
        ].to_numpy(),
        rtol=1e-10,
        atol=1e-8,
    ):
        fail(
            f"Gap {gap_length}: saved prediction "
            "verification failed."
        )

    prediction_path = save_predictions(
        prediction_dataframe=prediction_dataframe,
        gap_length=gap_length,
    )

    print(
        "Saved predictions:"
    )
    print(
        f"    {prediction_path}"
    )

    # -----------------------------------------------------------------
    # Reload prediction verification
    # -----------------------------------------------------------------

    reloaded = pd.read_parquet(
        prediction_path
    )

    if reloaded.shape != (
        prediction_dataframe.shape
    ):
        fail(
            f"Gap {gap_length}: prediction reload "
            "shape mismatch."
        )

    print(
        "Prediction reload    : PASSED"
    )

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    metadata_path = save_metadata(
        gap_length=gap_length,
        dataframe=dataframe,
        metrics=metrics,
        model_path=model_path,
        prediction_path=prediction_path,
    )

    print(
        "Saved metadata:"
    )
    print(
        f"    {metadata_path}"
    )

    return {
        "gap_length": gap_length,
        "samples": int(len(y_test)),
        **metrics,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION — "
        "V2.1 MODEL STAGE"
    )
    print("=" * 80)

    print()
    print(
        "V1 model                 : FROZEN"
    )
    print(
        "V2 model                 : FROZEN"
    )
    print(
        "V2.1 feature version     : v2.1"
    )
    print(
        "Algorithm                : "
        "RandomForestRegressor"
    )
    print(
        "Gap lengths              : "
        "1, 6, 24, 48 LP"
    )
    print(
        "96 LP gap                : REMOVED"
    )
    print(
        "One prediction / missing LP : ENABLED"
    )
    print(
        f"V1 predictor features    : "
        f"{len(V1_FEATURES)}"
    )
    print(
        f"Historical features      : "
        f"{len(HISTORICAL_FEATURES)}"
    )
    print(
        f"Recent-context features  : "
        f"{len(RECENT_CONTEXT_FEATURES)}"
    )
    print(
        f"V2.1 predictor features  : "
        f"{len(FEATURES)}"
    )
    print(
        "Ground-truth leakage     : PROHIBITED"
    )
    print(
        "Current-gap feature use  : PROHIBITED"
    )
    print(
        "MLflow                   : NOT YET"
    )
    print(
        "Source modification      : DISABLED"
    )

    print()
    print(
        "V2.1 FEATURE GROUPS"
    )
    line()

    print(
        f"V1 original features     : "
        f"{len(V1_FEATURES)}"
    )

    print(
        f"Historical features      : "
        f"{len(HISTORICAL_FEATURES)}"
    )

    print(
        f"Recent-context features  : "
        f"{len(RECENT_CONTEXT_FEATURES)}"
    )

    print(
        f"TOTAL V2.1 FEATURES      : "
        f"{len(FEATURES)}"
    )

    print()
    print(
        "RANDOM FOREST PARAMETERS"
    )

    for key, value in RF_PARAMS.items():
        print(
            f"    {key}: {value}"
        )

    print()

    validate_feature_definition()

    results: List[Dict[str, float]] = []

    for gap_length in GAP_LENGTHS:

        result = process_gap(
            gap_length
        )

        results.append(
            result
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = pd.DataFrame(
        results
    )

    print()
    line("=")
    print(
        "V2.1 MODEL TEST PERFORMANCE SUMMARY"
    )
    line()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "v21_test_performance_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "Summary saved:"
    )
    print(
        f"    {summary_path}"
    )

    print()
    print("=" * 80)
    print(
        "V2.1 RANDOM FOREST MODEL STAGE COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "V1 model.py              : UNCHANGED"
    )
    print(
        "V2 model_v2.py           : UNCHANGED"
    )
    print(
        "V2.1 supervised datasets : USED"
    )
    print(
        "V2.1 Random Forest       : TRAINED"
    )
    print(
        "V1 predictions           : UNCHANGED"
    )
    print(
        "V2 predictions           : UNCHANGED"
    )
    print(
        "V2.1 predictions         : CREATED"
    )
    print(
        "MLflow                   : NOT YET"
    )
    print(
        "Source CSV               : UNCHANGED"
    )

    print()
    print(
        "Model directory:"
    )
    print(
        f"    {MODEL_DIR}"
    )

    print(
        "Prediction directory:"
    )
    print(
        f"    {PREDICTION_DIR}"
    )

    print(
        "Metadata directory:"
    )
    print(
        f"    {METADATA_DIR}"
    )


if __name__ == "__main__":
    main()