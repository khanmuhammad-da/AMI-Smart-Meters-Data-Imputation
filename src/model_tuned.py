"""
src/model_tuned.py

RANDOM FOREST CDP IMPUTATION — V1 TUNED MODEL

Purpose
-------
Fine-tune the existing V1 Random Forest using the existing V1
supervised datasets.

IMPORTANT
---------
This module does NOT:
    - modify V1 datasets
    - modify V1 models
    - modify V1 predictions
    - use JMR
    - use an inference CSV/XLSX
    - perform JMR reconciliation
    - use the TEST set for hyperparameter selection

Training strategy
-----------------
1. Load existing V1 supervised datasets.
2. Separate TRAIN / VALIDATION / TEST.
3. Search Random Forest hyperparameters using TRAIN.
4. Select the best configuration using VALIDATION MAE.
5. Retrain the selected configuration on TRAIN + VALIDATION.
6. Evaluate once on TEST.
7. Save tuned model, predictions, metrics, feature importance,
   and metadata.
8. Track each gap-specific model and the parent experiment in local MLflow.

Gap lengths
-----------
1, 6, 24, 48 LP

Each gap length receives its own independent Random Forest model.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*"
)

# scikit-learn/joblib versions may emit a non-fatal warning about the
# internal delayed() implementation used by RandomForestRegressor.
# It does not indicate a model-training failure, so suppress only this
# specific warning while keeping other warnings/errors visible.
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPERVISED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "supervised"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "model_tuned"
)

MODEL_ROOT = (
    OUTPUT_ROOT
    / "models"
)

PREDICTION_ROOT = (
    OUTPUT_ROOT
    / "predictions"
)

METRICS_ROOT = (
    OUTPUT_ROOT
    / "metrics"
)

FEATURE_IMPORTANCE_ROOT = (
    OUTPUT_ROOT
    / "feature_importance"
)

METADATA_ROOT = (
    OUTPUT_ROOT
    / "metadata"
)

SUMMARY_ROOT = (
    OUTPUT_ROOT
    / "summary"
)

# MLflow is deliberately local for the current project stage.
# This creates a portable local tracking store inside the repository.
MLFLOW_ROOT = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "AMI-Smart-Meters-V1-Tuned"


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

GAP_LENGTHS = [1, 6, 24, 48]

RANDOM_STATE = 42

TARGET_COLUMN = "ground_truth"

SPLIT_COLUMN = "split"

# V1 metadata columns — these must NOT be supplied to the model.
NON_FEATURE_COLUMNS = {
    "event_id",
    "gap_length",
    "split",
    "target_index",
    "gap_position",
    "gap_position_fraction",
    "Time",
    "ground_truth",
    "prediction",
}


# =============================================================================
# TUNING SEARCH SPACE
# =============================================================================
#
# This is intentionally controlled because the project has a limited
# execution window.
#
# We are not doing an enormous GridSearchCV.
#
# Each combination is trained on TRAIN and evaluated on VALIDATION.
#
# The TEST set remains completely untouched until the final model.
# =============================================================================

PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.7],
}


# =============================================================================
# LOGGING
# =============================================================================

def info(message: str) -> None:
    print(message)


def section(title: str) -> None:
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# MLFLOW SETUP
# =============================================================================

def configure_mlflow() -> None:
    """
    Configure MLflow to use a local SQLite tracking backend.

    SQLite is used instead of the deprecated filesystem backend
    (./mlruns), which is no longer accepted by recent MLflow versions.
    """

    mlflow_db = PROJECT_ROOT / "mlflow.db"

    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    print()
    print("MLflow configuration")
    print("-" * 80)
    print(f"Tracking URI         : {tracking_uri}")
    print(f"Experiment            : {MLFLOW_EXPERIMENT_NAME}")

def safe_log_param(key: str, value) -> None:
    """Log a scalar MLflow parameter after normalizing its type."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = "None"

    if isinstance(value, np.generic):
        value = value.item()

    mlflow.log_param(key, value)


def safe_log_metric(key: str, value) -> None:
    """Log a finite numeric MLflow metric when available."""

    if value is None:
        return

    value = float(value)
    if np.isfinite(value):
        mlflow.log_metric(key, value)


def log_tuning_results(tuning_results: pd.DataFrame, gap_length: int) -> Path:
    """Save and log the complete validation tuning table."""

    tuning_path = (
        SUMMARY_ROOT
        / f"tuning_results_gap_{gap_length}.csv"
    )

    tuning_results.to_csv(
        tuning_path,
        index=False,
    )

    mlflow.log_artifact(
    str(tuning_path),
    artifact_path="tuning",
    )

    return tuning_path


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def create_output_directories() -> None:

    for directory in [
        MODEL_ROOT,
        PREDICTION_ROOT,
        METRICS_ROOT,
        FEATURE_IMPORTANCE_ROOT,
        METADATA_ROOT,
        SUMMARY_ROOT,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def find_supervised_dataset(gap_length: int) -> Path:

    gap_dir = SUPERVISED_ROOT / f"gap_{gap_length}"

    if not gap_dir.exists():
        fail(
            f"Supervised dataset directory not found:\n"
            f"{gap_dir}"
        )

    candidates = sorted(
        gap_dir.glob("*_supervised_gap_*.parquet")
    )

    if not candidates:
        fail(
            f"No supervised dataset found for gap "
            f"{gap_length} LP in:\n{gap_dir}"
        )

    if len(candidates) > 1:
        info(
            f"WARNING: Multiple datasets found for gap "
            f"{gap_length}. Using:\n{candidates[0]}"
        )

    return candidates[0]


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_dataset(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:

    required_columns = {
        TARGET_COLUMN,
        SPLIT_COLUMN,
        "Time",
        "target_index",
        "gap_position",
        "gap_length",
    }

    missing = sorted(
        required_columns - set(dataframe.columns)
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

    actual_gap_lengths = (
        dataframe["gap_length"]
        .dropna()
        .unique()
        .tolist()
    )

    if actual_gap_lengths != [gap_length]:
        fail(
            f"Gap {gap_length}: invalid gap_length values: "
            f"{actual_gap_lengths}"
        )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        dataframe[SPLIT_COLUMN]
        .dropna()
        .unique()
    )

    if not expected_splits.issubset(actual_splits):
        fail(
            f"Gap {gap_length}: expected splits "
            f"{expected_splits}, found {actual_splits}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[TARGET_COLUMN]
    ):
        fail(
            f"Gap {gap_length}: ground_truth must be numeric."
        )

    if not pd.to_datetime(
        dataframe["Time"],
        errors="coerce",
    ).notna().all():
        fail(
            f"Gap {gap_length}: invalid timestamps."
        )

    if dataframe["target_index"].duplicated().any():
        fail(
            f"Gap {gap_length}: duplicate target_index detected."
        )


# =============================================================================
# FEATURE DISCOVERY
# =============================================================================

def get_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        fail(
            "No model feature columns were found."
        )

    return feature_columns


# =============================================================================
# FEATURE VALIDATION
# =============================================================================

def validate_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    gap_length: int,
) -> None:

    X = dataframe[feature_columns]

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if non_numeric:
        fail(
            f"Gap {gap_length}: non-numeric features:\n"
            f"{non_numeric}"
        )

    if X.isna().any().any():
        nan_columns = (
            X.columns[X.isna().any()]
            .tolist()
        )

        fail(
            f"Gap {gap_length}: feature NaN detected:\n"
            f"{nan_columns}"
        )

    if np.isinf(X.to_numpy()).any():
        fail(
            f"Gap {gap_length}: Inf detected in features."
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

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

    nonzero = np.abs(y_true) > 1e-12

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (
                    y_true[nonzero]
                    - y_pred[nonzero]
                )
                / y_true[nonzero]
            )
        ) * 100.0
    else:
        mape = np.nan

    error = y_pred - y_true

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
        "mean_error_bias": float(np.mean(error)),
        "max_absolute_error": float(
            np.max(np.abs(error))
        ),
    }


# =============================================================================
# PARAMETER COMBINATIONS
# =============================================================================

def parameter_combinations() -> list[dict]:

    keys = list(PARAM_GRID.keys())

    values = [
        PARAM_GRID[key]
        for key in keys
    ]

    combinations = []

    for combination in product(*values):

        params = dict(
            zip(
                keys,
                combination,
            )
        )

        combinations.append(params)

    return combinations


# =============================================================================
# VALIDATION-BASED HYPERPARAMETER SEARCH
# =============================================================================

def tune_gap(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    gap_length: int,
) -> tuple[dict, pd.DataFrame]:

    train_df = dataframe[
        dataframe[SPLIT_COLUMN] == "train"
    ].copy()

    validation_df = dataframe[
        dataframe[SPLIT_COLUMN] == "validation"
    ].copy()

    if train_df.empty:
        fail(
            f"Gap {gap_length}: TRAIN split is empty."
        )

    if validation_df.empty:
        fail(
            f"Gap {gap_length}: VALIDATION split is empty."
        )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].to_numpy()

    X_validation = validation_df[feature_columns]
    y_validation = (
        validation_df[TARGET_COLUMN]
        .to_numpy()
    )

    results = []

    combinations = parameter_combinations()

    print(
        f"\nHyperparameter combinations: "
        f"{len(combinations)}"
    )

    for index, params in enumerate(
        combinations,
        start=1,
    ):

        print(
            f"  [{index:03d}/{len(combinations):03d}] "
            f"{params}"
        )

        model = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                X_train,
                y_train,
            )

        validation_prediction = model.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            validation_prediction,
        )

        results.append(
            {
                **params,
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
            }
        )

    results_df = pd.DataFrame(results)

    # MAE is the primary optimization objective.
    results_df = results_df.sort_values(
        by=[
            "validation_MAE",
            "validation_RMSE",
        ],
        ascending=True,
    ).reset_index(drop=True)

    best_parameters = {
        key: results_df.iloc[0][key]
        for key in PARAM_GRID.keys()
    }

    # Convert values coming from the pandas DataFrame back to the exact
    # Python/scikit-learn types expected by RandomForestRegressor.
    #
    # IMPORTANT:
    # pandas represents None inside a mixed-type DataFrame as NaN.
    # Therefore max_depth=None can come back from results_df as np.nan.
    # Passing np.nan to RandomForestRegressor(max_depth=...) causes:
    #   InvalidParameterError: max_depth must be int >= 1 or None.
    #
    # We explicitly restore None and cast integer hyperparameters.
    for key, value in best_parameters.items():

        if pd.isna(value):
            best_parameters[key] = None

        elif isinstance(value, np.generic):
            best_parameters[key] = value.item()

    # Explicit type normalization after pandas extraction.
    if best_parameters["n_estimators"] is not None:
        best_parameters["n_estimators"] = int(
            best_parameters["n_estimators"]
        )

    if best_parameters["max_depth"] is not None:
        best_parameters["max_depth"] = int(
            best_parameters["max_depth"]
        )

    if best_parameters["min_samples_split"] is not None:
        best_parameters["min_samples_split"] = int(
            best_parameters["min_samples_split"]
        )

    if best_parameters["min_samples_leaf"] is not None:
        best_parameters["min_samples_leaf"] = int(
            best_parameters["min_samples_leaf"]
        )

    return (
        best_parameters,
        results_df,
    )


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    best_parameters: dict,
) -> RandomForestRegressor:

    training_df = dataframe[
        dataframe[SPLIT_COLUMN].isin(
            ["train", "validation"]
        )
    ].copy()

    X = training_df[
        feature_columns
    ]

    y = training_df[
        TARGET_COLUMN
    ].to_numpy()

    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_parameters,
    )

    model.fit(
        X,
        y,
    )

    return model


# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================

def evaluate_final_model(
    model: RandomForestRegressor,
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:

    test_df = dataframe[
        dataframe[SPLIT_COLUMN] == "test"
    ].copy()

    if test_df.empty:
        fail(
            "TEST split is empty."
        )

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET_COLUMN
    ].to_numpy()

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    prediction_df = test_df[
        [
            "event_id",
            "gap_length",
            "split",
            "target_index",
            "gap_position",
            "Time",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_df["prediction"] = predictions

    prediction_df["error"] = (
        prediction_df["prediction"]
        - prediction_df[TARGET_COLUMN]
    )

    prediction_df["absolute_error"] = (
        prediction_df["error"].abs()
    )

    prediction_df["squared_error"] = (
        prediction_df["error"] ** 2
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        prediction_df["percentage_error"] = (
            prediction_df["absolute_error"]
            / prediction_df[TARGET_COLUMN].abs()
            * 100.0
        )

    return (
        metrics,
        prediction_df,
    )


# =============================================================================
# SAVE FEATURE IMPORTANCE
# =============================================================================

def save_feature_importance(
    model: RandomForestRegressor,
    feature_columns: list[str],
    gap_length: int,
) -> Path:

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(importance_df) + 1,
        ),
    )

    output_path = (
        FEATURE_IMPORTANCE_ROOT
        / f"feature_importance_gap_{gap_length}_tuned.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# =============================================================================
# MAIN GAP TRAINING
# =============================================================================

def process_gap(
    gap_length: int,
) -> dict:

    section(
        f"V1 TUNED RANDOM FOREST — GAP {gap_length} LP"
    )

    run = mlflow.start_run(
        run_name=f"v1_tuned_gap_{gap_length}LP",
        nested=True,
    )

    safe_log_param("model_version", "v1_tuned")
    safe_log_param("algorithm", "RandomForestRegressor")
    safe_log_param("gap_length_lp", gap_length)
    safe_log_param("random_state", RANDOM_STATE)
    safe_log_param("selection_split", "validation")
    safe_log_param("selection_metric", "MAE")
    safe_log_param("final_evaluation_split", "test")
    safe_log_param("test_used_for_selection", False)

    dataset_path = find_supervised_dataset(
        gap_length
    )

    print("Input dataset:")
    print(f"    {dataset_path}")

    dataframe = pd.read_parquet(
        dataset_path
    )

    print(
        f"Rows                 : {len(dataframe):,}"
    )

    print(
        f"Columns              : {len(dataframe.columns)}"
    )

    validate_dataset(
        dataframe,
        gap_length,
    )

    print(
        "Input validation     : PASSED"
    )

    feature_columns = get_feature_columns(
        dataframe
    )

    validate_features(
        dataframe,
        feature_columns,
        gap_length,
    )

    print(
        f"Model features       : "
        f"{len(feature_columns)}"
    )

    print(
        "Feature validation   : PASSED"
    )

    safe_log_param("feature_count", len(feature_columns))
    safe_log_param("feature_columns", ",".join(feature_columns))
    safe_log_param("dataset_name", dataset_path.name)

    split_counts = (
        dataframe[SPLIT_COLUMN]
        .value_counts()
        .to_dict()
    )

    print(
        f"Train samples        : "
        f"{split_counts.get('train', 0):,}"
    )

    print(
        f"Validation samples   : "
        f"{split_counts.get('validation', 0):,}"
    )

    print(
        f"Test samples         : "
        f"{split_counts.get('test', 0):,}"
    )

    # -------------------------------------------------------------------------
    # TUNING
    # -------------------------------------------------------------------------

    section(
        f"HYPERPARAMETER TUNING — GAP {gap_length} LP"
    )

    print(
        "Selection metric     : Validation MAE"
    )

    print(
        "Test set usage       : DISABLED"
    )

    best_parameters, tuning_results = tune_gap(
        dataframe=dataframe,
        feature_columns=feature_columns,
        gap_length=gap_length,
    )

    print()
    print(
        "BEST PARAMETERS"
    )

    for key, value in best_parameters.items():
        print(
            f"    {key}: {value}"
        )
        safe_log_param(f"best_{key}", value)

    safe_log_metric(
        "best_validation_MAE",
        tuning_results.iloc[0]["validation_MAE"],
    )
    safe_log_metric(
        "best_validation_RMSE",
        tuning_results.iloc[0]["validation_RMSE"],
    )
    safe_log_metric(
        "best_validation_R2",
        tuning_results.iloc[0]["validation_R2"],
    )

    tuning_path = (
        SUMMARY_ROOT
        / f"tuning_results_gap_{gap_length}.csv"
    )

    tuning_results.to_csv(
        tuning_path,
        index=False,
    )

    mlflow.log_artifact(
    str(tuning_path),
    artifact_path="tuning",
    )

    print()
    print(
        f"Tuning results saved:"
    )
    print(
        f"    {tuning_path}"
    )

    # -------------------------------------------------------------------------
    # FINAL TRAINING
    # -------------------------------------------------------------------------

    section(
        f"FINAL TRAINING — GAP {gap_length} LP"
    )

    print(
        "Training data        : TRAIN + VALIDATION"
    )

    print(
        "TEST data             : HELD OUT"
    )

    model = train_final_model(
        dataframe=dataframe,
        feature_columns=feature_columns,
        best_parameters=best_parameters,
    )

    print(
        "Final model training : PASSED"
    )

    # -------------------------------------------------------------------------
    # TEST EVALUATION
    # -------------------------------------------------------------------------

    section(
        f"FINAL TEST EVALUATION — GAP {gap_length} LP"
    )

    metrics, prediction_df = evaluate_final_model(
        model=model,
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    print(
        f"Samples              : "
        f"{len(prediction_df):,}"
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

    for metric_name, metric_value in metrics.items():
        safe_log_metric(metric_name, metric_value)

    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------

    model_path = (
        MODEL_ROOT
        / f"random_forest_v1_tuned_gap_{gap_length}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print()
    print(
        "Saved model:"
    )
    print(
        f"    {model_path}"
    )

    # -------------------------------------------------------------------------
    # SAVE PREDICTIONS
    # -------------------------------------------------------------------------

    prediction_path = (
        PREDICTION_ROOT
        / f"predictions_gap_{gap_length}_v1_tuned.parquet"
    )

    prediction_df.to_parquet(
        prediction_path,
        index=False,
    )

    print(
        "Saved predictions:"
    )
    print(
        f"    {prediction_path}"
    )

    # -------------------------------------------------------------------------
    # SAVE METRICS
    # -------------------------------------------------------------------------

    metrics_record = {
        "model_version": "v1_tuned",
        "gap_length": gap_length,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "selection_metric": "validation_MAE",
        "training_splits": [
            "train",
            "validation",
        ],
        "final_evaluation_split": "test",
        "random_state": RANDOM_STATE,
        "best_parameters": best_parameters,
        **metrics,
    }

    metrics_path = (
        METRICS_ROOT
        / f"metrics_gap_{gap_length}_v1_tuned.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_record,
            file,
            indent=4,
        )

    print(
        "Saved metrics:"
    )
    print(
        f"    {metrics_path}"
    )

    # -------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------------------------

    importance_path = save_feature_importance(
        model=model,
        feature_columns=feature_columns,
        gap_length=gap_length,
    )

    print(
        "Saved feature importance:"
    )
    print(
        f"    {importance_path}"
    )

    # -------------------------------------------------------------------------
    # MODEL METADATA
    # -------------------------------------------------------------------------

    metadata = {
        "model_version": "v1_tuned",
        "algorithm": "RandomForestRegressor",
        "gap_length": gap_length,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "best_parameters": best_parameters,
        "random_state": RANDOM_STATE,
        "training_samples": int(
            (
                dataframe[SPLIT_COLUMN]
                .isin(["train", "validation"])
            ).sum()
        ),
        "test_samples": int(
            (
                dataframe[SPLIT_COLUMN]
                == "test"
            ).sum()
        ),
        "hyperparameter_selection": {
            "method": "controlled_grid_search",
            "selection_split": "validation",
            "selection_metric": "MAE",
            "test_used_for_selection": False,
        },
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata_path = (
        METADATA_ROOT
        / f"model_metadata_gap_{gap_length}_v1_tuned.json"
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

    print(
        "Saved metadata:"
    )

    print(
        f"    {metadata_path}"
    )

    # -------------------------------------------------------------------------
    # MLFLOW ARTIFACTS
    # -------------------------------------------------------------------------

    mlflow.log_artifact(
        str(model_path),
        artifact_path="local_model",
    )
    mlflow.log_artifact(
        str(prediction_path),
        artifact_path="predictions",
    )
    mlflow.log_artifact(
        str(metrics_path),
        artifact_path="metrics",
    )
    mlflow.log_artifact(
        str(importance_path),
        artifact_path="feature_importance",
    )
    mlflow.log_artifact(
        str(metadata_path),
        artifact_path="metadata",
    )

    # Also log the fitted sklearn model in MLflow's native format.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
    )

    mlflow.set_tag(
        "model_family",
        "random_forest",
    )
    mlflow.set_tag(
        "project",
        "AMI Smart Meters Data Imputation",
    )
    mlflow.set_tag(
        "gap_strategy",
        "gap_specific_model",
    )
    mlflow.set_tag(
        "jmr_used",
        "false",
    )
    mlflow.set_tag(
        "test_leakage",
        "prohibited",
    )

    # -------------------------------------------------------------------------
    # MODEL RELOAD VERIFICATION
    # -------------------------------------------------------------------------

    reloaded_model = joblib.load(
        model_path
    )

    reload_predictions = reloaded_model.predict(
        dataframe[
            dataframe[SPLIT_COLUMN] == "test"
        ][feature_columns]
    )

    if not np.allclose(
        reload_predictions,
        prediction_df["prediction"].to_numpy(),
    ):
        fail(
            f"Gap {gap_length}: "
            "reloaded model predictions differ."
        )

    print(
        "Model reload         : PASSED"
    )

    mlflow.set_tag(
        "model_reload_verified",
        "true",
    )
    mlflow.end_run(
        status="FINISHED",
    )

    return {
        "gap_length": gap_length,
        "samples": len(prediction_df),
        **metrics,
        "feature_count": len(feature_columns),
        **{
            f"best_{key}": value
            for key, value in best_parameters.items()
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION — V1 TUNED MODEL"
    )
    print("=" * 80)

    print()
    print(
        "V1 baseline models       : FROZEN"
    )

    print(
        "V1 baseline predictions  : FROZEN"
    )

    print(
        "V1 supervised datasets   : USED"
    )

    print(
        "Hyperparameter tuning    : ENABLED"
    )

    print(
        "Selection split          : VALIDATION"
    )

    print(
        "Final evaluation split   : TEST"
    )

    print(
        "Test leakage             : PROHIBITED"
    )

    print(
        "JMR during training     : NOT USED"
    )

    print(
        "Inference CSV/XLSX       : NOT USED"
    )

    print(
        "JMR reconciliation       : NOT USED"
    )

    print(
        "MLflow                   : ENABLED"
    )

    print(
        "Source modification      : DISABLED"
    )

    print()

    print(
        "TUNING SEARCH SPACE"
    )

    for parameter, values in PARAM_GRID.items():

        print(
            f"    {parameter}: {values}"
        )

    create_output_directories()
    configure_mlflow()

    print()
    print(
        "MLflow tracking URI    : "
        f"{mlflow.get_tracking_uri()}"
    )
    print(
        "MLflow experiment      : "
        f"{MLFLOW_EXPERIMENT_NAME}"
    )

    parent_run = mlflow.start_run(
        run_name="v1_tuned_all_gaps",
    )
    safe_log_param("model_version", "v1_tuned")
    safe_log_param("algorithm", "RandomForestRegressor")
    safe_log_param("gap_lengths", ",".join(map(str, GAP_LENGTHS)))
    safe_log_param("random_state", RANDOM_STATE)
    safe_log_param("selection_metric", "validation_MAE")
    safe_log_param("final_evaluation_split", "test")
    safe_log_param("number_of_models", len(GAP_LENGTHS))
    mlflow.set_tag("run_type", "parent")
    mlflow.set_tag("gap_strategy", "four_gap_specific_models")
    mlflow.set_tag("project", "AMI Smart Meters Data Imputation")
    mlflow.set_tag("jmr_used", "false")

    results = []

    for gap_length in GAP_LENGTHS:

        result = process_gap(
            gap_length
        )

        results.append(
            result
        )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    summary_path = (
        SUMMARY_ROOT
        / "v1_tuned_test_performance_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # Log the consolidated summary to the parent MLflow run.
    mlflow.log_artifact(
        str(summary_path),
        artifact_path="summary",
    )

    for _, row in summary.iterrows():
        gap = int(row["gap_length"])
        for metric in ["MAE", "RMSE", "R2", "MAPE_percent"]:
            safe_log_metric(
                f"gap_{gap}_{metric}",
                row[metric],
            )

    best_gap = summary.loc[summary["MAE"].idxmin(), "gap_length"]
    safe_log_param("best_gap_by_MAE", int(best_gap))
    mlflow.end_run(status="FINISHED")

    print()
    print("=" * 80)
    print(
        "V1 TUNED MODEL TEST PERFORMANCE SUMMARY"
    )
    print("=" * 80)

    display_columns = [
        "gap_length",
        "samples",
        "MAE",
        "RMSE",
        "R2",
        "MAPE_percent",
        "mean_error_bias",
        "max_absolute_error",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
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
        "V1 TUNED MODEL STAGE COMPLETE"
    )
    print("=" * 80)

    print()
    print(
        "V1 baseline models       : UNCHANGED"
    )

    print(
        "V1 baseline predictions  : UNCHANGED"
    )

    print(
        "V1 tuned models          : CREATED"
    )

    print(
        "V1 tuned predictions     : CREATED"
    )

    print(
        "JMR                     : NOT USED"
    )

    print(
        "MLflow                  : NOT YET"
    )

    print()
    print(
        "Model directory:"
    )

    print(
        f"    {MODEL_ROOT}"
    )

    print()
    print(
        "Prediction directory:"
    )

    print(
        f"    {PREDICTION_ROOT}"
    )

    print()
    print(
        "Metadata directory:"
    )

    print(
        f"    {METADATA_ROOT}"
    )

    print()
    print(
        "MLflow tracking directory:"
    )
    print(
        f"    {MLFLOW_ROOT}"
    )


if __name__ == "__main__":
    main()