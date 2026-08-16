# =============================================================================
# src/model.py
# =============================================================================
"""
Random Forest model training and evaluation for CDP load-profile imputation.

Pipeline position
-----------------
supervised datasets
        ↓
Random Forest models
        ↓
predictions
        ↓
evaluation
        ↓
model comparison

Design
------
1. Independent model for each gap length:
       1 LP
       6 LP
       24 LP
       48 LP

2. 96 LP experiment is removed.

3. Fixed-event formulation.

4. One supervised sample = one missing LP.

5. Therefore:
       one RF prediction = one missing LP

6. No target-derived lag/lead features are created in this stage.

7. No normalization is performed.

8. Source data is never modified.
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
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    CONFIG["project_root"]
)

DATA_ROOT = (
    PROJECT_ROOT
    / CONFIG["outputs"]["processed_data_dir"]
)

SUPERVISED_ROOT = (
    PROJECT_ROOT
    / CONFIG["outputs"]["supervised_data_dir"]
)

MODEL_ROOT = (
    PROJECT_ROOT
    / CONFIG["outputs"]["model_dir"]
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / CONFIG["outputs"]["output_dir"]
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

GAP_LENGTHS = [
    int(value)
    for value in CONFIG["gaps"]["lengths"]
]

# Explicitly enforce the current approved experiment design.
EXPECTED_GAP_LENGTHS = [
    1,
    6,
    24,
    48,
]

if GAP_LENGTHS != EXPECTED_GAP_LENGTHS:
    raise RuntimeError(
        "Configured gap lengths do not match the "
        "current project design.\n"
        f"Expected: {EXPECTED_GAP_LENGTHS}\n"
        f"Configured: {GAP_LENGTHS}"
    )


TARGET_COLUMN = "ground_truth"

SPLITS = [
    "train",
    "validation",
    "test",
]


# =============================================================================
# LOGGING
# =============================================================================

def section(title: str) -> None:
    """Print a major section."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    """Print a subsection."""

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def info(message: str) -> None:
    """Print an informational message."""

    print(message)


def fail(message: str) -> None:
    """Raise a descriptive model-stage error."""

    raise RuntimeError(message)


# =============================================================================
# PATH HELPERS
# =============================================================================

def supervised_file(
    gap_length: int,
) -> Path:
    """
    Return supervised parquet path for a gap experiment.
    """

    meter_id = CONFIG["data"]["selected_meter"]
    direction = CONFIG["data"]["target_direction"]

    filename = (
        f"{meter_id}_{direction}"
        f"_supervised_gap_{gap_length}.parquet"
    )

    return (
        SUPERVISED_ROOT
        / f"gap_{gap_length}"
        / filename
    )


def model_file(
    gap_length: int,
) -> Path:

    return (
        MODEL_ROOT
        / f"gap_{gap_length}"
        / f"random_forest_gap_{gap_length}.joblib"
    )


def prediction_file(
    gap_length: int,
) -> Path:

    return (
        OUTPUT_ROOT
        / "model"
        / "predictions"
        / f"predictions_gap_{gap_length}.parquet"
    )


def metrics_file(
    gap_length: int,
) -> Path:

    return (
        OUTPUT_ROOT
        / "model"
        / "metrics"
        / f"metrics_gap_{gap_length}.json"
    )


def feature_importance_file(
    gap_length: int,
) -> Path:

    return (
        OUTPUT_ROOT
        / "model"
        / "feature_importance"
        / f"feature_importance_gap_{gap_length}.csv"
    )


def model_metadata_file(
    gap_length: int,
) -> Path:

    return (
        OUTPUT_ROOT
        / "model"
        / "metadata"
        / f"model_metadata_gap_{gap_length}.json"
    )


# =============================================================================
# DIRECTORY CREATION
# =============================================================================

def create_output_directories() -> None:
    """Create required model-stage directories."""

    directories = [
        MODEL_ROOT,
        OUTPUT_ROOT / "model" / "predictions",
        OUTPUT_ROOT / "model" / "metrics",
        OUTPUT_ROOT / "model" / "feature_importance",
        OUTPUT_ROOT / "model" / "metadata",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# INPUT LOADING
# =============================================================================

def load_supervised_dataframe(
    gap_length: int,
) -> pd.DataFrame:
    """
    Load one supervised dataset.
    """

    input_file = supervised_file(
        gap_length
    )

    if not input_file.exists():

        fail(
            "Supervised dataset not found:\n"
            f"    {input_file}\n\n"
            "Run the supervised dataset stage first."
        )

    dataframe = pd.read_parquet(
        input_file
    )

    return dataframe


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_supervised_input(
    dataframe: pd.DataFrame,
    gap_length: int,
) -> None:
    """
    Validate supervised dataset before model training.
    """

    subsection(
        f"GAP {gap_length} INPUT VALIDATION"
    )

    required_columns = [
        "event_id",
        "gap_length",
        "split",
        "target_index",
        "gap_position",
        "Time",
        "ground_truth",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        fail(
            f"Gap {gap_length}: missing required columns:\n"
            f"{missing_columns}"
        )

    info(
        "Required columns       : PASSED"
    )

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------

    if dataframe[TARGET_COLUMN].isna().any():

        fail(
            f"Gap {gap_length}: ground_truth contains NaN."
        )

    info(
        "Ground truth           : PASSED"
    )

    # -------------------------------------------------------------------------
    # Gap length
    # -------------------------------------------------------------------------

    unique_gap_lengths = sorted(
        dataframe["gap_length"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if unique_gap_lengths != [gap_length]:

        fail(
            f"Gap {gap_length}: incorrect gap_length values.\n"
            f"Detected: {unique_gap_lengths}"
        )

    info(
        "Gap length             : PASSED"
    )

    # -------------------------------------------------------------------------
    # Split labels
    # -------------------------------------------------------------------------

    actual_splits = set(
        dataframe["split"]
        .astype(str)
        .unique()
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    if not actual_splits.issubset(
        expected_splits
    ):

        fail(
            f"Gap {gap_length}: invalid split labels.\n"
            f"Detected: {actual_splits}"
        )

    for split in expected_splits:

        if split not in actual_splits:

            fail(
                f"Gap {gap_length}: missing split '{split}'."
            )

    info(
        "Split labels           : PASSED"
    )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe["Time"]
    ):

        dataframe["Time"] = pd.to_datetime(
            dataframe["Time"]
        )

    if dataframe["Time"].isna().any():

        fail(
            f"Gap {gap_length}: Time contains NaN."
        )

    info(
        "Timestamp              : PASSED"
    )

    # -------------------------------------------------------------------------
    # Target index
    # -------------------------------------------------------------------------

    if dataframe["target_index"].isna().any():

        fail(
            f"Gap {gap_length}: target_index contains NaN."
        )

    if not pd.api.types.is_integer_dtype(
        dataframe["target_index"]
    ):

        # Accept integer-valued numeric columns.
        values = pd.to_numeric(
            dataframe["target_index"],
            errors="coerce",
        )

        if values.isna().any():

            fail(
                f"Gap {gap_length}: target_index "
                "contains non-numeric values."
            )

    info(
        "Target index            : PASSED"
    )

    # -------------------------------------------------------------------------
    # Duplicate samples
    # -------------------------------------------------------------------------

    duplicate_mask = dataframe.duplicated(
        subset=[
            "event_id",
            "target_index",
        ],
        keep=False,
    )

    if duplicate_mask.any():

        duplicates = dataframe.loc[
            duplicate_mask,
            [
                "event_id",
                "target_index",
            ],
        ]

        fail(
            f"Gap {gap_length}: duplicate supervised samples.\n"
            f"{duplicates.head(20)}"
        )

    info(
        "Duplicate samples       : PASSED"
    )

    # -------------------------------------------------------------------------
    # One sample per missing LP
    # -------------------------------------------------------------------------

    event_counts = (
        dataframe
        .groupby("event_id")
        .size()
    )

    if not (
        event_counts
        == gap_length
    ).all():

        fail(
            f"Gap {gap_length}: events do not contain "
            f"exactly {gap_length} samples."
        )

    info(
        "Samples per event      : PASSED"
    )

    # -------------------------------------------------------------------------
    # Gap positions
    # -------------------------------------------------------------------------

    expected_positions = set(
        range(
            1,
            gap_length + 1,
        )
    )

    for event_id, group in dataframe.groupby(
        "event_id"
    ):

        positions = set(
            group["gap_position"]
            .astype(int)
            .tolist()
        )

        if positions != expected_positions:

            fail(
                f"Gap {gap_length}, event {event_id}: "
                "incorrect gap positions."
            )

    info(
        "Gap positions          : PASSED"
    )

    # -------------------------------------------------------------------------
    # Numeric feature validation
    # -------------------------------------------------------------------------

    excluded = {
        TARGET_COLUMN,
        "event_id",
        "gap_length",
        "split",
        "target_index",
        "gap_position",
        "gap_position_fraction",
        "Time",
    }

    candidate_features = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    numeric_features = [
        column
        for column in candidate_features
        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if not numeric_features:

        fail(
            f"Gap {gap_length}: no numeric model features found."
        )

    numeric_values = dataframe[
        numeric_features
    ]

    if numeric_values.isna().any().any():

        bad_columns = (
            numeric_values
            .columns[
                numeric_values.isna().any()
            ]
            .tolist()
        )

        fail(
            f"Gap {gap_length}: model features "
            f"contain NaN.\n"
            f"Columns: {bad_columns}"
        )

    if np.isinf(
        numeric_values.to_numpy(
            dtype=float
        )
    ).any():

        fail(
            f"Gap {gap_length}: model features "
            "contain Inf."
        )

    info(
        "Feature NaN / Inf       : PASSED"
    )


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def get_model_features(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Determine model features.

    Metadata columns, timestamp, and ground truth are excluded.

    The supervised dataset currently contains 28 model features.
    """

    excluded_columns = {
        "event_id",
        "gap_length",
        "split",
        "target_index",
        "gap_position",
        "gap_position_fraction",
        "Time",
        "ground_truth",
    }

    features = []

    for column in dataframe.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):

            features.append(
                column
            )

    return features


def validate_model_features(
    dataframe: pd.DataFrame,
    features: List[str],
    gap_length: int,
) -> None:
    """
    Validate the selected model features.
    """

    expected_feature_count = 28

    if len(features) != expected_feature_count:

        fail(
            f"Gap {gap_length}: unexpected number "
            "of model features.\n"
            f"Expected: {expected_feature_count}\n"
            f"Actual: {len(features)}\n"
            f"Features: {features}"
        )

    info(
        f"Model features         : {len(features)}"
    )

    info(
        f"Target column          : {TARGET_COLUMN}"
    )


# =============================================================================
# SPLIT
# =============================================================================

def split_xy(
    dataframe: pd.DataFrame,
    features: List[str],
) -> Tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Create X and y.
    """

    X = dataframe[
        features
    ].copy()

    y = dataframe[
        TARGET_COLUMN
    ].copy()

    return X, y


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model() -> RandomForestRegressor:
    """
    Create Random Forest from YAML configuration.
    """

    model_config = CONFIG["model"]

    return RandomForestRegressor(
        n_estimators=int(
            model_config["n_estimators"]
        ),

        max_depth=(
            None
            if model_config["max_depth"] is None
            else int(
                model_config["max_depth"]
            )
        ),

        min_samples_split=int(
            model_config["min_samples_split"]
        ),

        min_samples_leaf=int(
            model_config["min_samples_leaf"]
        ),

        max_features=model_config[
            "max_features"
        ],

        random_state=int(
            model_config["random_state"]
        ),

        n_jobs=int(
            model_config["n_jobs"]
        ),
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    dataframe: pd.DataFrame,
    features: List[str],
    gap_length: int,
) -> RandomForestRegressor:
    """
    Train one independent Random Forest.
    """

    subsection(
        f"TRAINING RANDOM FOREST — {gap_length} LP"
    )

    train_df = dataframe[
        dataframe["split"]
        == "train"
    ].copy()

    validation_df = dataframe[
        dataframe["split"]
        == "validation"
    ].copy()

    test_df = dataframe[
        dataframe["split"]
        == "test"
    ].copy()

    if train_df.empty:

        fail(
            f"Gap {gap_length}: training dataset is empty."
        )

    # -------------------------------------------------------------------------
    # Important:
    #
    # Validation and test data are NOT used during model fitting.
    # -------------------------------------------------------------------------

    X_train, y_train = split_xy(
        train_df,
        features,
    )

    # Prevent accidental target leakage.
    if TARGET_COLUMN in X_train.columns:

        fail(
            f"Gap {gap_length}: target leakage detected."
        )

    model = create_model()

    model_config = CONFIG["model"]

    info(
        "Random Forest parameters:"
    )

    info(
        f"  n_estimators      : "
        f"{model_config['n_estimators']}"
    )

    info(
        f"  max_depth         : "
        f"{model_config['max_depth']}"
    )

    info(
        f"  min_samples_split : "
        f"{model_config['min_samples_split']}"
    )

    info(
        f"  min_samples_leaf  : "
        f"{model_config['min_samples_leaf']}"
    )

    info(
        f"  max_features      : "
        f"{model_config['max_features']}"
    )

    info(
        f"  random_state      : "
        f"{model_config['random_state']}"
    )

    info(
        f"  n_jobs            : "
        f"{model_config['n_jobs']}"
    )

    model.fit(
        X_train,
        y_train,
    )

    info(
        "Random Forest training : PASSED"
    )

    return model


# =============================================================================
# PREDICTIONS
# =============================================================================

def generate_predictions(
    model: RandomForestRegressor,
    dataframe: pd.DataFrame,
    features: List[str],
    gap_length: int,
) -> pd.DataFrame:
    """
    Generate predictions for train, validation and test.

    One row in the supervised dataset represents one missing LP.
    Therefore every row receives exactly one RF prediction.
    """

    subsection(
        f"GENERATING PREDICTIONS — {gap_length} LP"
    )

    prediction_parts = []

    for split in SPLITS:

        split_df = dataframe[
            dataframe["split"]
            == split
        ].copy()

        if split_df.empty:

            fail(
                f"Gap {gap_length}: split "
                f"'{split}' is empty."
            )

        X_split = split_df[
            features
        ]

        predictions = model.predict(
            X_split
        )

        split_df[
            "prediction"
        ] = predictions

        split_df[
            "absolute_error"
        ] = (
            split_df[
                TARGET_COLUMN
            ]
            - split_df[
                "prediction"
            ]
        ).abs()

        prediction_parts.append(
            split_df
        )

        info(
            f"{split.capitalize():21s}: "
            f"{len(split_df):,} predictions"
        )

    predictions_df = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    return predictions_df


# =============================================================================
# EVALUATION
# =============================================================================

def calculate_metrics(
    predictions_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate MAE, RMSE and R2 for every split.
    """

    results = {}

    for split in SPLITS:

        subset = predictions_df[
            predictions_df["split"]
            == split
        ]

        y_true = subset[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        y_pred = subset[
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

        results[split] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "samples": int(
                len(subset)
            ),
        }

    return results


def print_metrics(
    metrics: Dict[str, Dict[str, float]],
    gap_length: int,
) -> None:
    """
    Print model metrics.
    """

    subsection(
        f"EVALUATION — {gap_length} LP"
    )

    for split in SPLITS:

        values = metrics[
            split
        ]

        info(
            f"{split.capitalize():12s} | "
            f"MAE: {values['MAE']:,.4f} | "
            f"RMSE: {values['RMSE']:,.4f} | "
            f"R2: {values['R2']:.6f}"
        )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def create_feature_importance(
    model: RandomForestRegressor,
    features: List[str],
) -> pd.DataFrame:
    """
    Create feature importance dataframe.
    """

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return importance


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(
    model: RandomForestRegressor,
    gap_length: int,
) -> None:

    output_file = model_file(
        gap_length
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_file,
    )

    info(
        "Saved model:"
    )

    info(
        f"    {output_file}"
    )


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    predictions_df: pd.DataFrame,
    gap_length: int,
) -> None:

    output_file = prediction_file(
        gap_length
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_parquet(
        output_file,
        index=False,
    )

    info(
        "Saved predictions:"
    )

    info(
        f"    {output_file}"
    )


# =============================================================================
# SAVE METRICS
# =============================================================================

def save_metrics(
    metrics: Dict[str, Dict[str, float]],
    gap_length: int,
) -> None:

    output_file = metrics_file(
        gap_length
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "gap_length": gap_length,
        "metrics": metrics,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=4,
        )

    info(
        "Saved metrics:"
    )

    info(
        f"    {output_file}"
    )


# =============================================================================
# SAVE FEATURE IMPORTANCE
# =============================================================================

def save_feature_importance(
    importance_df: pd.DataFrame,
    gap_length: int,
) -> None:

    output_file = feature_importance_file(
        gap_length
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        output_file,
        index=False,
    )

    info(
        "Saved feature importance:"
    )

    info(
        f"    {output_file}"
    )


# =============================================================================
# SAVE MODEL METADATA
# =============================================================================

def save_model_metadata(
    gap_length: int,
    features: List[str],
    dataframe: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
) -> None:

    output_file = model_metadata_file(
        gap_length
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_config = CONFIG["model"]

    metadata = {
        "project": CONFIG["project"],
        "meter_id": CONFIG["data"][
            "selected_meter"
        ],
        "target_direction": CONFIG["data"][
            "target_direction"
        ],
        "gap_length": gap_length,
        "formulation": CONFIG[
            "supervised"
        ]["formulation"],
        "one_prediction_per_missing_lp": True,
        "sliding_windows": False,
        "target_lags": False,
        "target_leads": False,
        "normalization": False,
        "algorithm": model_config[
            "algorithm"
        ],
        "model_parameters": model_config,
        "feature_count": len(features),
        "features": features,
        "sample_counts": {
            split: int(
                (
                    dataframe["split"]
                    == split
                ).sum()
            )
            for split in SPLITS
        },
        "metrics": metrics,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    info(
        "Saved model metadata:"
    )

    info(
        f"    {output_file}"
    )


# =============================================================================
# MODEL VERIFICATION
# =============================================================================

def verify_saved_model(
    model: RandomForestRegressor,
    predictions_df: pd.DataFrame,
    gap_length: int,
) -> None:

    subsection(
        f"MODEL VERIFICATION — {gap_length} LP"
    )

    # -------------------------------------------------------------------------
    # Reload model
    # -------------------------------------------------------------------------

    saved_model = joblib.load(
        model_file(
            gap_length
        )
    )

    if not isinstance(
        saved_model,
        RandomForestRegressor,
    ):

        fail(
            f"Gap {gap_length}: reloaded model "
            "is not RandomForestRegressor."
        )

    info(
        "Model reload           : PASSED"
    )

    # -------------------------------------------------------------------------
    # Reload predictions
    # -------------------------------------------------------------------------

    reloaded_predictions = pd.read_parquet(
        prediction_file(
            gap_length
        )
    )

    if len(
        reloaded_predictions
    ) != len(
        predictions_df
    ):

        fail(
            f"Gap {gap_length}: prediction "
            "reload row count mismatch."
        )

    if "prediction" not in (
        reloaded_predictions.columns
    ):

        fail(
            f"Gap {gap_length}: prediction "
            "column missing after reload."
        )

    info(
        "Prediction reload      : PASSED"
    )


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

def create_comparison_summary(
    all_metrics: Dict[
        int,
        Dict[str, Dict[str, float]]
    ],
) -> pd.DataFrame:
    """
    Create one comparison table across gap lengths.
    """

    rows = []

    for gap_length in GAP_LENGTHS:

        metrics = all_metrics[
            gap_length
        ]

        for split in SPLITS:

            values = metrics[
                split
            ]

            rows.append(
                {
                    "gap_length": gap_length,
                    "split": split,
                    "samples": values[
                        "samples"
                    ],
                    "MAE": values[
                        "MAE"
                    ],
                    "RMSE": values[
                        "RMSE"
                    ],
                    "R2": values[
                        "R2"
                    ],
                }
            )

    return pd.DataFrame(
        rows
    )


def save_comparison_summary(
    comparison: pd.DataFrame,
) -> None:

    output_file = (
        OUTPUT_ROOT
        / "model"
        / "metrics"
        / "model_comparison_summary.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    info(
        "Saved comparison summary:"
    )

    info(
        f"    {output_file}"
    )


def print_comparison_summary(
    comparison: pd.DataFrame,
) -> None:

    subsection(
        "MODEL COMPARISON SUMMARY"
    )

    print()

    print(
        comparison.to_string(
            index=False
        )
    )


# =============================================================================
# SINGLE GAP EXPERIMENT
# =============================================================================

def run_gap_experiment(
    gap_length: int,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate one independent gap model.
    """

    subsection(
        f"GAP LENGTH: {gap_length} LP"
    )

    input_file = supervised_file(
        gap_length
    )

    info(
        "Input dataframe:"
    )

    info(
        f"    {input_file}"
    )

    dataframe = load_supervised_dataframe(
        gap_length
    )

    info(
        f"Rows                 : "
        f"{len(dataframe):,}"
    )

    info(
        f"Columns              : "
        f"{len(dataframe.columns):,}"
    )

    validate_supervised_input(
        dataframe=dataframe,
        gap_length=gap_length,
    )

    features = get_model_features(
        dataframe
    )

    validate_model_features(
        dataframe=dataframe,
        features=features,
        gap_length=gap_length,
    )

    # -------------------------------------------------------------------------
    # Training sample counts
    # -------------------------------------------------------------------------

    for split in SPLITS:

        count = int(
            (
                dataframe["split"]
                == split
            ).sum()
        )

        info(
            f"{split.capitalize():21s}: "
            f"{count:,}"
        )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model = train_model(
        dataframe=dataframe,
        features=features,
        gap_length=gap_length,
    )

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    predictions_df = generate_predictions(
        model=model,
        dataframe=dataframe,
        features=features,
        gap_length=gap_length,
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    metrics = calculate_metrics(
        predictions_df
    )

    print_metrics(
        metrics=metrics,
        gap_length=gap_length,
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    importance_df = create_feature_importance(
        model=model,
        features=features,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    subsection(
        f"SAVING MODEL — {gap_length} LP"
    )

    save_model(
        model=model,
        gap_length=gap_length,
    )

    save_predictions(
        predictions_df=predictions_df,
        gap_length=gap_length,
    )

    save_metrics(
        metrics=metrics,
        gap_length=gap_length,
    )

    save_feature_importance(
        importance_df=importance_df,
        gap_length=gap_length,
    )

    save_model_metadata(
        gap_length=gap_length,
        features=features,
        dataframe=dataframe,
        metrics=metrics,
    )

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    verify_saved_model(
        model=model,
        predictions_df=predictions_df,
        gap_length=gap_length,
    )

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Execute the complete model stage.
    """

    section(
        "RANDOM FOREST CDP IMPUTATION — MODEL STAGE"
    )

    info(
        "Stage: supervised datasets → Random Forest models"
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
        "Sliding windows: DISABLED"
    )

    info(
        "One RF prediction per missing LP: ENABLED"
    )

    info(
        "Target-derived lag/lead features: DISABLED"
    )

    info(
        "Four independent RF models: ENABLED"
    )

    info(
        "Model training: ENABLED"
    )

    info(
        "Evaluation: ENABLED"
    )

    info(
        "Normalization: DISABLED"
    )

    info(
        "Source modification: DISABLED"
    )

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------

    section(
        "MODEL CONFIGURATION"
    )

    model_config = CONFIG["model"]

    info(
        f"Algorithm             : "
        f"{model_config['algorithm']}"
    )

    info(
        f"n_estimators          : "
        f"{model_config['n_estimators']}"
    )

    info(
        f"max_depth             : "
        f"{model_config['max_depth']}"
    )

    info(
        f"min_samples_split     : "
        f"{model_config['min_samples_split']}"
    )

    info(
        f"min_samples_leaf      : "
        f"{model_config['min_samples_leaf']}"
    )

    info(
        f"max_features          : "
        f"{model_config['max_features']}"
    )

    info(
        f"random_state          : "
        f"{model_config['random_state']}"
    )

    info(
        f"n_jobs                : "
        f"{model_config['n_jobs']}"
    )

    create_output_directories()

    all_metrics = {}

    # -------------------------------------------------------------------------
    # Four independent models
    # -------------------------------------------------------------------------

    for gap_length in GAP_LENGTHS:

        metrics = run_gap_experiment(
            gap_length
        )

        all_metrics[
            gap_length
        ] = metrics

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    comparison = create_comparison_summary(
        all_metrics
    )

    print_comparison_summary(
        comparison
    )

    save_comparison_summary(
        comparison
    )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    section(
        "MODEL TRAINING STAGE COMPLETE"
    )

    info(
        "Independent Random Forest models:"
    )

    for gap_length in GAP_LENGTHS:

        info(
            f"{gap_length:6d} LP : "
            f"{model_file(gap_length)}"
        )

    print()

    info(
        "Prediction outputs:"
    )

    for gap_length in GAP_LENGTHS:

        info(
            f"{gap_length:6d} LP : "
            f"{prediction_file(gap_length)}"
        )

    print()

    info(
        "Metrics summary:"
    )

    info(
        f"    {OUTPUT_ROOT / 'model' / 'metrics' / 'model_comparison_summary.csv'}"
    )

    print()

    info(
        "96 LP model          : NOT TRAINED"
    )

    info(
        "Sliding windows      : NOT USED"
    )

    info(
        "One prediction/LP    : ENABLED"
    )

    info(
        "Source CSV           : UNCHANGED"
    )

    print()

    info(
        "MODEL STAGE COMPLETE"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()