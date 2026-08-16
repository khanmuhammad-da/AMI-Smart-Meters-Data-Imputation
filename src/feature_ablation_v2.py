"""
RANDOM FOREST CDP IMPUTATION — V2 FEATURE IMPORTANCE + ABLATION

Purpose
-------
Analyze the contribution of the V2 feature groups without modifying V1
or the existing V2 model/prediction outputs.

This stage performs:

1. Random Forest feature importance for all 43 V2 predictors.
2. Feature-group ablation experiments.
3. Comparison against the V1 baseline.
4. Test-set evaluation using the same fixed-event test locations.

IMPORTANT
---------
- V1 source files are NOT modified.
- V1 predictions are NOT modified.
- V2 supervised datasets are NOT modified.
- Existing V2 models are NOT modified.
- Existing V2 predictions are NOT modified.
- No MLflow is used yet.
- Temporary models are kept in memory only.
- Test event locations remain identical across experiments.

Experiments
-----------
V1_baseline
V2_all
V2_no_historical
V2_no_context_stats
V2_no_recent_context
V2_no_trend
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "processed"
V1_SUPERVISED_ROOT = DATA_ROOT / "supervised"
V2_SUPERVISED_ROOT = DATA_ROOT / "supervised_v2"

V1_PREDICTIONS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "model"
    / "predictions"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "ablation_v2"
)

IMPORTANCE_ROOT = OUTPUT_ROOT / "feature_importance"
EXPERIMENT_ROOT = OUTPUT_ROOT / "experiments"
SUMMARY_ROOT = OUTPUT_ROOT / "summary"


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

GAP_LENGTHS = [1, 6, 24, 48]

RANDOM_STATE = 42

RF_PARAMETERS = {
    "random_state": 42,
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
}


# =============================================================================
# DATASET COLUMNS
# =============================================================================

IDENTIFIER_COLUMNS = {
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
# V2 FEATURE GROUPS
# =============================================================================

HISTORICAL_FEATURES = [
    "target_previous_day_same_slot",
    "target_previous_week_same_slot",
    "previous_week_available",
]


CONTEXT_STAT_FEATURES = [
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
]


RECENT_CONTEXT_FEATURES = [
    "left_recent_mean",
    "right_recent_mean",
    "left_last_value",
    "right_first_value",
]


TREND_FEATURES = [
    "left_slope",
    "right_slope",
]


V2_NEW_FEATURES = (
    HISTORICAL_FEATURES
    + CONTEXT_STAT_FEATURES
    + RECENT_CONTEXT_FEATURES
    + TREND_FEATURES
)


V2_FEATURES = V1_FEATURES + V2_NEW_FEATURES


FEATURE_GROUPS = {
    "historical": HISTORICAL_FEATURES,
    "context_statistics": CONTEXT_STAT_FEATURES,
    "recent_context": RECENT_CONTEXT_FEATURES,
    "trend": TREND_FEATURES,
}


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

ABLATION_EXPERIMENTS = {
    "V1_baseline": V1_FEATURES,

    "V2_all": V2_FEATURES,

    "V2_no_historical": [
        f
        for f in V2_FEATURES
        if f not in HISTORICAL_FEATURES
    ],

    "V2_no_context_stats": [
        f
        for f in V2_FEATURES
        if f not in CONTEXT_STAT_FEATURES
    ],

    "V2_no_recent_context": [
        f
        for f in V2_FEATURES
        if f not in RECENT_CONTEXT_FEATURES
    ],

    "V2_no_trend": [
        f
        for f in V2_FEATURES
        if f not in TREND_FEATURES
    ],
}


# =============================================================================
# LOGGING
# =============================================================================

def banner(message: str) -> None:
    print()
    print("=" * 80)
    print(message)
    print("=" * 80)


def section(message: str) -> None:
    print()
    print("-" * 80)
    print(message)
    print("-" * 80)


def info(message: str) -> None:
    print(message)


def fail(message: str) -> None:
    raise RuntimeError(message)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def create_output_directories() -> None:

    IMPORTANCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# DATASET PATHS
# =============================================================================

def v1_dataset_path(gap_length: int) -> Path:

    return (
        V1_SUPERVISED_ROOT
        / f"gap_{gap_length}"
        / (
            "CDP_00526_P_01_A-_"
            f"supervised_gap_{gap_length}.parquet"
        )
    )


def v2_dataset_path(gap_length: int) -> Path:

    return (
        V2_SUPERVISED_ROOT
        / f"gap_{gap_length}"
        / (
            "CDP_00526_P_01_A-_"
            f"supervised_gap_{gap_length}_v2.parquet"
        )
    )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_v1_dataset(
    gap_length: int,
) -> pd.DataFrame:

    path = v1_dataset_path(gap_length)

    if not path.exists():
        fail(
            f"V1 supervised dataset not found:\n{path}"
        )

    dataframe = pd.read_parquet(path)

    return dataframe


def load_v2_dataset(
    gap_length: int,
) -> pd.DataFrame:

    path = v2_dataset_path(gap_length)

    if not path.exists():
        fail(
            f"V2 supervised dataset not found:\n{path}"
        )

    dataframe = pd.read_parquet(path)

    return dataframe


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(
    dataframe: pd.DataFrame,
    features: List[str],
    gap_length: int,
) -> None:

    required = set(features)

    required.update(
        {
            "split",
            "ground_truth",
            "event_id",
            "target_index",
            "gap_position",
            "Time",
        }
    )

    missing = sorted(
        required - set(dataframe.columns)
    )

    if missing:
        fail(
            f"Gap {gap_length}: missing columns:\n"
            f"{missing}"
        )

    if dataframe.empty:
        fail(
            f"Gap {gap_length}: dataset is empty."
        )

    feature_data = dataframe[features]

    if feature_data.isna().any().any():
        bad_columns = (
            feature_data.columns[
                feature_data.isna().any()
            ].tolist()
        )

        fail(
            f"Gap {gap_length}: NaN detected in features:\n"
            f"{bad_columns}"
        )

    numeric = feature_data.select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():

        fail(
            f"Gap {gap_length}: Inf detected in features."
        )

    if dataframe["ground_truth"].isna().any():
        fail(
            f"Gap {gap_length}: ground truth contains NaN."
        )

    splits = set(
        dataframe["split"].astype(str).unique()
    )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    if not required_splits.issubset(splits):
        fail(
            f"Gap {gap_length}: invalid split labels: "
            f"{splits}"
        )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = predicted - actual

    absolute_error = np.abs(error)

    squared_error = error ** 2

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    if len(actual) >= 2:
        r2 = float(
            r2_score(
                actual,
                predicted,
            )
        )
    else:
        r2 = float("nan")

    nonzero = actual != 0

    if nonzero.any():
        mape = float(
            np.mean(
                np.abs(
                    (
                        actual[nonzero]
                        - predicted[nonzero]
                    )
                    / actual[nonzero]
                )
            )
            * 100
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
        "samples": int(len(actual)),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "mean_error_bias": bias,
        "max_absolute_error": max_absolute_error,
    }


# =============================================================================
# TRAIN / TEST
# =============================================================================

def train_and_evaluate(
    dataframe: pd.DataFrame,
    features: List[str],
) -> Dict[str, float]:

    train = dataframe[
        dataframe["split"] == "train"
    ].copy()

    test = dataframe[
        dataframe["split"] == "test"
    ].copy()

    if train.empty:
        fail("Training dataset is empty.")

    if test.empty:
        fail("Test dataset is empty.")

    X_train = train[features]

    y_train = train["ground_truth"]

    X_test = test[features]

    y_test = test["ground_truth"]

    model = RandomForestRegressor(
        **RF_PARAMETERS
    )

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        actual=y_test.to_numpy(),
        predicted=prediction,
    )

    return {
        **metrics,
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def calculate_feature_importance(
    dataframe: pd.DataFrame,
    features: List[str],
    gap_length: int,
) -> pd.DataFrame:

    train = dataframe[
        dataframe["split"] == "train"
    ].copy()

    X_train = train[features]

    y_train = train["ground_truth"]

    model = RandomForestRegressor(
        **RF_PARAMETERS
    )

    model.fit(
        X_train,
        y_train,
    )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )

    importance["gap_length"] = gap_length

    importance["feature_version"] = importance[
        "feature"
    ].apply(
        lambda x: (
            "V1"
            if x in V1_FEATURES
            else "V2_new"
        )
    )

    importance["feature_group"] = importance[
        "feature"
    ].apply(
        get_feature_group
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance["rank"] = (
        np.arange(len(importance)) + 1
    )

    return importance[
        [
            "gap_length",
            "rank",
            "feature",
            "feature_version",
            "feature_group",
            "importance",
        ]
    ]


def get_feature_group(
    feature: str,
) -> str:

    if feature in V1_FEATURES:
        return "V1_original"

    for group, features in FEATURE_GROUPS.items():

        if feature in features:
            return group

    return "unknown"


# =============================================================================
# IMPORTANCE SUMMARY
# =============================================================================

def create_importance_summary(
    all_importance: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        all_importance
        .groupby(
            [
                "feature",
                "feature_version",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            mean_importance=(
                "importance",
                "mean",
            ),
            std_importance=(
                "importance",
                "std",
            ),
            mean_rank=(
                "rank",
                "mean",
            ),
        )
    )

    summary = summary.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(drop=True)

    summary["overall_rank"] = (
        np.arange(len(summary)) + 1
    )

    return summary[
        [
            "overall_rank",
            "feature",
            "feature_version",
            "feature_group",
            "mean_importance",
            "std_importance",
            "mean_rank",
        ]
    ]


# =============================================================================
# ABLATION EXPERIMENT
# =============================================================================

def run_ablation_experiment(
    gap_length: int,
    dataframe: pd.DataFrame,
    experiment_name: str,
    features: List[str],
) -> Dict:

    metrics = train_and_evaluate(
        dataframe=dataframe,
        features=features,
    )

    return {
        "gap_length": gap_length,
        "experiment": experiment_name,
        "feature_count": len(features),
        **metrics,
    }


# =============================================================================
# DELTA CALCULATION
# =============================================================================

def add_ablation_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:

    output = results.copy()

    output[
        "MAE_change_vs_V1_percent"
    ] = np.nan

    output[
        "RMSE_change_vs_V1_percent"
    ] = np.nan

    output[
        "R2_change_vs_V1"
    ] = np.nan

    output[
        "MAPE_change_vs_V1_percent"
    ] = np.nan

    for gap_length in GAP_LENGTHS:

        mask = (
            output["gap_length"]
            == gap_length
        )

        baseline = output[
            mask
            & (
                output["experiment"]
                == "V1_baseline"
            )
        ]

        if baseline.empty:
            continue

        baseline_row = baseline.iloc[0]

        mae_base = baseline_row["MAE"]

        rmse_base = baseline_row["RMSE"]

        r2_base = baseline_row["R2"]

        mape_base = baseline_row[
            "MAPE_percent"
        ]

        gap_rows = output.loc[mask].index

        for idx in gap_rows:

            mae = output.at[
                idx,
                "MAE",
            ]

            rmse = output.at[
                idx,
                "RMSE",
            ]

            r2 = output.at[
                idx,
                "R2",
            ]

            mape = output.at[
                idx,
                "MAPE_percent",
            ]

            if mae_base != 0:

                output.at[
                    idx,
                    "MAE_change_vs_V1_percent",
                ] = (
                    (
                        mae
                        - mae_base
                    )
                    / mae_base
                    * 100
                )

            if rmse_base != 0:

                output.at[
                    idx,
                    "RMSE_change_vs_V1_percent",
                ] = (
                    (
                        rmse
                        - rmse_base
                    )
                    / rmse_base
                    * 100
                )

            if np.isfinite(r2):

                output.at[
                    idx,
                    "R2_change_vs_V1",
                ] = (
                    r2
                    - r2_base
                )

            if (
                np.isfinite(mape)
                and mape_base != 0
            ):

                output.at[
                    idx,
                    "MAPE_change_vs_V1_percent",
                ] = (
                    (
                        mape
                        - mape_base
                    )
                    / mape_base
                    * 100
                )

    return output


# =============================================================================
# PRINT IMPORTANCE
# =============================================================================

def print_top_features(
    importance: pd.DataFrame,
    gap_length: int,
    top_n: int = 15,
) -> None:

    section(
        f"TOP FEATURE IMPORTANCE — GAP {gap_length} LP"
    )

    display = importance.head(top_n)

    print(
        display[
            [
                "rank",
                "feature",
                "feature_version",
                "feature_group",
                "importance",
            ]
        ].to_string(
            index=False,
            formatters={
                "importance": "{:.6f}".format,
            },
        )
    )


# =============================================================================
# PRINT ABLATION RESULTS
# =============================================================================

def print_ablation_results(
    results: pd.DataFrame,
    gap_length: int,
) -> None:

    section(
        f"FEATURE ABLATION RESULTS — GAP {gap_length} LP"
    )

    data = results[
        results["gap_length"]
        == gap_length
    ].copy()

    columns = [
        "experiment",
        "feature_count",
        "MAE",
        "RMSE",
        "R2",
        "MAPE_percent",
        "mean_error_bias",
        "MAE_change_vs_V1_percent",
        "R2_change_vs_V1",
    ]

    print(
        data[columns].to_string(
            index=False,
            formatters={
                "MAE": "{:.4f}".format,
                "RMSE": "{:.4f}".format,
                "R2": "{:.4f}".format,
                "MAPE_percent": "{:.4f}".format,
                "mean_error_bias": "{:.4f}".format,
                "MAE_change_vs_V1_percent": (
                    "{:.2f}".format
                ),
                "R2_change_vs_V1": (
                    "{:.4f}".format
                ),
            },
        )
    )


# =============================================================================
# SAVE CONFIGURATION
# =============================================================================

def save_metadata() -> None:

    metadata = {
        "experiment": (
            "V2 feature importance and ablation"
        ),
        "version": "v2",
        "gap_lengths": GAP_LENGTHS,
        "random_state": RANDOM_STATE,
        "random_forest_parameters": RF_PARAMETERS,
        "v1_feature_count": len(V1_FEATURES),
        "v2_feature_count": len(V2_FEATURES),
        "v2_new_feature_count": len(
            V2_NEW_FEATURES
        ),
        "feature_groups": FEATURE_GROUPS,
        "experiments": {
            name: features
            for name, features
            in ABLATION_EXPERIMENTS.items()
        },
        "v1_unchanged": True,
        "v2_existing_outputs_unchanged": True,
        "mlflow": False,
    }

    path = (
        SUMMARY_ROOT
        / "ablation_metadata_v2.json"
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "RANDOM FOREST CDP IMPUTATION — "
        "V2 FEATURE IMPORTANCE + ABLATION"
    )

    print()
    print(
        "V1 pipeline              : FROZEN"
    )
    print(
        "Existing V2 model        : FROZEN"
    )
    print(
        "V2 supervised datasets   : FROZEN"
    )
    print(
        "Gap lengths              : 1, 6, 24, 48 LP"
    )
    print(
        "Feature importance       : ENABLED"
    )
    print(
        "Feature ablation        : ENABLED"
    )
    print(
        "MLflow                   : NOT YET"
    )
    print(
        "Source modification      : DISABLED"
    )

    create_output_directories()

    all_importance = []

    all_ablation = []

    # -------------------------------------------------------------------------
    # GAP LOOP
    # -------------------------------------------------------------------------

    for gap_length in GAP_LENGTHS:

        section(
            f"GAP {gap_length} LP"
        )

        v2 = load_v2_dataset(
            gap_length
        )

        validate_dataset(
            dataframe=v2,
            features=V2_FEATURES,
            gap_length=gap_length,
        )

        info(
            f"V2 dataset rows       : {len(v2)}"
        )

        info(
            f"V2 predictor features : "
            f"{len(V2_FEATURES)}"
        )

        # ---------------------------------------------------------------------
        # FEATURE IMPORTANCE
        # ---------------------------------------------------------------------

        importance = calculate_feature_importance(
            dataframe=v2,
            features=V2_FEATURES,
            gap_length=gap_length,
        )

        all_importance.append(
            importance
        )

        importance_path = (
            IMPORTANCE_ROOT
            / (
                f"feature_importance_gap_"
                f"{gap_length}.csv"
            )
        )

        importance.to_csv(
            importance_path,
            index=False,
        )

        print_top_features(
            importance=importance,
            gap_length=gap_length,
        )

        # ---------------------------------------------------------------------
        # ABLATION EXPERIMENTS
        # ---------------------------------------------------------------------

        for experiment_name, features in (
            ABLATION_EXPERIMENTS.items()
        ):

            info(
                f"Running: {experiment_name} "
                f"({len(features)} features)"
            )

            # V1 baseline uses the original V1 dataset.
            if experiment_name == "V1_baseline":

                dataframe = load_v1_dataset(
                    gap_length
                )

                validate_dataset(
                    dataframe=dataframe,
                    features=V1_FEATURES,
                    gap_length=gap_length,
                )

            else:

                dataframe = v2

                validate_dataset(
                    dataframe=dataframe,
                    features=features,
                    gap_length=gap_length,
                )

            result = run_ablation_experiment(
                gap_length=gap_length,
                dataframe=dataframe,
                experiment_name=experiment_name,
                features=features,
            )

            all_ablation.append(
                result
            )

    # =========================================================================
    # CONSOLIDATE FEATURE IMPORTANCE
    # =========================================================================

    banner(
        "CREATING CONSOLIDATED FEATURE IMPORTANCE"
    )

    importance_dataframe = pd.concat(
        all_importance,
        ignore_index=True,
    )

    importance_summary = (
        create_importance_summary(
            importance_dataframe
        )
    )

    importance_dataframe.to_csv(
        SUMMARY_ROOT
        / "all_feature_importance_v2.csv",
        index=False,
    )

    importance_summary.to_csv(
        SUMMARY_ROOT
        / "feature_importance_summary_v2.csv",
        index=False,
    )

    # =========================================================================
    # CONSOLIDATE ABLATION
    # =========================================================================

    banner(
        "CREATING CONSOLIDATED ABLATION RESULTS"
    )

    ablation_dataframe = pd.DataFrame(
        all_ablation
    )

    ablation_dataframe = (
        add_ablation_deltas(
            ablation_dataframe
        )
    )

    ablation_dataframe = (
        ablation_dataframe.sort_values(
            [
                "gap_length",
                "MAE",
            ]
        )
        .reset_index(drop=True)
    )

    ablation_dataframe.to_csv(
        SUMMARY_ROOT
        / "v2_ablation_results.csv",
        index=False,
    )

    # =========================================================================
    # PRINT FINAL SUMMARY
    # =========================================================================

    banner(
        "V2 ABLATION SUMMARY"
    )

    for gap_length in GAP_LENGTHS:

        data = ablation_dataframe[
            ablation_dataframe["gap_length"]
            == gap_length
        ]

        best = data.loc[
            data["MAE"].idxmin()
        ]

        print()
        print(
            f"Gap {gap_length} LP"
        )

        print(
            f"Best experiment : "
            f"{best['experiment']}"
        )

        print(
            f"Best MAE        : "
            f"{best['MAE']:.4f}"
        )

        print(
            f"Best RMSE       : "
            f"{best['RMSE']:.4f}"
        )

        print(
            f"Best R2         : "
            f"{best['R2']:.4f}"
        )

    # =========================================================================
    # FEATURE GROUP IMPORTANCE
    # =========================================================================

    group_summary = (
        importance_dataframe
        .groupby(
            [
                "gap_length",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            total_importance=(
                "importance",
                "sum",
            ),
            mean_importance=(
                "importance",
                "mean",
            ),
        )
        .sort_values(
            [
                "gap_length",
                "total_importance",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    group_summary.to_csv(
        SUMMARY_ROOT
        / "feature_group_importance_v2.csv",
        index=False,
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    save_metadata()

    # =========================================================================
    # FINAL
    # =========================================================================

    banner(
        "V2 FEATURE IMPORTANCE + ABLATION COMPLETE"
    )

    print()
    print(
        "V1 model/predictions    : UNCHANGED"
    )
    print(
        "V2 model/predictions    : UNCHANGED"
    )
    print(
        "V2 datasets             : UNCHANGED"
    )
    print(
        "Temporary RF models     : NOT SAVED"
    )
    print(
        "MLflow                  : NOT ADDED"
    )
    print(
        "Source data             : UNCHANGED"
    )

    print()
    print(
        "Output directory:"
    )
    print(
        f"    {OUTPUT_ROOT}"
    )

    print()
    print(
        "Key outputs:"
    )
    print(
        f"    {SUMMARY_ROOT / 'all_feature_importance_v2.csv'}"
    )
    print(
        f"    {SUMMARY_ROOT / 'feature_importance_summary_v2.csv'}"
    )
    print(
        f"    {SUMMARY_ROOT / 'feature_group_importance_v2.csv'}"
    )
    print(
        f"    {SUMMARY_ROOT / 'v2_ablation_results.csv'}"
    )
    print(
        f"    {SUMMARY_ROOT / 'ablation_metadata_v2.json'}"
    )


if __name__ == "__main__":
    main()