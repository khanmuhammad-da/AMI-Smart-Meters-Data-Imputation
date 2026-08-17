"""
V1 TUNED RANDOM FOREST — DETAILED REPORT

Generates a self-contained text report from the already-trained V1 tuned
models and their saved prediction / tuning / feature-importance outputs.

Run from project root:
    python -m src.report_tuned

Output:
    outputs/model_tuned/report/v1_tuned_detailed_report.txt

This script DOES NOT:
- retrain models
- modify models
- modify predictions
- modify source data
- use JMR
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GAP_LENGTHS = [1, 6, 24, 48]

MODEL_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "models"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "predictions"
METRICS_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "metrics"
IMPORTANCE_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "feature_importance"
METADATA_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "metadata"
TUNING_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "summary"
REPORT_DIR = PROJECT_ROOT / "outputs" / "model_tuned" / "report"
REPORT_FILE = REPORT_DIR / "v1_tuned_detailed_report.txt"


def line(char="=", n=100):
    return char * n


def fmt_num(value, digits=4):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def fmt_int(value):
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate regression metrics without requiring sklearn."""
    y_true = pd.to_numeric(df["ground_truth"], errors="coerce")
    y_pred = pd.to_numeric(df["prediction"], errors="coerce")

    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].to_numpy(dtype=float)
    y_pred = y_pred[valid].to_numpy(dtype=float)

    if len(y_true) == 0:
        return {
            "samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE_percent": np.nan,
            "mean_error_bias": np.nan,
            "max_absolute_error": np.nan,
        }

    error = y_pred - y_true
    absolute_error = np.abs(error)

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    nonzero = y_true != 0
    mape = (
        np.nan
        if not np.any(nonzero)
        else float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100)
    )

    return {
        "samples": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "mean_error_bias": float(np.mean(error)),
        "max_absolute_error": float(np.max(absolute_error)),
    }


def load_model(gap):
    path = MODEL_DIR / f"random_forest_v1_tuned_gap_{gap}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path), path


def load_predictions(gap):
    path = PREDICTION_DIR / f"predictions_gap_{gap}_v1_tuned.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return pd.read_parquet(path), path


def load_tuning_results(gap):
    path = TUNING_DIR / f"tuning_results_gap_{gap}.csv"
    if not path.exists():
        return None, path
    return pd.read_csv(path), path


def load_saved_metrics(gap):
    path = METRICS_DIR / f"metrics_gap_{gap}_v1_tuned.json"
    if not path.exists():
        return None, path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def load_metadata(gap):
    path = METADATA_DIR / f"model_metadata_gap_{gap}_v1_tuned.json"
    if not path.exists():
        return None, path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def load_importance(gap):
    candidates = [
        IMPORTANCE_DIR / f"feature_importance_gap_{gap}_tuned.csv",
        IMPORTANCE_DIR / f"feature_importance_gap_{gap}_v1_tuned.csv",
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path), path

    return None, candidates[0]


def get_feature_names(model, predictions):
    """Prefer the exact features recorded by the fitted RF."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    excluded = {
        "ground_truth",
        "prediction",
        "error",
        "absolute_error",
        "squared_error",
        "percentage_error",
        "split",
        "event_id",
        "gap_length",
        "target_index",
        "gap_position",
        "gap_position_fraction",
        "Time",
        "feature_version",
    }

    return [
        c for c in predictions.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(predictions[c])
    ]


def get_best_parameters(tuning_df):
    if tuning_df is None or tuning_df.empty:
        return None

    # The tuning code may use either validation_MAE or MAE.
    metric_col = None
    for candidate in [
        "validation_MAE",
        "val_MAE",
        "MAE",
        "validation_mae",
    ]:
        if candidate in tuning_df.columns:
            metric_col = candidate
            break

    if metric_col is None:
        return None

    work = tuning_df.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col])

    if work.empty:
        return None

    row = work.loc[work[metric_col].idxmin()]

    parameter_names = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ]

    result = {}
    for name in parameter_names:
        if name not in row.index:
            continue

        value = row[name]

        if pd.isna(value):
            value = None
        elif name in {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        }:
            value = int(value)

        result[name] = value

    return result


def importance_dataframe(model, gap):
    importance_df, path = load_importance(gap)

    if importance_df is not None and not importance_df.empty:
        df = importance_df.copy()

        # Normalize likely column names.
        feature_col = next(
            (c for c in ["feature", "Feature", "feature_name"] if c in df.columns),
            None,
        )
        importance_col = next(
            (
                c
                for c in [
                    "importance",
                    "Importance",
                    "feature_importance",
                    "mean_importance",
                ]
                if c in df.columns
            ),
            None,
        )

        if feature_col and importance_col:
            df = df[[feature_col, importance_col]].copy()
            df.columns = ["feature", "importance"]
            df["importance"] = pd.to_numeric(
                df["importance"], errors="coerce"
            )
            return df.sort_values("importance", ascending=False), path

    # Fallback: derive importance directly from the fitted model.
    if hasattr(model, "feature_importances_"):
        names = get_feature_names(model, pd.DataFrame())
        df = pd.DataFrame(
            {
                "feature": names,
                "importance": model.feature_importances_,
            }
        )
        return df.sort_values("importance", ascending=False), path

    return None, path


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = []

    report.append(line())
    report.append("RANDOM FOREST CDP IMPUTATION — V1 TUNED DETAILED REPORT")
    report.append(line())
    report.append("")
    report.append("Purpose:")
    report.append("  Detailed record of the frozen V1 tuned Random Forest models.")
    report.append("  This report is generated from saved artifacts only.")
    report.append("")
    report.append("IMPORTANT:")
    report.append("  Training/retraining        : NO")
    report.append("  Predictions modified       : NO")
    report.append("  Source CSV modified        : NO")
    report.append("  JMR used                   : NO")
    report.append("  MLflow used                : NO")
    report.append("  Evaluation split           : TEST")
    report.append("  Gap lengths                : 1, 6, 24, 48 LP")
    report.append("")

    all_metrics = []

    for gap in GAP_LENGTHS:
        model, model_path = load_model(gap)
        predictions, prediction_path = load_predictions(gap)
        tuning_df, tuning_path = load_tuning_results(gap)
        saved_metrics, metrics_path = load_saved_metrics(gap)
        metadata, metadata_path = load_metadata(gap)

        features = get_feature_names(model, predictions)
        best_params = get_best_parameters(tuning_df)

        test_df = predictions[
            predictions["split"].astype(str).str.lower() == "test"
        ].copy()

        if "Time" in test_df.columns:
            test_df["Time"] = pd.to_datetime(
                test_df["Time"], errors="coerce"
            )

        test_metrics = calculate_metrics(test_df)
        all_metrics.append({"gap_length": gap, **test_metrics})

        report.append("")
        report.append(line("-"))
        report.append(f"GAP LENGTH: {gap} LP")
        report.append(line("-"))
        report.append("")

        report.append("MODEL")
        report.append(f"  Algorithm                : {type(model).__name__}")
        report.append(f"  Model file               : {model_path}")
        report.append(f"  Prediction file          : {prediction_path}")
        report.append(f"  Tuning results           : {tuning_path}")
        report.append(f"  Metadata                 : {metadata_path}")
        report.append("")

        report.append("FEATURES")
        report.append(f"  Total predictor features : {len(features)}")
        for i, feature in enumerate(features, 1):
            report.append(f"    {i:02d}. {feature}")
        report.append("")

        report.append("BEST PARAMETERS")
        if best_params:
            for key, value in best_params.items():
                report.append(f"  {key:22s}: {value}")
        else:
            report.append("  Not available from tuning CSV.")
        report.append("")

        report.append("DATASET / SPLITS")
        report.append(f"  Total prediction rows     : {len(predictions):,}")
        report.append(f"  Test rows                 : {len(test_df):,}")

        if "split" in predictions.columns:
            split_counts = predictions["split"].value_counts()
            for split_name in ["train", "validation", "test"]:
                if split_name in split_counts.index:
                    report.append(
                        f"  {split_name.capitalize():24s}: "
                        f"{int(split_counts[split_name]):,}"
                    )
        report.append("")

        report.append("TEST PERFORMANCE")
        report.append(f"  Samples                  : {fmt_int(test_metrics['samples'])}")
        report.append(f"  MAE                      : {fmt_num(test_metrics['MAE'])}")
        report.append(f"  RMSE                     : {fmt_num(test_metrics['RMSE'])}")
        report.append(f"  R2                       : {fmt_num(test_metrics['R2'])}")
        report.append(
            f"  MAPE                     : "
            f"{fmt_num(test_metrics['MAPE_percent'])}%"
        )
        report.append(
            f"  Mean error bias          : "
            f"{fmt_num(test_metrics['mean_error_bias'])}"
        )
        report.append(
            f"  Maximum absolute error   : "
            f"{fmt_num(test_metrics['max_absolute_error'])}"
        )
        report.append("")

        report.append("GROUND TRUTH VS PREDICTED — EVERY TEST LP")
        report.append("-" * 100)

        display_columns = [
            "split",
            "event_id",
            "target_index",
            "gap_position",
            "Time",
            "ground_truth",
            "prediction",
        ]

        available = [c for c in display_columns if c in test_df.columns]
        detail = test_df[available].copy()

        if "ground_truth" in detail.columns and "prediction" in detail.columns:
            detail["error"] = (
                detail["prediction"] - detail["ground_truth"]
            )
            detail["absolute_error"] = detail["error"].abs()
            detail["percentage_error"] = np.where(
                detail["ground_truth"] != 0,
                detail["error"] / detail["ground_truth"] * 100,
                np.nan,
            )

        if "Time" in detail.columns:
            detail["Time"] = detail["Time"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        if "ground_truth" in detail.columns:
            detail["ground_truth"] = detail["ground_truth"].map(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            )

        if "prediction" in detail.columns:
            detail["prediction"] = detail["prediction"].map(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            )

        if "error" in detail.columns:
            detail["error"] = detail["error"].map(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            )

        if "absolute_error" in detail.columns:
            detail["absolute_error"] = detail["absolute_error"].map(
                lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
            )

        if "percentage_error" in detail.columns:
            detail["percentage_error"] = detail["percentage_error"].map(
                lambda x: f"{x:,.2f}%" if pd.notna(x) else "N/A"
            )

        report.append(
            detail.to_string(index=False)
        )
        report.append("")

        report.append("FEATURE IMPORTANCE")
        report.append("-" * 100)

        imp_df, imp_path = importance_dataframe(model, gap)

        if imp_df is None or imp_df.empty:
            report.append("  Feature importance not available.")
        else:
            report.append(f"  Source: {imp_path}")
            report.append("")
            report.append(
                f"{'Rank':>6} {'Feature':<45} {'Importance':>15}"
            )
            report.append("-" * 72)

            for rank, (_, row) in enumerate(imp_df.iterrows(), 1):
                feature = str(row["feature"])
                importance = row["importance"]

                report.append(
                    f"{rank:>6} "
                    f"{feature:<45.45} "
                    f"{importance:>15.8f}"
                )
        report.append("")

    # Consolidated summary
    summary_df = pd.DataFrame(all_metrics)

    report.append("")
    report.append(line())
    report.append("CONSOLIDATED TEST PERFORMANCE")
    report.append(line())
    report.append("")
    report.append(
        f"{'Gap':>8} {'Samples':>10} {'MAE':>15} {'RMSE':>15} "
        f"{'R2':>12} {'MAPE %':>12} {'Bias':>15} {'Max Abs Error':>18}"
    )
    report.append("-" * 115)

    for _, row in summary_df.iterrows():
        report.append(
            f"{int(row['gap_length']):>8} "
            f"{int(row['samples']):>10} "
            f"{row['MAE']:>15,.4f} "
            f"{row['RMSE']:>15,.4f} "
            f"{row['R2']:>12.4f} "
            f"{row['MAPE_percent']:>12.4f} "
            f"{row['mean_error_bias']:>15,.4f} "
            f"{row['max_absolute_error']:>18,.4f}"
        )

    report.append("")
    report.append("BEST GAP BY MAE")
    best_mae = summary_df.loc[summary_df["MAE"].idxmin()]
    report.append(
        f"  Gap {int(best_mae['gap_length'])} LP "
        f"(MAE = {best_mae['MAE']:,.4f})"
    )

    report.append("")
    report.append("BEST GAP BY RMSE")
    best_rmse = summary_df.loc[summary_df["RMSE"].idxmin()]
    report.append(
        f"  Gap {int(best_rmse['gap_length'])} LP "
        f"(RMSE = {best_rmse['RMSE']:,.4f})"
    )

    report.append("")
    report.append("BEST GAP BY R2")
    best_r2 = summary_df.loc[summary_df["R2"].idxmax()]
    report.append(
        f"  Gap {int(best_r2['gap_length'])} LP "
        f"(R2 = {best_r2['R2']:.4f})"
    )

    report.append("")
    report.append("BEST GAP BY MAPE")
    best_mape = summary_df.loc[summary_df["MAPE_percent"].idxmin()]
    report.append(
        f"  Gap {int(best_mape['gap_length'])} LP "
        f"(MAPE = {best_mape['MAPE_percent']:.4f}%)"
    )

    report.append("")
    report.append(line())
    report.append("REPORT COMPLETE")
    report.append(line())
    report.append(f"Saved to: {REPORT_FILE}")

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print(line())
    print("V1 TUNED RANDOM FOREST — DETAILED REPORT")
    print(line())
    print("")
    print(f"Report saved:")
    print(f"    {REPORT_FILE}")
    print("")
    print("Consolidated test performance:")
    print(summary_df.to_string(index=False))
    print("")
    print("No models were retrained.")
    print("No predictions were modified.")
    print("JMR was not used.")


if __name__ == "__main__":
    main()