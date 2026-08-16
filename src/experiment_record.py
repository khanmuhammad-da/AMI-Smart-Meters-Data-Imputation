from pathlib import Path

import pandas as pd
import yaml


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"

SUPERVISED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "supervised"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "model"
    / "predictions"
)

RESIDUAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "residual"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "experiment_record_v1.txt"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8",
) as file:

    CONFIG = yaml.safe_load(file)


GAP_LENGTHS = CONFIG["gaps"]["lengths"]


# =============================================================================
# REPORT HELPERS
# =============================================================================

report = []


def write(message=""):
    report.append(str(message))


def separator(character="=", length=110):
    write(character * length)


# =============================================================================
# HEADER
# =============================================================================

separator()

write(
    "RANDOM FOREST CDP IMPUTATION — V1 EXPERIMENT RECORD"
)

separator()

write()


# =============================================================================
# 1. FEATURES
# =============================================================================

write("1. FEATURES")
write("-" * 110)

feature_file = (
    SUPERVISED_DIR
    / "gap_24"
    / "CDP_00526_P_01_A-_supervised_gap_24.parquet"
)

feature_df = pd.read_parquet(feature_file)


metadata_columns = {
    "ground_truth",
    "prediction",
    "split",
    "event_id",
    "gap_length",
    "target_index",
    "gap_position",
    "gap_position_fraction",
    "Time",
}

feature_columns = [
    column
    for column in feature_df.columns
    if column not in metadata_columns
]


write(
    f"Total features: {len(feature_columns)}"
)

write()

for number, feature in enumerate(
    feature_columns,
    start=1,
):

    write(
        f"  {number:02d}. {feature}"
    )


write()


# =============================================================================
# 2. MODEL TYPE AND PARAMETERS
# =============================================================================

write("2. MODEL TYPE AND PARAMETERS")
write("-" * 110)

model_config = CONFIG["model"]

write(
    f"Algorithm: {model_config['algorithm']}"
)

write()

for parameter, value in model_config.items():

    if parameter == "algorithm":
        continue

    write(
        f"  {parameter}: {value}"
    )


write()


# =============================================================================
# 3. GROUND TRUTH VS PREDICTION
# =============================================================================

write("3. GROUND TRUTH vs PREDICTED VALUES")
write("-" * 110)

for gap_length in GAP_LENGTHS:

    prediction_file = (
        PREDICTION_DIR
        / f"predictions_gap_{gap_length}.parquet"
    )

    if not prediction_file.exists():

        write()
        write(
            f"GAP {gap_length} LP: "
            "PREDICTION FILE NOT FOUND"
        )

        continue

    dataframe = pd.read_parquet(
        prediction_file
    )

    write()

    write(
        f"GAP {gap_length} LP — "
        f"{len(dataframe)} predictions"
    )

    write()

    columns = [
        "split",
        "event_id",
        "gap_position",
        "Time",
        "ground_truth",
        "prediction",
    ]

    write(
        dataframe[
            columns
        ].to_string(index=False)
    )


write()


# =============================================================================
# 4. ERRORS BY GAP LENGTH
# =============================================================================

write("4. ERRORS BY GAP LENGTH — TEST SET")
write("-" * 110)

header = (
    f"{'Gap':>8}"
    f"{'Samples':>10}"
    f"{'MAE':>15}"
    f"{'RMSE':>15}"
    f"{'R2':>12}"
    f"{'MAPE %':>12}"
    f"{'Bias':>15}"
    f"{'Max Abs Error':>18}"
)

write(header)

write("-" * 110)


for gap_length in GAP_LENGTHS:

    residual_file = (
        RESIDUAL_DIR
        / f"prediction_errors_gap_{gap_length}.parquet"
    )

    if not residual_file.exists():

        write(
            f"{gap_length:>8}"
            f"{'FILE NOT FOUND':>10}"
        )

        continue

    dataframe = pd.read_parquet(
        residual_file
    )

    test = dataframe[
        dataframe["split"] == "test"
    ].copy()

    if test.empty:

        write(
            f"{gap_length:>8}"
            f"{'NO TEST DATA':>10}"
        )

        continue


    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    samples = len(test)

    mae = test[
        "absolute_error"
    ].mean()

    rmse = (
        test["squared_error"].mean()
        ** 0.5
    )

    actual = test[
        "ground_truth"
    ]

    squared_errors = test[
        "squared_error"
    ]

    denominator = (
        (actual - actual.mean()) ** 2
    ).sum()

    if denominator == 0:

        r2 = float("nan")

    else:

        r2 = (
            1
            -
            squared_errors.sum()
            / denominator
        )

    mape = test[
        "percentage_error"
    ].mean()

    bias = test[
        "error"
    ].mean()

    max_absolute_error = test[
        "absolute_error"
    ].max()


    # -------------------------------------------------------------------------
    # Write row
    # -------------------------------------------------------------------------

    write(
        f"{gap_length:>8}"
        f"{samples:>10}"
        f"{mae:>15.4f}"
        f"{rmse:>15.4f}"
        f"{r2:>12.4f}"
        f"{mape:>12.4f}"
        f"{bias:>15.4f}"
        f"{max_absolute_error:>18.4f}"
    )


# =============================================================================
# FOOTER
# =============================================================================

write()

separator()

write(
    "Experiment record generated successfully."
)

write(
    f"Project root: {PROJECT_ROOT}"
)

write(
    f"Gap lengths: {GAP_LENGTHS}"
)

separator()


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8",
)


# =============================================================================
# DISPLAY
# =============================================================================

print(
    "\n".join(report)
)

print()

print(
    f"Saved: {OUTPUT_FILE}"
)