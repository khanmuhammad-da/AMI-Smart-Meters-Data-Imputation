# =============================================================================
# src/config.py
# =============================================================================
"""
Central configuration loader for the Random Forest CDP Load Profile
Imputation project.

Project structure:

D:\\02- Personal\\08- GIKI Bootcamp
│
├── config
│   └── config.yaml
│
├── data
│   ├── raw
│   └── processed
│
├── models
├── outputs
├── src
└── tests

This module:
    1. Determines the project root.
    2. Loads config/config.yaml.
    3. Exposes the configuration as CONFIG.
    4. Provides helper functions for configuration access.

It does NOT:
    - modify datasets
    - perform feature engineering
    - generate gaps
    - create supervised datasets
    - train models
    - evaluate models
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# PROJECT PATHS
# =============================================================================

# src/config.py
#
# parents[0] = src
# parents[1] = project root
#
# Therefore:
# D:\02- Personal\08- GIKI Bootcamp\src\config.py
#
# becomes:
# D:\02- Personal\08- GIKI Bootcamp

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "config.yaml"
)


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def load_config(
    config_path: Path = CONFIG_PATH,
) -> dict:
    """
    Load the project YAML configuration.

    Parameters
    ----------
    config_path:
        Path to config.yaml.

    Returns
    -------
    dict
        Loaded configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist.

    ValueError
        If the YAML file does not contain
        a dictionary.
    """

    if not config_path.exists():

        raise FileNotFoundError(
            "Configuration file not found:\n"
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if not isinstance(
        config,
        dict,
    ):

        raise ValueError(
            "config.yaml must contain "
            "a dictionary."
        )

    return config


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
#
# This is intentionally loaded at module import time so that other modules
# can simply use:
#
#     from src.config import CONFIG
#
# Example:
#
#     CONFIG["data"]["selected_meter"]
#
# =============================================================================

CONFIG = load_config()

# Add resolved project root for modules that need filesystem paths.
CONFIG["project_root"] = str(PROJECT_ROOT)


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def get_config(
    *keys: str,
) -> Any:
    """
    Retrieve a nested configuration value.

    Example
    -------
    get_config("data", "selected_meter")

    is equivalent to:

    CONFIG["data"]["selected_meter"]
    """

    value: Any = CONFIG

    for key in keys:

        if not isinstance(
            value,
            dict,
        ):

            raise KeyError(
                "Cannot access configuration key "
                f"'{key}' because the parent "
                "configuration value is not a dictionary."
            )

        if key not in value:

            raise KeyError(
                "Configuration key not found: "
                f"{'.'.join(keys)}"
            )

        value = value[key]

    return value


# =============================================================================
# PATH HELPER
# =============================================================================

def resolve_project_path(
    path_value: str | Path,
) -> Path:
    """
    Resolve a project-relative path.

    Relative paths are resolved from PROJECT_ROOT.

    Absolute paths are returned unchanged.

    Examples
    --------
    resolve_project_path(
        "data/raw/Merged_All_Year_2024-25.csv"
    )

    ->

    D:\\02- Personal\\08- GIKI Bootcamp\\
    data\\raw\\Merged_All_Year_2024-25.csv
    """

    path = Path(
        path_value
    )

    if path.is_absolute():

        return path

    return (
        PROJECT_ROOT
        / path
    )


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config() -> None:
    """
    Validate the configuration required by the project.

    This performs configuration checks only.
    It does not inspect or modify datasets.
    """

    required_sections = [
        "project",
        "data",
        "dataset",
        "feature_engineering",
        "split",
        "gaps",
        "supervised",
        "model",
        "evaluation",
        "outputs",
    ]

    for section in required_sections:

        if section not in CONFIG:

            raise ValueError(
                f"Required configuration section "
                f"'{section}' is missing."
            )

    # -------------------------------------------------------------------------
    # Data configuration
    # -------------------------------------------------------------------------

    required_data_keys = [
        "input_file",
        "selected_meter",
        "target_direction",
        "timestamp_column",
        "delimiter",
        "timestamp_format",
        "expected_interval_minutes",
        "never_modify_source_data",
    ]

    for key in required_data_keys:

        if key not in CONFIG["data"]:

            raise ValueError(
                f"Required configuration key "
                f"'data.{key}' is missing."
            )

    # -------------------------------------------------------------------------
    # Split configuration
    # -------------------------------------------------------------------------

    split = CONFIG["split"]

    train_ratio = float(
        split["train_ratio"]
    )

    validation_ratio = float(
        split["validation_ratio"]
    )

    test_ratio = float(
        split["test_ratio"]
    )

    ratio_sum = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(
        ratio_sum - 1.0
    ) > 1e-9:

        raise ValueError(
            "Train/validation/test ratios "
            f"must sum to 1.0. "
            f"Current sum: {ratio_sum}"
        )

    # -------------------------------------------------------------------------
    # Gap configuration
    # -------------------------------------------------------------------------

    gap_lengths = [
        int(value)
        for value
        in CONFIG["gaps"]["lengths"]
    ]

    approved_gap_lengths = [
        1,
        6,
        24,
        48,
    ]

    if gap_lengths != approved_gap_lengths:

        raise ValueError(
            "Gap lengths do not match the "
            "approved experiment design.\n"
            f"Expected: {approved_gap_lengths}\n"
            f"Configured: {gap_lengths}"
        )

    # -------------------------------------------------------------------------
    # Supervised formulation
    # -------------------------------------------------------------------------

    formulation = (
        CONFIG["supervised"]
        ["formulation"]
    )

    if formulation != "fixed_event_window":

        raise ValueError(
            "Supervised formulation must be "
            "'fixed_event_window'.\n"
            f"Configured: {formulation}"
        )

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------

    model = CONFIG["model"]

    if model["algorithm"] != (
        "RandomForestRegressor"
    ):

        raise ValueError(
            "This project currently requires "
            "RandomForestRegressor.\n"
            f"Configured: {model['algorithm']}"
        )

    # -------------------------------------------------------------------------
    # Important project constraints
    # -------------------------------------------------------------------------

    if CONFIG["data"][
        "never_modify_source_data"
    ] is not True:

        raise ValueError(
            "data.never_modify_source_data "
            "must be true."
        )

    if CONFIG[
        "feature_engineering"
    ]["include_target_lags"]:

        raise ValueError(
            "Target-derived lag features "
            "must remain disabled."
        )

    if CONFIG[
        "feature_engineering"
    ]["include_target_leads"]:

        raise ValueError(
            "Target-derived lead features "
            "must remain disabled."
        )

    if CONFIG[
        "feature_engineering"
    ]["normalize"]:

        raise ValueError(
            "Normalization must remain "
            "disabled."
        )

    if CONFIG[
        "supervised"
    ]["formulation"] != "fixed_event_window":

        raise ValueError(
            "Sliding-window formulation is "
            "not permitted."
        )


# =============================================================================
# VALIDATE ON IMPORT
# =============================================================================

validate_config()


# =============================================================================
# CONFIGURATION TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "RANDOM FOREST CDP IMPUTATION"
    )
    print(
        "CONFIGURATION TEST"
    )
    print("=" * 80)

    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        "Configuration file:"
    )

    print(
        CONFIG_PATH
    )

    print()

    print(
        "Configuration loading: PASSED"
    )

    print()

    print(
        "Configuration validation: PASSED"
    )

    print()

    print(
        "Selected meter:"
    )

    print(
        f"    {CONFIG['data']['selected_meter']}"
    )

    print()

    print(
        "Target direction:"
    )

    print(
        f"    {CONFIG['data']['target_direction']}"
    )

    print()

    print(
        "Gap experiments:"
    )

    for gap in CONFIG["gaps"]["lengths"]:

        print(
            f"    {gap} LP"
        )

    print()

    print(
        "Supervised formulation:"
    )

    print(
        f"    {CONFIG['supervised']['formulation']}"
    )

    print()

    print(
        "One prediction per missing LP:"
    )

    print(
        "    ENABLED"
    )

    print()

    print(
        "Target lags:"
    )

    print(
        "    DISABLED"
    )

    print()

    print(
        "Target leads:"
    )

    print(
        "    DISABLED"
    )

    print()

    print(
        "Normalization:"
    )

    print(
        "    DISABLED"
    )

    print()

    print(
        "96 LP experiment:"
    )

    print(
        "    REMOVED"
    )

    print()

    print(
        "Configuration:"
    )

    print(
        CONFIG
    )

    print()

    print(
        "CONFIGURATION TEST COMPLETE"
    )

    print(
        "=" * 80
    )