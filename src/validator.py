"""Validation module for AeroGuard AI Phase 1 dataset integrity checks."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

from src.config import (
    REPORTS_DIR,
    EXPECTED_COLUMNS_COUNT,
    EXPECTED_TRAIN_UNITS,
    EXPECTED_TEST_UNITS,
    EXPECTED_RUL_COUNT,
    ALL_COLUMNS,
)


def validate_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    save_reports: bool = True,
    output_dir: Path = REPORTS_DIR
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run comprehensive data integrity checks on train, test, and RUL datasets.

    Checks:
    1. Dataset shapes and column alignment (26 columns)
    2. Engine counts (100 train units, 100 test units)
    3. Missing / null values across all features
    4. Duplicate rows and (unit, cycle) primary key uniqueness
    5. Cycle ranges (min cycle == 1, max cycle distribution, monotonicity)

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    rul_df : pd.DataFrame
    save_reports : bool
        Whether to export JSON and Markdown validation reports.
    output_dir : Path
        Directory where reports will be saved.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (all_passed, validation_report_dict)
    """
    checks_passed = True
    report: Dict[str, Any] = {
        "status": "PENDING",
        "shapes": {},
        "engine_counts": {},
        "missing_values": {},
        "duplicate_rows": {},
        "cycle_ranges": {},
        "failures": []
    }

    # 1. Dataset Shapes and Column Count
    train_shape = list(train_df.shape)
    test_shape = list(test_df.shape)
    rul_shape = list(rul_df.shape)

    report["shapes"] = {
        "train": {"rows": train_shape[0], "columns": train_shape[1]},
        "test": {"rows": test_shape[0], "columns": test_shape[1]},
        "rul": {"rows": rul_shape[0], "columns": rul_shape[1]}
    }

    if train_shape[1] != EXPECTED_COLUMNS_COUNT:
        checks_passed = False
        report["failures"].append(
            f"Train columns count mismatch: expected {EXPECTED_COLUMNS_COUNT}, got {train_shape[1]}"
        )

    if test_shape[1] != EXPECTED_COLUMNS_COUNT:
        checks_passed = False
        report["failures"].append(
            f"Test columns count mismatch: expected {EXPECTED_COLUMNS_COUNT}, got {test_shape[1]}"
        )

    # Column name verification
    if list(train_df.columns) != ALL_COLUMNS:
        checks_passed = False
        report["failures"].append("Train column names do not match expected schema")

    if list(test_df.columns) != ALL_COLUMNS:
        checks_passed = False
        report["failures"].append("Test column names do not match expected schema")

    # 2. Engine Counts
    train_units = int(train_df["unit_number"].nunique())
    test_units = int(test_df["unit_number"].nunique())
    rul_units = int(rul_df["unit_number"].nunique())

    report["engine_counts"] = {
        "train_unique_engines": train_units,
        "test_unique_engines": test_units,
        "rul_engine_count": rul_units,
        "train_unit_ids_min_max": [int(train_df["unit_number"].min()), int(train_df["unit_number"].max())],
        "test_unit_ids_min_max": [int(test_df["unit_number"].min()), int(test_df["unit_number"].max())]
    }

    if train_units != EXPECTED_TRAIN_UNITS:
        checks_passed = False
        report["failures"].append(
            f"Train unit count mismatch: expected {EXPECTED_TRAIN_UNITS}, got {train_units}"
        )

    if test_units != EXPECTED_TEST_UNITS:
        checks_passed = False
        report["failures"].append(
            f"Test unit count mismatch: expected {EXPECTED_TEST_UNITS}, got {test_units}"
        )

    if rul_units != EXPECTED_RUL_COUNT:
        checks_passed = False
        report["failures"].append(
            f"RUL count mismatch: expected {EXPECTED_RUL_COUNT}, got {rul_units}"
        )

    # 3. Missing Values
    train_nans = int(train_df.isna().sum().sum())
    test_nans = int(test_df.isna().sum().sum())
    rul_nans = int(rul_df.isna().sum().sum())

    report["missing_values"] = {
        "train_total_nans": train_nans,
        "test_total_nans": test_nans,
        "rul_total_nans": rul_nans
    }

    if train_nans > 0 or test_nans > 0 or rul_nans > 0:
        checks_passed = False
        report["failures"].append(
            f"Missing values detected: train={train_nans}, test={test_nans}, rul={rul_nans}"
        )

    # 4. Duplicate Rows & Key Uniqueness
    train_duplicates = int(train_df.duplicated().sum())
    test_duplicates = int(test_df.duplicated().sum())
    train_key_duplicates = int(train_df.duplicated(subset=["unit_number", "cycle"]).sum())
    test_key_duplicates = int(test_df.duplicated(subset=["unit_number", "cycle"]).sum())

    report["duplicate_rows"] = {
        "train_full_row_duplicates": train_duplicates,
        "test_full_row_duplicates": test_duplicates,
        "train_unit_cycle_duplicates": train_key_duplicates,
        "test_unit_cycle_duplicates": test_key_duplicates
    }

    if train_duplicates > 0 or test_duplicates > 0:
        checks_passed = False
        report["failures"].append("Duplicate rows detected in raw dataset")

    if train_key_duplicates > 0 or test_key_duplicates > 0:
        checks_passed = False
        report["failures"].append("(unit_number, cycle) primary key uniqueness violated")

    # 5. Cycle Ranges & Monotonicity
    train_min_cycles = train_df.groupby("unit_number")["cycle"].min()
    test_min_cycles = test_df.groupby("unit_number")["cycle"].min()
    train_max_cycles = train_df.groupby("unit_number")["cycle"].max()
    test_max_cycles = test_df.groupby("unit_number")["cycle"].max()

    train_all_start_at_1 = bool((train_min_cycles == 1).all())
    test_all_start_at_1 = bool((test_min_cycles == 1).all())

    # Check monotonicity
    train_is_monotonic = bool(
        train_df.groupby("unit_number")["cycle"].apply(lambda s: s.is_monotonic_increasing).all()
    )
    test_is_monotonic = bool(
        test_df.groupby("unit_number")["cycle"].apply(lambda s: s.is_monotonic_increasing).all()
    )

    report["cycle_ranges"] = {
        "train_engines_start_at_cycle_1": train_all_start_at_1,
        "test_engines_start_at_cycle_1": test_all_start_at_1,
        "train_cycles_monotonic": train_is_monotonic,
        "test_cycles_monotonic": test_is_monotonic,
        "train_max_cycle_stats": {
            "min": int(train_max_cycles.min()),
            "max": int(train_max_cycles.max()),
            "mean": round(float(train_max_cycles.mean()), 2),
            "median": float(train_max_cycles.median()),
            "std": round(float(train_max_cycles.std()), 2)
        },
        "test_max_cycle_stats": {
            "min": int(test_max_cycles.min()),
            "max": int(test_max_cycles.max()),
            "mean": round(float(test_max_cycles.mean()), 2),
            "median": float(test_max_cycles.median()),
            "std": round(float(test_max_cycles.std()), 2)
        }
    }

    if not train_all_start_at_1 or not test_all_start_at_1:
        checks_passed = False
        report["failures"].append("Not all engine trajectories begin at cycle 1")

    if not train_is_monotonic or not test_is_monotonic:
        checks_passed = False
        report["failures"].append("Cycles are not monotonically increasing per engine unit")

    report["status"] = "PASSED" if checks_passed else "FAILED"

    # Save reports if requested
    if save_reports:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "data_validation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_path = output_dir / "data_validation_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_generate_markdown_summary(report))

    return checks_passed, report


def _generate_markdown_summary(report: Dict[str, Any]) -> str:
    """Format validation dictionary into a clean markdown summary."""
    status_badge = "PASS" if report["status"] == "PASSED" else "FAIL"
    train_s = report["cycle_ranges"]["train_max_cycle_stats"]
    test_s = report["cycle_ranges"]["test_max_cycle_stats"]

    md = f"""# AeroGuard AI - Phase 1 Data Validation Report

**Status:** `{status_badge}`

## 1. Dataset Dimensions & Schema
- **Training Set (train_FD001):** {report['shapes']['train']['rows']:,} rows × {report['shapes']['train']['columns']} columns
- **Testing Set (test_FD001):** {report['shapes']['test']['rows']:,} rows × {report['shapes']['test']['columns']} columns
- **Ground Truth RUL (RUL_FD001):** {report['shapes']['rul']['rows']} rows × {report['shapes']['rul']['columns']} columns
- **Column Schema:** 26 assigned columns (unit_number, cycle, 3 operational settings, 21 sensor measurements).

## 2. Fleet Engine Counts
- **Training Engines:** {report['engine_counts']['train_unique_engines']} (IDs {report['engine_counts']['train_unit_ids_min_max'][0]} to {report['engine_counts']['train_unit_ids_min_max'][1]})
- **Testing Engines:** {report['engine_counts']['test_unique_engines']} (IDs {report['engine_counts']['test_unit_ids_min_max'][0]} to {report['engine_counts']['test_unit_ids_min_max'][1]})
- **RUL Engine Labels:** {report['engine_counts']['rul_engine_count']} ground-truth values

## 3. Data Hygiene & Integrity
- **Missing Values (NaNs):**
  - Train: {report['missing_values']['train_total_nans']}
  - Test: {report['missing_values']['test_total_nans']}
  - RUL: {report['missing_values']['rul_total_nans']}
- **Duplicate Rows:**
  - Full Row Duplicates (Train / Test): {report['duplicate_rows']['train_full_row_duplicates']} / {report['duplicate_rows']['test_full_row_duplicates']}
  - Primary Key (unit, cycle) Collisions: {report['duplicate_rows']['train_unit_cycle_duplicates']} / {report['duplicate_rows']['test_unit_cycle_duplicates']}

## 4. Cycle Range & Monotonicity
- All engines start at Cycle 1: Train = `{report['cycle_ranges']['train_engines_start_at_cycle_1']}`, Test = `{report['cycle_ranges']['test_engines_start_at_cycle_1']}`
- Cycles are strictly monotonic: Train = `{report['cycle_ranges']['train_cycles_monotonic']}`, Test = `{report['cycle_ranges']['test_cycles_monotonic']}`
- **Train Run-to-Failure Lifespan (Cycles):**
  - Min: {train_s['min']} cycles
  - Max: {train_s['max']} cycles
  - Mean: {train_s['mean']} cycles
  - Median: {train_s['median']} cycles
  - Standard Deviation: {train_s['std']} cycles
- **Test Cut-off Snapshot (Cycles):**
  - Min: {test_s['min']} cycles
  - Max: {test_s['max']} cycles
  - Mean: {test_s['mean']} cycles
  - Median: {test_s['median']} cycles

## 5. Result
**Total Failures:** {len(report['failures'])}
"""
    if report["failures"]:
        md += "\n### Detected Issues:\n"
        for failure in report["failures"]:
            md += f"- [ERROR] {failure}\n"
    else:
        md += "\nAll Phase 1 data validation checks passed successfully with 0 defects.\n"

    return md
