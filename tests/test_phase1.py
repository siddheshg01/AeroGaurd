"""Unit and integration tests for AeroGuard AI Phase 1."""

from pathlib import Path
import pytest
import pandas as pd

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    ALL_COLUMNS,
    EXPECTED_COLUMNS_COUNT,
    EXPECTED_TRAIN_UNITS,
    EXPECTED_TEST_UNITS,
    EXPECTED_RUL_COUNT,
    FIGURES_DIR,
    REPORTS_DIR,
)
from src.data_loader import load_dataset_file, load_rul_file, add_training_rul, load_all_data
from src.validator import validate_dataset
from src.eda import compute_sensor_statistics


def test_files_exist():
    """Verify raw dataset files exist and are non-empty."""
    assert TRAIN_FILE.exists(), f"Missing train file: {TRAIN_FILE}"
    assert TEST_FILE.exists(), f"Missing test file: {TEST_FILE}"
    assert RUL_FILE.exists(), f"Missing RUL file: {RUL_FILE}"

    assert TRAIN_FILE.stat().st_size > 0, "Train file is empty"
    assert TEST_FILE.stat().st_size > 0, "Test file is empty"
    assert RUL_FILE.stat().st_size > 0, "RUL file is empty"


def test_dataset_loading_and_columns():
    """Verify loading produces 26 exact column names and expected row counts."""
    train_df, test_df, rul_df = load_all_data()

    assert train_df.shape[1] == EXPECTED_COLUMNS_COUNT
    assert test_df.shape[1] == EXPECTED_COLUMNS_COUNT
    assert list(train_df.columns) == ALL_COLUMNS
    assert list(test_df.columns) == ALL_COLUMNS

    assert len(train_df) == 20631
    assert len(test_df) == 13096
    assert len(rul_df) == EXPECTED_RUL_COUNT


def test_engine_counts():
    """Verify 100 unique units in train, test, and RUL."""
    train_df, test_df, rul_df = load_all_data()

    assert train_df["unit_number"].nunique() == EXPECTED_TRAIN_UNITS
    assert test_df["unit_number"].nunique() == EXPECTED_TEST_UNITS
    assert len(rul_df) == EXPECTED_RUL_COUNT

    # Engine IDs must span 1 to 100
    assert sorted(train_df["unit_number"].unique().tolist()) == list(range(1, 101))
    assert sorted(test_df["unit_number"].unique().tolist()) == list(range(1, 101))


def test_training_rul_calculation():
    """Verify training RUL logic: RUL = max_cycle - cycle."""
    train_df = load_dataset_file(TRAIN_FILE)
    df_rul = add_training_rul(train_df)

    assert "RUL" in df_rul.columns
    # For every unit, at its maximum cycle, RUL must be exactly 0
    final_cycles = df_rul.groupby("unit_number").tail(1)
    assert (final_cycles["RUL"] == 0).all(), "Final cycle of training units must have RUL=0"

    # For unit 1, check cycle 1
    unit1 = df_rul[df_rul["unit_number"] == 1]
    unit1_max = unit1["cycle"].max()
    assert unit1.iloc[0]["RUL"] == unit1_max - 1


def test_validation_suite_passes(tmp_path):
    """Verify the validator passes with no failures."""
    train_df, test_df, rul_df = load_all_data()
    passed, report = validate_dataset(train_df, test_df, rul_df, save_reports=True, output_dir=tmp_path)

    assert passed is True
    assert report["status"] == "PASSED"
    assert len(report["failures"]) == 0
    assert report["missing_values"]["train_total_nans"] == 0
    assert report["missing_values"]["test_total_nans"] == 0
    assert report["duplicate_rows"]["train_full_row_duplicates"] == 0


def test_sensor_statistics_computation():
    """Verify sensor variance computation accurately flags constant sensors."""
    train_df = load_dataset_file(TRAIN_FILE)
    stats_df = compute_sensor_statistics(train_df)

    assert len(stats_df) == 21
    # In FD001 under sea-level conditions, sensors 1, 5, 10, 16, 18, 19 have zero variance
    constant_sensors = stats_df[stats_df["is_constant"]]["sensor"].tolist()
    expected_constants = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
    for s in expected_constants:
        assert s in constant_sensors, f"Expected {s} to be flagged as constant in FD001"
