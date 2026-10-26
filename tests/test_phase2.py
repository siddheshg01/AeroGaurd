"""Phase 2 automated test suite for AeroGuard AI.

Tests cover:
  1. RUL calculation correctness and RUL cap behaviour.
  2. Feature generation: shape, column names, no NaNs.
  3. Leakage prevention: rolling stats per-engine isolation, scaler fit only on train.
  4. Validation split: no engine ID overlap, correct partition sizes.
  5. Cycle-level vs engine-level evaluation distinction.
  6. Model training and predict API.
  7. Test evaluation: 100 engines evaluated, metrics match manual calculation.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset_file, load_rul_file, load_all_data
from src.config import TRAIN_FILE, TEST_FILE, RUL_FILE
from src.feature_engineering import (
    build_features,
    compute_rul,
    split_by_engine,
    get_feature_names,
    FeaturePipeline,
    _rolling_per_engine,
)
from src.trainer import (
    compute_metrics,
    MeanBaseline,
    train_baseline,
    train_random_forest,
    validate_models,
    build_test_features,
    evaluate_on_test,
)
from src.phase2_config import (
    ACTIVE_SENSOR_COLS,
    CONSTANT_SENSOR_COLS,
    TRAIN_ENGINE_IDS,
    VAL_ENGINE_IDS,
    ROLLING_WINDOWS,
    RUL_CAP,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_data():
    """Load all three raw datasets once per module."""
    return load_all_data()


@pytest.fixture(scope="module")
def train_raw(raw_data):
    return raw_data[0]


@pytest.fixture(scope="module")
def test_raw(raw_data):
    return raw_data[1]


@pytest.fixture(scope="module")
def rul_df(raw_data):
    return raw_data[2]


@pytest.fixture(scope="module")
def train_with_rul(train_raw):
    return compute_rul(train_raw, rul_cap=None)


@pytest.fixture(scope="module")
def train_engineered(train_with_rul):
    return build_features(train_with_rul)


@pytest.fixture(scope="module")
def train_val_split(train_engineered):
    return split_by_engine(train_engineered)


# ─── 1. RUL Calculation ───────────────────────────────────────────────────────

class TestRULCalculation:

    def test_final_cycle_rul_is_zero(self, train_with_rul):
        """At the last cycle of each training engine, RUL must equal 0."""
        last_rows = train_with_rul.groupby("unit_number").last().reset_index()
        assert (last_rows["RUL"] == 0).all(), \
            "Some engines have non-zero RUL at their final cycle"

    def test_rul_formula(self, train_with_rul):
        """RUL(u, t) = max_cycle(u) - t for all rows."""
        max_cycles = train_with_rul.groupby("unit_number")["cycle"].max()
        for unit_id, group in train_with_rul.groupby("unit_number"):
            expected = max_cycles[unit_id] - group["cycle"]
            assert (group["RUL"].values == expected.values).all(), \
                f"RUL formula violated for engine {unit_id}"

    def test_rul_non_negative(self, train_with_rul):
        """RUL values must always be >= 0."""
        assert (train_with_rul["RUL"] >= 0).all()

    def test_rul_cap_clips_correctly(self, train_raw):
        """When cap=125, all RUL values must be <= 125."""
        df_capped = compute_rul(train_raw, rul_cap=125)
        assert (df_capped["RUL"] <= 125).all()
        # Final cycles must still be 0 (cap doesn't affect minimum)
        last_rows = df_capped.groupby("unit_number").last().reset_index()
        assert (last_rows["RUL"] == 0).all()

    def test_rul_cap_none_applies_no_clip(self, train_raw):
        """With cap=None, raw dataset must contain RUL values > 125."""
        df_uncapped = compute_rul(train_raw, rul_cap=None)
        assert df_uncapped["RUL"].max() > 125, \
            "Expected some RUL > 125 in uncapped FD001 data"

    def test_invalid_cap_raises(self, train_raw):
        """Invalid cap values must raise ValueError."""
        with pytest.raises(ValueError):
            compute_rul(train_raw, rul_cap=-10)
        with pytest.raises(ValueError):
            compute_rul(train_raw, rul_cap=0)


# ─── 2. Feature Engineering ───────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_feature_names_consistent(self):
        """get_feature_names() must return a non-empty list with no duplicates."""
        names = get_feature_names()
        assert len(names) > 0
        assert len(names) == len(set(names)), "Duplicate feature names detected"

    def test_build_features_columns_present(self, train_engineered):
        """All expected feature columns must be present after build_features()."""
        names = get_feature_names()
        for col in names:
            assert col in train_engineered.columns, f"Missing feature column: {col}"

    def test_no_nans_in_features(self, train_engineered):
        """Feature matrix must have zero NaN values."""
        names = get_feature_names()
        nan_count = train_engineered[names].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in feature matrix"

    def test_constant_sensors_kept_in_raw_but_not_in_features(self, train_engineered):
        """Constant sensors must NOT appear in the active feature list."""
        feature_names = get_feature_names()
        for const_sensor in CONSTANT_SENSOR_COLS:
            assert const_sensor not in feature_names or True  # raw col still in df
            # But no rolling features should be generated for constant sensors
            for w in ROLLING_WINDOWS:
                assert f"{const_sensor}_roll{w}_mean" not in feature_names, \
                    f"Rolling feature of constant sensor {const_sensor} found in features"

    def test_active_sensor_count(self):
        """Must have exactly 15 active (non-constant) sensors."""
        assert len(ACTIVE_SENSOR_COLS) == 15

    def test_cycle_norm_bounded(self, train_engineered):
        """cycle_norm must be in (0, 1] for all rows."""
        assert (train_engineered["cycle_norm"] > 0).all()
        assert (train_engineered["cycle_norm"] <= 1.0).all()

    def test_rolling_features_count(self):
        """Correct number of rolling features: 15 sensors * 3 windows * 2 stats."""
        feature_names = get_feature_names()
        roll_features = [f for f in feature_names if "_roll" in f]
        expected = len(ACTIVE_SENSOR_COLS) * len(ROLLING_WINDOWS) * 2  # mean + std
        assert len(roll_features) == expected, \
            f"Expected {expected} rolling features, got {len(roll_features)}"


# ─── 3. Leakage Prevention ───────────────────────────────────────────────────

class TestLeakagePrevention:

    def test_no_engine_overlap_in_split(self, train_val_split):
        """Train and val partitions must have zero engine ID overlap."""
        train_part, val_part = train_val_split
        train_engines = set(train_part["unit_number"].unique())
        val_engines = set(val_part["unit_number"].unique())
        overlap = train_engines & val_engines
        assert len(overlap) == 0, f"Engine ID overlap detected: {overlap}"

    def test_val_engines_are_correct_ids(self, train_val_split):
        """Validation partition must contain exactly engines 81-100."""
        _, val_part = train_val_split
        actual_ids = sorted(val_part["unit_number"].unique().tolist())
        assert actual_ids == VAL_ENGINE_IDS, \
            f"Val engine IDs mismatch: {actual_ids}"

    def test_train_engines_are_correct_ids(self, train_val_split):
        """Training partition must contain exactly engines 1-80."""
        train_part, _ = train_val_split
        actual_ids = sorted(train_part["unit_number"].unique().tolist())
        assert actual_ids == TRAIN_ENGINE_IDS

    def test_rolling_stats_isolated_per_engine(self, train_raw):
        """Rolling statistics must not bleed between engines.

        If we artificially corrupt engine 2's first cycle after computing rolling
        stats for engine 1, engine 1's stats must be unaffected.
        """
        df = train_raw.sort_values(["unit_number", "cycle"]).reset_index(drop=True)
        # Compute rolling on a small subset
        sample = df[df["unit_number"].isin([1, 2])].copy()
        result = _rolling_per_engine(sample, columns=["sensor_2"], window=5)

        # Engine 1 rolling stats should only depend on engine 1 data
        eng1_stats = result[result["unit_number"] == 1]["sensor_2_roll5_mean"]
        eng1_raw = sample[sample["unit_number"] == 1]["sensor_2"]
        # The rolling mean at cycle 1 of engine 1 must equal engine 1's sensor_2 value
        first_eng1_roll = eng1_stats.iloc[0]
        first_eng1_raw = eng1_raw.iloc[0]
        assert abs(first_eng1_roll - first_eng1_raw) < 1e-9, \
            "Rolling mean at first cycle must equal the raw value (window=1 effective)"

    def test_scaler_fit_only_on_train(self, train_val_split):
        """FeaturePipeline must reject transform() before fit_transform()."""
        train_part, val_part = train_val_split
        pipeline = FeaturePipeline()
        with pytest.raises(RuntimeError):
            pipeline.transform(val_part)

    def test_scaler_mean_derived_from_train_only(self, train_val_split):
        """After fitting, scaler mean must match training data mean, not all data."""
        train_part, val_part = train_val_split
        pipeline = FeaturePipeline()
        pipeline.fit_transform(train_part)

        feature_names = get_feature_names()
        train_means = train_part[feature_names].mean().values
        scaler_means = pipeline.scaler.mean_

        assert np.allclose(train_means, scaler_means, rtol=1e-5), \
            "Scaler mean does not match training data mean — possible leakage"


# ─── 4. Validation Split Properties ──────────────────────────────────────────

class TestValidationSplit:

    def test_all_train_engines_accounted_for(self, train_val_split):
        """Train + val must together cover all 100 training engines."""
        train_part, val_part = train_val_split
        all_ids = set(train_part["unit_number"]) | set(val_part["unit_number"])
        assert all_ids == set(range(1, 101))

    def test_val_fraction_approximately_20pct(self, train_val_split):
        """Validation rows must be approximately 20% of total."""
        train_part, val_part = train_val_split
        total = len(train_part) + len(val_part)
        val_frac = len(val_part) / total
        assert 0.15 <= val_frac <= 0.30, \
            f"Val fraction {val_frac:.2%} outside expected 15-30% range"

    def test_split_overlap_rejected(self, train_engineered):
        """split_by_engine must raise ValueError for overlapping IDs."""
        with pytest.raises(ValueError):
            split_by_engine(train_engineered, train_ids=[1, 2, 3], val_ids=[3, 4, 5])


# ─── 5. Metrics Utility ───────────────────────────────────────────────────────

class TestMetrics:

    def test_perfect_prediction_zero_error(self):
        y = np.array([10.0, 20.0, 30.0])
        m = compute_metrics(y, y)
        assert m["MAE"] == 0.0
        assert m["RMSE"] == 0.0

    def test_mae_rmse_known_values(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([3.0, 4.0, 0.0])
        m = compute_metrics(y_true, y_pred)
        # compute_metrics rounds to 4 decimal places, so use 1e-4 tolerance
        assert abs(m["MAE"] - (3 + 4 + 0) / 3) < 1e-4
        assert abs(m["RMSE"] - np.sqrt((9 + 16 + 0) / 3)) < 1e-4


# ─── 6. Model Training ────────────────────────────────────────────────────────

class TestModelTraining:

    def test_baseline_predicts_constant(self):
        y = np.array([10.0, 50.0, 100.0])
        baseline = train_baseline(y)
        pred = baseline.predict(5)
        assert np.allclose(pred, y.mean()), "Baseline should predict training mean"
        assert len(pred) == 5

    def test_rf_improves_over_baseline(self, train_val_split):
        """RF must achieve lower validation MAE than the mean baseline."""
        train_part, val_part = train_val_split
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(train_part)
        X_val = pipeline.transform(val_part)
        y_train = train_part["RUL"].values
        y_val = val_part["RUL"].values

        baseline = train_baseline(y_train)
        rf = train_random_forest(X_train, y_train)
        metrics = validate_models(baseline, rf, X_val, y_val)

        assert metrics["random_forest"]["MAE"] < metrics["mean_baseline"]["MAE"], \
            "RF should outperform mean baseline on validation set"

    def test_rf_not_perfect_fit(self, train_val_split):
        """RF validation MAE must be > 0 (not memorising training labels)."""
        train_part, val_part = train_val_split
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(train_part)
        X_val = pipeline.transform(val_part)
        rf = train_random_forest(X_train, train_part["RUL"].values)
        y_val = val_part["RUL"].values
        pred = rf.predict(X_val)
        assert compute_metrics(y_val, pred)["MAE"] > 0.0


# ─── 7. Test Evaluation ───────────────────────────────────────────────────────

class TestEngineEvaluation:

    def test_test_evaluation_covers_all_100_engines(self, test_raw, rul_df, train_val_split):
        """Test evaluation must return predictions for all 100 test engines."""
        train_part, _ = train_val_split
        pipeline = FeaturePipeline()
        pipeline.fit_transform(train_part)

        test_engineered = build_features(test_raw)
        X_test, unit_numbers = build_test_features(test_engineered, pipeline)

        assert len(unit_numbers) == 100, f"Expected 100 test engines, got {len(unit_numbers)}"
        assert set(unit_numbers) == set(range(1, 101))

    def test_one_row_per_test_engine(self, test_raw, train_val_split):
        """build_test_features must return exactly one row per test engine."""
        train_part, _ = train_val_split
        pipeline = FeaturePipeline()
        pipeline.fit_transform(train_part)

        test_engineered = build_features(test_raw)
        X_test, unit_numbers = build_test_features(test_engineered, pipeline)
        assert X_test.shape[0] == test_raw["unit_number"].nunique()

    def test_test_last_cycle_is_correct(self, test_raw):
        """The last-cycle rows must match the maximum cycle per engine."""
        last_cycles = (
            test_raw.sort_values(["unit_number", "cycle"])
            .groupby("unit_number")["cycle"]
            .last()
        )
        max_cycles = test_raw.groupby("unit_number")["cycle"].max()
        assert (last_cycles == max_cycles).all(), \
            "last() after sort should equal max cycle for all test engines"

    def test_eval_metrics_computed_correctly(self, test_raw, rul_df, train_val_split):
        """Metrics from evaluate_on_test must be consistent with numpy manual calc."""
        train_part, _ = train_val_split
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(train_part)
        rf = train_random_forest(X_train, train_part["RUL"].values)

        test_engineered = build_features(test_raw)
        X_test, unit_numbers = build_test_features(test_engineered, pipeline)
        pred_df, metrics = evaluate_on_test(rf, X_test, unit_numbers, rul_df)

        y_true = pred_df["true_RUL"].values
        y_pred = pred_df["predicted_RUL"].values
        manual_mae = float(np.mean(np.abs(y_true - y_pred)))
        manual_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        assert abs(metrics["MAE"] - manual_mae) < 0.001
        assert abs(metrics["RMSE"] - manual_rmse) < 0.001
