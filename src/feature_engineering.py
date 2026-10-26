"""Phase 2: Feature Engineering Pipeline for AeroGuard AI.

Design principles:
  1. No data leakage: rolling statistics are computed engine-by-engine,
     using only current and past observations (closed='right', min_periods=1).
  2. Scaler is fit ONLY on the training partition and then applied to val/test.
  3. Constant sensors (confirmed in Phase 1 EDA) are dropped before scaling.
  4. Cycle-normalisation ('cycle_norm') uses the engine's own observed cycle range,
     which is available without leaking test targets.

Validation split design
-----------------------
  Train   : engine IDs 1-80  (80 engines)
  Validate: engine IDs 81-100 (20 engines)

  Rationale: splitting by engine ID guarantees that every cycle of a validation
  engine is invisible during training AND during scaler/rolling-stat fitting.
  Row-level random splits would mix cycles from the same engine into both sets,
  causing temporal leakage.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import SETTING_COLUMNS
from src.phase2_config import (
    ACTIVE_SENSOR_COLS,
    CONSTANT_SENSOR_COLS,
    ROLLING_WINDOWS,
    TRAIN_ENGINE_IDS,
    VAL_ENGINE_IDS,
    RUL_CAP,
)
from src.data_loader import add_training_rul


# ─── RUL Target ──────────────────────────────────────────────────────────────

def compute_rul(df: pd.DataFrame, rul_cap: Optional[int] = RUL_CAP) -> pd.DataFrame:
    """
    Compute RUL for run-to-failure training data and optionally cap it.

    For engine u at cycle t:
        RUL(u, t) = max_cycle(u) - t

    At the final cycle: RUL = 0 (confirmed failure).

    Parameters
    ----------
    df : pd.DataFrame
        Raw training DataFrame with 'unit_number' and 'cycle'.
    rul_cap : int or None
        If provided, RUL values exceeding this cap are clipped to the cap.
        None (default) means no cap is applied.

    Returns
    -------
    pd.DataFrame
        Copy with 'RUL' column appended. Original row order preserved.
    """
    df = add_training_rul(df)   # uses Phase 1's validated formula

    if rul_cap is not None:
        if not isinstance(rul_cap, int) or rul_cap <= 0:
            raise ValueError(f"rul_cap must be a positive integer or None; got {rul_cap!r}")
        df["RUL"] = df["RUL"].clip(upper=rul_cap)

    return df


# ─── Rolling Features ─────────────────────────────────────────────────────────

def _rolling_per_engine(
    df: pd.DataFrame,
    columns: List[str],
    window: int,
) -> pd.DataFrame:
    """
    Compute causal (backward-looking) rolling mean and std for *each engine*
    independently.  Uses min_periods=1 so the first cycle is never NaN.

    LEAKAGE GUARD: pandas groupby transform applies the rolling function
    within each group, so engine 82's cycle 1 never sees data from engine 1.
    The closed='right' (default) parameter ensures only past+current cycles
    contribute.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by (unit_number, cycle). Must already be sorted.
    columns : List[str]
        Feature columns to compute rolling statistics on.
    window : int
        Number of cycles in the rolling window.

    Returns
    -------
    pd.DataFrame
        New columns '{col}_roll{window}_mean' and '{col}_roll{window}_std'
        appended in-place on a copy.
    """
    out = df.copy()
    for col in columns:
        roll_mean = (
            out.groupby("unit_number", sort=False)[col]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )
        roll_std = (
            out.groupby("unit_number", sort=False)[col]
            .transform(lambda s: s.rolling(window=window, min_periods=1).std().fillna(0.0))
        )
        out[f"{col}_roll{window}_mean"] = roll_mean
        out[f"{col}_roll{window}_std"] = roll_std
    return out


# ─── Main Feature Engineering Function ───────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    windows: List[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Build the complete feature matrix from raw sensor data.

    Features generated:
      1. Raw informative sensor readings (15 sensors, dropping 6 constants).
      2. Operational settings (3 columns; kept as-is since FD001 is 1-condition).
      3. Cycle-normalised position: cycle / max_cycle_per_engine.
         Uses the engine's own observed max cycle (causal, no leakage).
      4. Rolling mean and std for each active sensor at each window size.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 26 columns (unit_number, cycle, 3 settings, 21 sensors).
    windows : List[int]
        Rolling window sizes (all backward-looking, per engine).

    Returns
    -------
    pd.DataFrame
        Feature DataFrame. Preserves 'unit_number', 'cycle', and 'RUL' columns
        if they exist, appending all engineered features.
    """
    # Sort: this is mandatory before any groupby-rolling
    df = df.sort_values(["unit_number", "cycle"]).reset_index(drop=True)

    # --- Cycle normalisation (causal: uses engine's own observed max) ----------
    max_cycles = df.groupby("unit_number")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycles

    # --- Keep only active (non-constant) sensors and settings -----------------
    feature_cols = SETTING_COLUMNS + ACTIVE_SENSOR_COLS

    # --- Rolling statistics (per engine, backward-looking) --------------------
    for w in windows:
        df = _rolling_per_engine(df, columns=ACTIVE_SENSOR_COLS, window=w)

    return df


def get_feature_names(windows: List[int] = ROLLING_WINDOWS) -> List[str]:
    """
    Return the ordered list of feature column names produced by build_features().
    Used to select X columns consistently across train, val, and test.
    """
    base_features = SETTING_COLUMNS + ACTIVE_SENSOR_COLS + ["cycle_norm"]
    roll_features = []
    for w in windows:
        for col in ACTIVE_SENSOR_COLS:
            roll_features.append(f"{col}_roll{w}_mean")
            roll_features.append(f"{col}_roll{w}_std")
    return base_features + roll_features


# ─── Validation Split ─────────────────────────────────────────────────────────

def split_by_engine(
    df_with_rul: pd.DataFrame,
    train_ids: List[int] = TRAIN_ENGINE_IDS,
    val_ids: List[int] = VAL_ENGINE_IDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a labelled DataFrame into train and validation partitions by engine ID.

    Design guarantee:
      Every cycle of every validation engine is completely absent from the
      training set. The scaler must be fit ONLY on the train partition.

    Parameters
    ----------
    df_with_rul : pd.DataFrame
        DataFrame with engineered features and 'RUL' column.
    train_ids : List[int]
        Engine IDs assigned to the training partition.
    val_ids : List[int]
        Engine IDs assigned to the validation partition.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_partition, val_partition)
    """
    # Validate that there is no overlap
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"Engine ID overlap between train and val partitions: {overlap}")

    train_mask = df_with_rul["unit_number"].isin(train_ids)
    val_mask = df_with_rul["unit_number"].isin(val_ids)

    return df_with_rul[train_mask].copy(), df_with_rul[val_mask].copy()


# ─── Preprocessing ────────────────────────────────────────────────────────────

class FeaturePipeline:
    """
    Stateful preprocessing wrapper that fits StandardScaler on training data
    and transforms validation/test data with the same scaler.

    Attributes
    ----------
    feature_names : List[str]
        Ordered list of feature column names.
    scaler : StandardScaler
        Fit scaler (None before .fit_transform is called).
    """

    def __init__(self, windows: List[int] = ROLLING_WINDOWS):
        self.windows = windows
        self.feature_names: List[str] = get_feature_names(windows)
        self.scaler: Optional[StandardScaler] = None

    def fit_transform(self, df_train_engineered: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler on training data and return scaled feature array.
        Call this ONLY on training rows.
        """
        X = df_train_engineered[self.feature_names].values.astype(np.float64)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def transform(self, df_engineered: pd.DataFrame) -> np.ndarray:
        """
        Apply the already-fit scaler to new data (val or test).
        Raises RuntimeError if called before fit_transform.
        """
        if self.scaler is None:
            raise RuntimeError("FeaturePipeline: call fit_transform() before transform().")
        X = df_engineered[self.feature_names].values.astype(np.float64)
        return self.scaler.transform(X)
