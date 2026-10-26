"""Phase 2: Model Training and Evaluation for AeroGuard AI.

Two-level evaluation design
---------------------------
1. CYCLE-LEVEL VALIDATION (val partition, engines 81-100)
   - Predict RUL for every cycle in the validation partition.
   - Used for model selection: compare Baseline vs RandomForest by MAE and RMSE.

2. ENGINE-LEVEL TEST EVALUATION (test_FD001.txt + RUL_FD001.txt)
   - For each test engine, take its FINAL OBSERVED cycle (last row in test file).
   - Predict RUL for that one cycle.
   - Compare against ground-truth RUL_FD001.txt values.
   - This partition is touched ONLY after model selection is complete.

These are conceptually distinct:
  - Cycle-level: row-wise regression over all training-set cycles.
  - Engine-level: single-point prediction per engine for remaining life.
"""

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.phase2_config import (
    MODELS_DIR,
    PHASE2_FIGURES_DIR,
    PHASE2_REPORTS_DIR,
    RF_PARAMS,
    BASELINE_MODEL_NAME,
    RF_MODEL_NAME,
    RUL_CAP,
)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MAE and RMSE, rounded to 4 decimal places."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4)}


# ─── Baseline Model ───────────────────────────────────────────────────────────

class MeanBaseline:
    """
    Predicts the global mean RUL of the training set for every query.
    Serves as the minimum-bar comparison for any learned model.
    """

    def __init__(self):
        self.mean_rul: Optional[float] = None

    def fit(self, y_train: np.ndarray) -> "MeanBaseline":
        self.mean_rul = float(np.mean(y_train))
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return an array of `n` identical predictions equal to the training mean."""
        if self.mean_rul is None:
            raise RuntimeError("MeanBaseline: call fit() before predict().")
        return np.full(n, self.mean_rul)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_baseline(y_train: np.ndarray) -> MeanBaseline:
    model = MeanBaseline()
    model.fit(y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    rf = RandomForestRegressor(**RF_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X_train, y_train)
    return rf


# ─── Validation Comparison ────────────────────────────────────────────────────

def validate_models(
    baseline: MeanBaseline,
    rf: RandomForestRegressor,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate both models on the cycle-level validation partition.
    Returns a dict keyed by model name with MAE and RMSE.
    """
    baseline_pred = baseline.predict(len(y_val))
    rf_pred = rf.predict(X_val)

    return {
        BASELINE_MODEL_NAME: compute_metrics(y_val, baseline_pred),
        RF_MODEL_NAME: compute_metrics(y_val, rf_pred),
    }


# ─── Test Evaluation ──────────────────────────────────────────────────────────

def build_test_features(
    df_test_engineered: pd.DataFrame,
    feature_pipeline,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the test evaluation set.

    For each test engine, take ONLY the final observed cycle (last row for that
    engine in test_FD001.txt).  The feature engineering pipeline (build_features)
    uses all cycles to compute rolling statistics, but evaluation uses only the
    last row — matching what RUL_FD001.txt measures.

    Parameters
    ----------
    df_test_engineered : pd.DataFrame
        Test set with engineered features (rolling stats computed over all cycles).
    feature_pipeline : FeaturePipeline
        Scaler fit on training data only.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_test_last_cycle, unit_numbers)
        X_test_last_cycle : scaled feature matrix, one row per engine.
        unit_numbers : corresponding engine IDs in the same order.
    """
    last_cycles = (
        df_test_engineered
        .sort_values(["unit_number", "cycle"])
        .groupby("unit_number")
        .last()
        .reset_index()
    )
    X_test = feature_pipeline.transform(last_cycles)
    unit_numbers = last_cycles["unit_number"].values
    return X_test, unit_numbers


def evaluate_on_test(
    rf: RandomForestRegressor,
    X_test: np.ndarray,
    unit_numbers: np.ndarray,
    rul_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run engine-level test evaluation.

    Parameters
    ----------
    rf : RandomForestRegressor
        Trained model selected from validation comparison.
    X_test : np.ndarray
        Scaled features for the final cycle of each test engine.
    unit_numbers : np.ndarray
        Engine IDs corresponding to each row of X_test.
    rul_df : pd.DataFrame
        Ground truth from RUL_FD001.txt with columns [unit_number, RUL].

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        (predictions_df, metrics_dict)
    """
    y_pred = rf.predict(X_test)

    pred_df = pd.DataFrame({
        "unit_number": unit_numbers,
        "predicted_RUL": y_pred,
    })
    merged = pred_df.merge(rul_df, on="unit_number", how="left")
    merged.rename(columns={"RUL": "true_RUL"}, inplace=True)
    merged["error"] = merged["predicted_RUL"] - merged["true_RUL"]
    merged["abs_error"] = merged["error"].abs()

    y_true = merged["true_RUL"].values
    metrics = compute_metrics(y_true, y_pred)
    return merged, metrics


# ─── Saving Artifacts ─────────────────────────────────────────────────────────

def save_model(model: Any, name: str, output_dir: Path = MODELS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def save_predictions(df: pd.DataFrame, name: str, output_dir: Path = PHASE2_REPORTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_predictions.csv"
    df.to_csv(path, index=False)
    return path


def save_metrics(metrics: Dict, name: str, output_dir: Path = PHASE2_REPORTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_validation_comparison(
    val_metrics: Dict[str, Dict[str, float]],
    output_dir: Path = PHASE2_FIGURES_DIR,
) -> Path:
    """Bar chart comparing MAE and RMSE of baseline vs RF on validation set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "validation_model_comparison.png"

    models = list(val_metrics.keys())
    mae_vals = [val_metrics[m]["MAE"] for m in models]
    rmse_vals = [val_metrics[m]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    labels = [m.replace("_", " ").title() for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, mae_vals, width, label="MAE", color="#1f77b4", edgecolor="black")
    bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE", color="#ff7f0e", edgecolor="black")

    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Error (cycles)", fontsize=11)
    ax.set_title("AeroGuard AI Phase 2 — Cycle-Level Validation: Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test_predictions(
    pred_df: pd.DataFrame,
    output_dir: Path = PHASE2_FIGURES_DIR,
) -> Path:
    """Scatter plot of predicted vs true RUL for each test engine."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_engine_rul_predictions.png"

    y_true = pred_df["true_RUL"].values
    y_pred = pred_df["predicted_RUL"].values
    max_val = max(y_true.max(), y_pred.max()) + 10

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true, y_pred, alpha=0.65, s=55, color="#2b5c8f", edgecolors="#1a3a5c", zorder=3)
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ax.text(
        0.05, 0.92,
        f"MAE  = {mae:.2f} cycles\nRMSE = {rmse:.2f} cycles",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.4")
    )

    ax.set_xlabel("True RUL (cycles)", fontsize=11)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=11)
    ax.set_title("AeroGuard AI — Test Set Engine-Level RUL Prediction", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(
    rf: RandomForestRegressor,
    feature_names: List[str],
    top_n: int = 25,
    output_dir: Path = PHASE2_FIGURES_DIR,
) -> Path:
    """Horizontal bar chart of top-N RF feature importances."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "feature_importances.png"

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    # Reverse for bottom-to-top display
    top_names = top_names[::-1]
    top_vals = top_vals[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top_names, top_vals, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
    ax.set_title(f"AeroGuard AI — Top {top_n} RF Feature Importances", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_distribution(
    pred_df: pd.DataFrame,
    output_dir: Path = PHASE2_FIGURES_DIR,
) -> Path:
    """Histogram of prediction errors (predicted - true) on the test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_error_distribution.png"

    errors = pred_df["error"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=20, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.axvline(errors.mean(), color="green", linestyle="-.", linewidth=1.5,
               label=f"Mean error = {errors.mean():.2f}")
    ax.set_xlabel("Prediction Error (predicted − true) [cycles]", fontsize=11)
    ax.set_ylabel("Count of Engines", fontsize=11)
    ax.set_title("AeroGuard AI — Test Set Prediction Error Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
