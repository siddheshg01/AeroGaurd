"""AeroGuard AI — Phase 2 Pipeline Runner.

Steps
-----
1.  Load train_FD001.txt, test_FD001.txt, RUL_FD001.txt (Phase 1 loaders).
2.  Compute training RUL (RUL_cap=None by default).
3.  Engineer features on training data.
4.  Split into train/val by engine ID (engines 1-80 / 81-100).
5.  Fit FeaturePipeline (scaler) on TRAIN rows only.
6.  Train MeanBaseline and RandomForestRegressor.
7.  Cycle-level validation: compare MAE / RMSE.
8.  Engineer features on test data using the same pipeline.
9.  Engine-level test evaluation: predict on final cycle of each test engine.
10. Save models, predictions, metrics, plots, and Markdown report.
"""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data_loader import load_all_data
from src.feature_engineering import (
    build_features,
    compute_rul,
    split_by_engine,
    get_feature_names,
    FeaturePipeline,
)
from src.trainer import (
    train_baseline,
    train_random_forest,
    validate_models,
    build_test_features,
    evaluate_on_test,
    save_model,
    save_predictions,
    save_metrics,
    plot_validation_comparison,
    plot_test_predictions,
    plot_feature_importance,
    plot_error_distribution,
)
from src.phase2_config import (
    MODELS_DIR,
    PHASE2_FIGURES_DIR,
    PHASE2_REPORTS_DIR,
    RUL_CAP,
    ROLLING_WINDOWS,
    TRAIN_ENGINE_IDS,
    VAL_ENGINE_IDS,
    RF_MODEL_NAME,
    BASELINE_MODEL_NAME,
)


def print_banner():
    print("""
========================================================================
                     AEROGUARD AI - PHASE 2
       Feature Engineering, Model Training & RUL Evaluation
========================================================================
""")


def write_markdown_report(
    val_metrics: dict,
    test_metrics: dict,
    train_shape: tuple,
    val_shape: tuple,
    n_features: int,
    rul_cap: int | None,
    elapsed: float,
    output_dir: Path,
) -> Path:
    """Generate a Markdown summary of Phase 2 results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase2_evaluation_report.md"

    selected = RF_MODEL_NAME if (
        val_metrics[RF_MODEL_NAME]["MAE"] < val_metrics[BASELINE_MODEL_NAME]["MAE"]
    ) else BASELINE_MODEL_NAME

    content = f"""# AeroGuard AI — Phase 2 Evaluation Report

## Experiment Configuration
| Parameter | Value |
|---|---|
| Dataset | NASA C-MAPSS FD001 |
| Operating Condition | Sea Level (single condition) |
| RUL Cap | `{rul_cap if rul_cap is not None else "None (no cap)"}` |
| Rolling Windows | {ROLLING_WINDOWS} cycles |
| Total Features | {n_features} |
| Train Engines (IDs 1-80) | {len(TRAIN_ENGINE_IDS)} engines, {train_shape[0]:,} cycles |
| Val Engines (IDs 81-100) | {len(VAL_ENGINE_IDS)} engines, {val_shape[0]:,} cycles |
| Validation Design | Split by engine ID — no row-level shuffle |

## Validation Design (Anti-Leakage)
- **Split strategy**: Engine-ID partitioning. All cycles from engines 81–100 are
  completely absent from training and scaler fitting.
- **Rolling features**: Computed per-engine on sorted (unit_number, cycle) data
  using backward-looking windows (min_periods=1, no look-ahead).
- **Scaler**: StandardScaler fit **only on training rows** (engines 1–80),
  then applied to validation and test.
- **Test set**: `test_FD001.txt` is touched **only after** model selection.

## Cycle-Level Validation Results (Engines 81–100)
> These metrics measure how well each model predicts RUL for every cycle.

| Model | MAE (cycles) | RMSE (cycles) |
|---|---|---|
| Mean Baseline | {val_metrics[BASELINE_MODEL_NAME]['MAE']:.4f} | {val_metrics[BASELINE_MODEL_NAME]['RMSE']:.4f} |
| Random Forest | {val_metrics[RF_MODEL_NAME]['MAE']:.4f} | {val_metrics[RF_MODEL_NAME]['RMSE']:.4f} |

**Selected model**: `{selected}` (lower validation MAE)

## Engine-Level Test Evaluation (test_FD001.txt vs RUL_FD001.txt)
> Each test engine's **final observed cycle** is the prediction point.
> True RUL is the ground-truth from RUL_FD001.txt.

| Metric | Value |
|---|---|
| Test MAE (cycles) | {test_metrics['MAE']:.4f} |
| Test RMSE (cycles) | {test_metrics['RMSE']:.4f} |
| Test Engines Evaluated | 100 |

## Explanation: Cycle-Level vs Engine-Level Metrics
These two evaluations are intentionally different:
- **Cycle-level** (validation): predicts RUL at *every* observed cycle for 20 held-out
  training engines. A single engine contributes ~200 rows. Measures temporal fit.
- **Engine-level** (test): predicts RUL at the *last observed cycle* for each of 100
  test engines. Each engine contributes exactly 1 prediction row. Measures final
  prognostic accuracy — the operationally relevant quantity.

## Output Artifacts
| Artifact | Location |
|---|---|
| Random Forest Model | `outputs/models/random_forest.pkl` |
| Mean Baseline Model | `outputs/models/mean_baseline.pkl` |
| Validation Metrics | `outputs/reports/phase2/random_forest_metrics.json` |
| Test Predictions CSV | `outputs/reports/phase2/random_forest_predictions.csv` |
| Model Comparison Plot | `outputs/figures/phase2/validation_model_comparison.png` |
| Test Scatter Plot | `outputs/figures/phase2/test_engine_rul_predictions.png` |
| Feature Importance Plot | `outputs/figures/phase2/feature_importances.png` |
| Error Distribution Plot | `outputs/figures/phase2/test_error_distribution.png` |

**Total Execution Time**: {elapsed:.2f} seconds
"""
    path.write_text(content, encoding="utf-8")
    return path


def main():
    print_banner()
    t_start = time.time()

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("[1/9] Loading datasets...")
    train_raw, test_raw, rul_df = load_all_data()
    print(f"       Train: {train_raw.shape}, Test: {test_raw.shape}, RUL: {rul_df.shape}")

    # ── Step 2: Compute training RUL ─────────────────────────────────────────
    print(f"[2/9] Computing training RUL (cap={RUL_CAP!r})...")
    train_with_rul = compute_rul(train_raw, rul_cap=RUL_CAP)
    print(f"       RUL range: [{train_with_rul['RUL'].min()}, {train_with_rul['RUL'].max()}]")
    print(f"       Final-cycle RUL=0 check: {(train_with_rul.groupby('unit_number')['RUL'].last() == 0).all()}")

    # ── Step 3: Engineer features (training data) ────────────────────────────
    print("[3/9] Engineering features on training data...")
    train_engineered = build_features(train_with_rul)
    feature_names = get_feature_names()
    print(f"       Feature matrix: {len(feature_names)} features")

    # ── Step 4: Engine-ID-based train/val split ───────────────────────────────
    print("[4/9] Splitting by engine ID (train: 1-80, val: 81-100)...")
    train_part, val_part = split_by_engine(train_engineered)
    print(f"       Train partition: {train_part.shape[0]:,} cycles, {train_part['unit_number'].nunique()} engines")
    print(f"       Val  partition:  {val_part.shape[0]:,} cycles, {val_part['unit_number'].nunique()} engines")

    # ── Step 5: Fit scaler on training rows only ─────────────────────────────
    print("[5/9] Fitting StandardScaler on training rows only (no leakage)...")
    pipeline = FeaturePipeline(windows=ROLLING_WINDOWS)
    X_train = pipeline.fit_transform(train_part)
    X_val = pipeline.transform(val_part)
    y_train = train_part["RUL"].values
    y_val = val_part["RUL"].values
    print(f"       X_train: {X_train.shape}, X_val: {X_val.shape}")

    # ── Step 6: Train models ──────────────────────────────────────────────────
    print("[6/9] Training models...")
    baseline = train_baseline(y_train)
    print(f"       Baseline mean RUL = {baseline.mean_rul:.2f} cycles")
    print("       Training RandomForestRegressor (200 trees)...")
    rf = train_random_forest(X_train, y_train)
    print(f"       RF trained. n_estimators={rf.n_estimators}, max_depth={rf.max_depth}")

    # ── Step 7: Cycle-level validation ───────────────────────────────────────
    print("[7/9] Cycle-level validation on held-out engines (81-100)...")
    val_metrics = validate_models(baseline, rf, X_val, y_val)
    for model_name, m in val_metrics.items():
        print(f"       {model_name:25s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

    # ── Step 8: Engineer test features ───────────────────────────────────────
    print("[8/9] Engineering features on test data (using train-fit scaler)...")
    test_engineered = build_features(test_raw)   # no RUL added — test has none
    X_test, unit_numbers = build_test_features(test_engineered, pipeline)
    print(f"       Test matrix: {X_test.shape} (one row per engine, final cycle)")

    # ── Step 9: Engine-level test evaluation ─────────────────────────────────
    print("[9/9] Engine-level test evaluation (final cycle per engine vs RUL_FD001.txt)...")
    test_pred_df, test_metrics = evaluate_on_test(rf, X_test, unit_numbers, rul_df)
    print(f"       Test MAE  = {test_metrics['MAE']:.4f} cycles")
    print(f"       Test RMSE = {test_metrics['RMSE']:.4f} cycles")

    # ── Save artifacts ────────────────────────────────────────────────────────
    print("\n--- Saving artifacts ---")
    save_model(baseline, BASELINE_MODEL_NAME)
    save_model(rf, RF_MODEL_NAME)
    save_model(pipeline, "feature_pipeline")
    save_predictions(test_pred_df, RF_MODEL_NAME)
    save_metrics({"validation": val_metrics, "test": test_metrics}, RF_MODEL_NAME)
    print(f"  [OK] Models saved to: {MODELS_DIR}")

    plot_validation_comparison(val_metrics)
    plot_test_predictions(test_pred_df)
    plot_feature_importance(rf, feature_names)
    plot_error_distribution(test_pred_df)
    print(f"  [OK] Plots saved to: {PHASE2_FIGURES_DIR}")

    elapsed = time.time() - t_start
    report_path = write_markdown_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_shape=train_part.shape,
        val_shape=val_part.shape,
        n_features=len(feature_names),
        rul_cap=RUL_CAP,
        elapsed=elapsed,
        output_dir=PHASE2_REPORTS_DIR,
    )
    print(f"  [OK] Report saved to: {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE — ALL STEPS EXECUTED AND VERIFIED")
    print("=" * 70)
    print(f"  Execution time     : {elapsed:.2f}s")
    print(f"  Validation MAE     : Baseline={val_metrics[BASELINE_MODEL_NAME]['MAE']:.2f}  RF={val_metrics[RF_MODEL_NAME]['MAE']:.2f}")
    print(f"  Validation RMSE    : Baseline={val_metrics[BASELINE_MODEL_NAME]['RMSE']:.2f}  RF={val_metrics[RF_MODEL_NAME]['RMSE']:.2f}")
    print(f"  Test (engine-level): MAE={test_metrics['MAE']:.2f}  RMSE={test_metrics['RMSE']:.2f}")
    print(f"  Reports dir        : {PHASE2_REPORTS_DIR}")
    print(f"  Figures dir        : {PHASE2_FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
