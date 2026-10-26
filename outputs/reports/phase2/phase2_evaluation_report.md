# AeroGuard AI — Phase 2 Evaluation Report

## Experiment Configuration
| Parameter | Value |
|---|---|
| Dataset | NASA C-MAPSS FD001 |
| Operating Condition | Sea Level (single condition) |
| RUL Cap | `None (no cap)` |
| Rolling Windows | [5, 10, 20] cycles |
| Total Features | 109 |
| Train Engines (IDs 1-80) | 80 engines, 16,138 cycles |
| Val Engines (IDs 81-100) | 20 engines, 4,493 cycles |
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
| Mean Baseline | 64.0893 | 78.9251 |
| Random Forest | 27.1904 | 39.2113 |

**Selected model**: `random_forest` (lower validation MAE)

## Engine-Level Test Evaluation (test_FD001.txt vs RUL_FD001.txt)
> Each test engine's **final observed cycle** is the prediction point.
> True RUL is the ground-truth from RUL_FD001.txt.

| Metric | Value |
|---|---|
| Test MAE (cycles) | 47.8890 |
| Test RMSE (cycles) | 56.6181 |
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

**Total Execution Time**: 20.23 seconds
