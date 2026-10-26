"""Phase 2 configuration: feature engineering, model and experiment settings."""

from pathlib import Path
from src.config import OUTPUT_DIR, SENSOR_COLUMNS

# ─── Output paths ────────────────────────────────────────────────────────────
MODELS_DIR = OUTPUT_DIR / "models"
PHASE2_FIGURES_DIR = OUTPUT_DIR / "figures" / "phase2"
PHASE2_REPORTS_DIR = OUTPUT_DIR / "reports" / "phase2"

# ─── RUL target ──────────────────────────────────────────────────────────────
# RUL cap: None means no cap (piece-wise linear target is NOT applied by default).
# Set to a positive integer (e.g. 125) to clip RUL values above the cap.
# Phase 2 uses no cap so that the baseline results reflect the raw dataset.
RUL_CAP: int | None = None

# ─── Sensor selection ────────────────────────────────────────────────────────
# Sensors confirmed zero-variance under FD001 sea-level single-condition regime.
# Source: Phase 1 sensor_summary_statistics.csv -- these six have std < 1e-6.
CONSTANT_SENSOR_COLS = [
    "sensor_1",   # Fan inlet temperature T2 -- constant 518.67 degR
    "sensor_5",   # Fan inlet pressure P2 -- constant 14.62 psia
    "sensor_10",  # Engine pressure ratio epr -- constant 1.3
    "sensor_16",  # Burner fuel-air ratio farB -- constant 0.03
    "sensor_18",  # Demanded fan speed Nf_dmd -- constant 2388.0 rpm
    "sensor_19",  # Demanded corrected fan speed PCNfR_dmd -- constant 100.0 rpm
]

# Informative sensors (21 total minus the 6 constant = 15 variable sensors)
ACTIVE_SENSOR_COLS = [s for s in SENSOR_COLUMNS if s not in CONSTANT_SENSOR_COLS]

# ─── Validation split ────────────────────────────────────────────────────────
# Engines are split by ID to prevent leakage:
#   - Train partition  : engine IDs  1-80  (80 engines, ~80% of rows)
#   - Validate partition: engine IDs 81-100 (20 engines, ~20% of rows)
# No row-level shuffling or random splits are used.
TRAIN_ENGINE_IDS = list(range(1, 81))
VAL_ENGINE_IDS = list(range(81, 101))

# ─── Rolling window settings ─────────────────────────────────────────────────
# Causal rolling: only uses past/current observations (no look-ahead).
ROLLING_WINDOWS = [5, 10, 20]  # window sizes in cycles

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ─── Model configs ───────────────────────────────────────────────────────────
BASELINE_MODEL_NAME = "mean_baseline"
RF_MODEL_NAME = "random_forest"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 4,
    "max_features": 0.6,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
