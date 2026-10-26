"""Business logic service layer for AeroGuard AI Streamlit Dashboard (Phase 4).

Keeps UI presentation strictly separated from core business logic, feature engineering,
predictive modeling, and retrieval pipelines.
"""

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ALL_COLUMNS,
    DATA_DIR,
    INDEX_COLUMNS,
    OUTPUT_DIR,
    RUL_FILE,
    SENSOR_COLUMNS,
    SENSOR_DESCRIPTIONS,
    SETTING_COLUMNS,
    TEST_FILE,
    TRAIN_FILE,
    REPORTS_DIR,
)
from src.data_loader import load_dataset_file, load_rul_file
from src.feature_engineering import FeaturePipeline, build_features
from src.phase2_config import (
    ACTIVE_SENSOR_COLS,
    CONSTANT_SENSOR_COLS,
    MODELS_DIR,
    PHASE2_REPORTS_DIR,
)
from src.phase3_config import (
    DEFAULT_DOC_TITLE,
    EMBEDDING_MODEL_NAME,
    EXTRACTED_DATA_PATH,
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    RAG_OUTPUT_DIR,
    RESEARCH_DOC_PATH,
    SIMILARITY_THRESHOLD,
    SYSTEM_DISCLAIMER,
)
from src.rag_retriever import EmbeddingEngine, RAGRetriever
from src.research_assistant import EngineContext, ResearchAssistant


# ─── Artifact Locations ───────────────────────────────────────────────────────
FEATURE_PIPELINE_PATH: Path = MODELS_DIR / "feature_pipeline.pkl"
RANDOM_FOREST_PATH: Path = MODELS_DIR / "random_forest.pkl"
PHASE2_METRICS_PATH: Path = PHASE2_REPORTS_DIR / "random_forest_metrics.json"
PHASE2_PREDICTIONS_PATH: Path = PHASE2_REPORTS_DIR / "random_forest_predictions.csv"


def format_metric(value: Any) -> str:
    """Render a saved numeric metric or an explicit unavailable marker."""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "Unavailable"


# ─── 1. Overview Statistics ───────────────────────────────────────────────────

def get_overview_statistics() -> dict[str, Any]:
    """Compile dataset dimensions, engine counts, Phase 2 metrics, and RAG knowledge base statistics.

    Returns:
        Structured dictionary of verified overview metrics.
    """
    # Load dataset counts
    train_df = load_dataset_file(TRAIN_FILE)
    test_df = load_dataset_file(TEST_FILE)
    rul_df = load_rul_file(RUL_FILE)
    validation_path = REPORTS_DIR / "data_validation_report.json"
    validation_report = {}
    if validation_path.exists():
        with open(validation_path, "r", encoding="utf-8") as f:
            validation_report = json.load(f)

    train_rows = len(train_df)
    train_units = train_df["unit_number"].nunique()
    test_rows = len(test_df)
    test_units = test_df["unit_number"].nunique()

    # Load Phase 2 test evaluation metrics
    phase2_metrics = {}
    if PHASE2_METRICS_PATH.exists():
        with open(PHASE2_METRICS_PATH, "r", encoding="utf-8") as f:
            phase2_metrics = json.load(f)

    test_metrics = phase2_metrics.get("test", {})
    val_metrics = phase2_metrics.get("validation", {})
    rf_validation = val_metrics.get("random_forest", {})
    baseline_validation = val_metrics.get("mean_baseline", {})
    test_mae = test_metrics.get("MAE")
    test_rmse = test_metrics.get("RMSE")
    val_rf_mae = rf_validation.get("MAE")
    val_rf_rmse = rf_validation.get("RMSE")
    val_baseline_mae = baseline_validation.get("MAE")
    val_baseline_rmse = baseline_validation.get("RMSE")

    # Load RAG metadata
    rag_info = {
        "doc_title": DEFAULT_DOC_TITLE,
        "doc_file": RESEARCH_DOC_PATH.name,
        "total_pages": None,
        "total_chunks": None,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": 384,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    if INDEX_METADATA_PATH.exists():
        try:
            with open(INDEX_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                rag_info["total_chunks"] = meta.get("total_chunks")
                rag_info["embedding_dim"] = meta.get("dimension", 384)
                chunks = meta.get("chunks", [])
                pages = {chunk.get("page_number") for chunk in chunks if chunk.get("page_number") is not None}
                rag_info["total_pages"] = len(pages) if pages else None
        except Exception:
            pass

    return {
        "dataset": {
            "name": "NASA C-MAPSS FD001",
            "train_rows": train_rows,
            "train_engines": train_units,
            "test_rows": test_rows,
            "test_engines": test_units,
            "total_sensors": len(SENSOR_COLUMNS),
            "active_sensors": len(ACTIVE_SENSOR_COLS),
            "constant_sensors": len(CONSTANT_SENSOR_COLS),
            "validation_status": validation_report.get("status", "unavailable"),
            "validation_failures": validation_report.get("failures", []),
        },
        "phase2_models": {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "val_rf_mae": val_rf_mae,
            "val_rf_rmse": val_rf_rmse,
            "val_baseline_mae": val_baseline_mae,
            "val_baseline_rmse": val_baseline_rmse,
            "improvement_pct": round((1.0 - val_rf_mae / val_baseline_mae) * 100, 1)
            if val_baseline_mae not in (None, 0)
            else None,
        },
        "rag_kb": rag_info,
        "disclaimer": SYSTEM_DISCLAIMER,
    }


# ─── 2. Sensor Analytics Services ─────────────────────────────────────────────

def get_sensor_options() -> list[dict[str, Any]]:
    """Return list of all 21 sensors with metadata descriptions and variance status."""
    options = []
    for s_id in SENSOR_COLUMNS:
        is_const = s_id in CONSTANT_SENSOR_COLS
        desc = SENSOR_DESCRIPTIONS.get(s_id, s_id)
        options.append(
            {
                "sensor_id": s_id,
                "label": f"{s_id} - {desc}" + (" [Constant in FD001]" if is_const else ""),
                "description": desc,
                "is_constant": is_const,
            }
        )
    return options


def load_engine_telemetry(
    dataset_type: str = "train",
    unit_number: int = 1,
) -> pd.DataFrame:
    """Load observed cycle trajectory for a selected engine from train or test dataset.

    Args:
        dataset_type: 'train' (run-to-failure) or 'test' (truncated).
        unit_number: Engine unit ID (1 to 100).

    Returns:
        Filtered DataFrame sorted by cycle.
    """
    file_path = TRAIN_FILE if dataset_type.lower() == "train" else TEST_FILE
    df = load_dataset_file(file_path)
    engine_df = df[df["unit_number"] == unit_number].sort_values("cycle").reset_index(drop=True)
    return engine_df


def compute_sensor_summary(engine_df: pd.DataFrame, sensor_name: str) -> dict[str, float]:
    """Calculate statistical properties of an engine's sensor trajectory over cycles."""
    if sensor_name not in engine_df.columns or len(engine_df) == 0:
        return {}

    series = engine_df[sensor_name]
    initial_val = series.iloc[: min(5, len(series))].mean()
    final_val = series.iloc[-min(5, len(series)) :].mean()
    delta = final_val - initial_val

    return {
        "initial_mean": round(float(initial_val), 3),
        "final_mean": round(float(final_val), 3),
        "delta": round(float(delta), 3),
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "mean": round(float(series.mean()), 3),
        "std": round(float(series.std()), 3),
        "total_cycles": int(len(series)),
    }


# ─── 3. RUL Prediction Services ───────────────────────────────────────────────

def check_model_artifacts_available() -> tuple[bool, str]:
    """Check that Phase 2 feature pipeline and trained model pickle files exist.

    Returns:
        (is_available, error_message)
    """
    if not FEATURE_PIPELINE_PATH.exists():
        return False, f"Missing feature pipeline artifact: {FEATURE_PIPELINE_PATH}"
    if not RANDOM_FOREST_PATH.exists():
        return False, f"Missing trained model artifact: {RANDOM_FOREST_PATH}"
    if not TEST_FILE.exists():
        return False, f"Missing test dataset file: {TEST_FILE}"
    if not RUL_FILE.exists():
        return False, f"Missing ground truth RUL file: {RUL_FILE}"
    return True, ""


@lru_cache(maxsize=1)
def _load_phase2_artifacts() -> tuple[FeaturePipeline, Any]:
    """Load the exact Phase 2 pipeline/model once per process."""
    with open(FEATURE_PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(RANDOM_FOREST_PATH, "rb") as f:
        model = pickle.load(f)
    return pipeline, model


def predict_test_engine_rul(unit_number: int) -> dict[str, Any]:
    """Predict RUL for a specific test engine at its final observed cycle using Phase 2 artifacts.

    Strictly preserves the exact feature engineering pipeline and model weights trained in Phase 2.
    Never fabricates predictions.

    Args:
        unit_number: Test engine ID (1 to 100).

    Returns:
        Dictionary with prediction, ground truth, error, and health classification.

    Raises:
        FileNotFoundError: If model or pipeline artifacts are missing.
        ValueError: If unit_number is not in the test dataset.
    """
    available, err_msg = check_model_artifacts_available()
    if not available:
        raise FileNotFoundError(err_msg)

    # 1. Load artifacts
    pipeline, model = _load_phase2_artifacts()

    # 2. Load test data and ground truth
    test_raw = load_dataset_file(TEST_FILE)
    rul_raw = load_rul_file(RUL_FILE)

    if unit_number not in test_raw["unit_number"].unique():
        raise ValueError(f"Engine unit_number {unit_number} not found in test dataset.")

    # 3. Filter for this engine and build full engineered features
    # (Note: rolling features need the engine's historical observations to calculate window stats)
    engine_raw = test_raw[test_raw["unit_number"] == unit_number].sort_values("cycle")
    df_engineered = build_features(engine_raw)

    # Extract final observed cycle row
    final_cycle_row = df_engineered.iloc[[-1]]
    last_cycle = int(final_cycle_row["cycle"].values[0])

    # Transform features with saved scaler pipeline
    X_scaled = pipeline.transform(final_cycle_row)

    # 4. Predict RUL using exact Phase 2 RandomForest
    # The serialized estimator may request process workers (n_jobs=-1), which
    # can be denied in restricted Windows environments. Single-worker inference
    # uses the same saved estimator and produces the same regression output.
    original_n_jobs = getattr(model, "n_jobs", None)
    try:
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        pred_rul = float(model.predict(X_scaled)[0])
    finally:
        if hasattr(model, "n_jobs"):
            model.n_jobs = original_n_jobs

    # 5. Extract ground truth RUL
    true_row = rul_raw[rul_raw["unit_number"] == unit_number]
    true_rul = float(true_row["RUL"].values[0]) if len(true_row) > 0 else None

    error = round(pred_rul - true_rul, 2) if true_rul is not None else None
    abs_error = round(abs(error), 2) if error is not None else None

    if pred_rul < 30:
        health_status, health_badge = "Critical Degradation", "Critical (<30 cycles)"
    elif pred_rul < 80:
        health_status, health_badge = "Moderate Degradation", "Moderate (30–80 cycles)"
    else:
        health_status, health_badge = "Nominal / Early Life", "Nominal (>80 cycles)"

    return {
        "unit_number": unit_number,
        "last_observed_cycle": last_cycle,
        "predicted_rul": round(pred_rul, 1),
        "true_rul": round(true_rul, 1) if true_rul is not None else None,
        "error": error,
        "abs_error": abs_error,
        "health_status": health_status,
        "health_badge": health_badge,
        "disclaimer": SYSTEM_DISCLAIMER,
    }


# ─── 4. Research Assistant Services ───────────────────────────────────────────

def get_research_assistant_instance(
    shared_engine: EmbeddingEngine | None = None,
) -> ResearchAssistant:
    """Instantiate and load the Phase 3 Research Assistant with cached index.

    Args:
        shared_engine: Optional existing EmbeddingEngine to avoid re-instantiation.

    Returns:
        Configured ResearchAssistant instance.
    """
    if not FAISS_INDEX_PATH.exists() or not INDEX_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index or metadata not found at {FAISS_INDEX_PATH}. "
            "Please run 'python src/run_phase3.py' first."
        )

    retriever = RAGRetriever.load(
        index_path=FAISS_INDEX_PATH,
        metadata_path=INDEX_METADATA_PATH,
        embedding_model=shared_engine,
    )
    return ResearchAssistant(
        retriever=retriever,
        similarity_threshold=SIMILARITY_THRESHOLD,
        disclaimer=SYSTEM_DISCLAIMER,
    )


def query_research_assistant(
    assistant: ResearchAssistant,
    query: str,
    top_k: int = 3,
    engine_context_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute query against Phase 3 research assistant with strict citation and safety checks."""
    context_obj = (
        EngineContext.from_dict(engine_context_dict) if engine_context_dict else None
    )
    return assistant.ask(query=query, top_k=top_k, engine_context=context_obj)


# ─── 5. Model & Governance Information ────────────────────────────────────────

def get_governance_details() -> dict[str, Any]:
    """Compile dataset provenance, ML methodology, evaluation benchmarks, and governance rules."""
    metrics = {}
    if PHASE2_METRICS_PATH.exists():
        with open(PHASE2_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    test = metrics.get("test", {})
    validation = metrics.get("validation", {})
    baseline = validation.get("mean_baseline", {})
    forest = validation.get("random_forest", {})
    return {
        "dataset_provenance": {
            "source": "NASA Ames Prognostics Center of Excellence (PCoE)",
            "simulator": "Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)",
            "subdataset": "FD001",
            "operating_regime": "Sea level (single operating condition, Mach 0, Alt 0)",
            "fault_mode": "High Pressure Compressor (HPC) degradation (flow & efficiency loss)",
            "synthetic_nature": "Benchmark numerical simulation generated for PHM competition research",
        },
        "model_methodology": {
            "algorithm": "RandomForestRegressor (scikit-learn)",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_leaf": 4,
                "max_features": 0.6,
                "random_state": 42,
            },
            "preprocessing": "StandardScaler fit exclusively on training engines 1–80 (zero leakage)",
            "feature_count": 109,
            "feature_breakdown": "15 active sensors + 3 settings + 1 cycle norm + 45 rolling means + 45 rolling stds",
            "validation_split": "Engine-ID partition (Engines 1–80 train, Engines 81–100 validate)",
        },
        "performance_benchmark": {
            "cycle_val_baseline_mae": baseline.get("MAE"),
            "cycle_val_baseline_rmse": baseline.get("RMSE"),
            "cycle_val_rf_mae": forest.get("MAE"),
            "cycle_val_rf_rmse": forest.get("RMSE"),
            "engine_test_mae": test.get("MAE"),
            "engine_test_rmse": test.get("RMSE"),
        },
        "known_limitations": [
            "Single operating condition: FD001 models only sea-level cruise; does not account for multi-regime flight envelopes (FD002/FD004).",
            "Single fault mode: Simulates only HPC deterioration, ignoring fan blade erosion, combustor cracks, or turbine creep.",
            "Cut-off point variance: Test set prediction error is higher at early truncation cycles where degradation has not yet distinctively manifested.",
            "Non-operational simulator: C-MAPSS is an academic research simulator, not an FAA/EASA certified Flight Data Monitoring (FDM) system.",
        ],
        "regulatory_disclaimer": SYSTEM_DISCLAIMER,
    }
