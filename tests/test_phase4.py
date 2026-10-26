"""Unit and integration test suite for AeroGuard AI Phase 4 Streamlit Dashboard services."""

from pathlib import Path
import pytest
import pandas as pd

from src.config import SENSOR_COLUMNS
from src.dashboard_service import (
    check_model_artifacts_available,
    compute_sensor_summary,
    format_metric,
    get_governance_details,
    get_overview_statistics,
    get_research_assistant_instance,
    get_sensor_options,
    load_engine_telemetry,
    predict_test_engine_rul,
    query_research_assistant,
)
from src.phase2_config import ACTIVE_SENSOR_COLS, CONSTANT_SENSOR_COLS
from src.phase3_config import SYSTEM_DISCLAIMER
from src.rag_retriever import EmbeddingEngine


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shared_assistant():
    """Load research assistant once for Phase 4 integration tests."""
    engine = EmbeddingEngine()
    return get_research_assistant_instance(shared_engine=engine)


# ─── 1. Overview Statistics Tests ─────────────────────────────────────────────

class TestOverviewServices:
    """Test suite for dashboard overview statistics."""

    def test_overview_statistics_schema(self):
        """Verify get_overview_statistics returns expected structure."""
        stats = get_overview_statistics()
        assert "dataset" in stats
        assert "phase2_models" in stats
        assert "rag_kb" in stats
        assert "disclaimer" in stats

    def test_format_metric_marks_missing_values_unavailable(self):
        assert format_metric(12.345) == "12.35"
        assert format_metric(None) == "Unavailable"

    def test_overview_counts_match_datasets(self):
        """Verify fleet counts and sensor partitions match ground truth."""
        stats = get_overview_statistics()
        assert stats["dataset"]["train_engines"] == 100
        assert stats["dataset"]["test_engines"] == 100
        assert stats["dataset"]["train_rows"] == 20631
        assert stats["dataset"]["test_rows"] == 13096
        assert stats["dataset"]["total_sensors"] == 21
        assert stats["dataset"]["active_sensors"] == len(ACTIVE_SENSOR_COLS)
        assert stats["dataset"]["constant_sensors"] == len(CONSTANT_SENSOR_COLS)

    def test_overview_phase2_metrics_values(self):
        """Verify Phase 2 evaluation metrics match saved benchmark values."""
        stats = get_overview_statistics()
        # Test metrics should be positive floats within realistic boundaries
        assert 40.0 < stats["phase2_models"]["test_mae"] < 55.0
        assert 50.0 < stats["phase2_models"]["test_rmse"] < 65.0
        assert stats["phase2_models"]["val_rf_mae"] < stats["phase2_models"]["val_baseline_mae"]


# ─── 2. Sensor Analytics Tests ────────────────────────────────────────────────

class TestSensorAnalyticsServices:
    """Test suite for sensor telemetry loading and statistics."""

    def test_get_sensor_options_count_and_schema(self):
        """Verify all 21 sensors are represented with required metadata."""
        opts = get_sensor_options()
        assert len(opts) == 21
        for opt in opts:
            assert "sensor_id" in opt
            assert "label" in opt
            assert "description" in opt
            assert "is_constant" in opt
            assert opt["sensor_id"] in SENSOR_COLUMNS

    def test_load_engine_telemetry_train(self):
        """Verify loading training trajectory for unit 1."""
        df_train = load_engine_telemetry(dataset_type="train", unit_number=1)
        assert isinstance(df_train, pd.DataFrame)
        assert len(df_train) == 192  # Engine 1 runs for 192 cycles in train_FD001
        assert (df_train["unit_number"] == 1).all()
        assert df_train["cycle"].is_monotonic_increasing

    def test_load_engine_telemetry_test(self):
        """Verify loading test trajectory for unit 1."""
        df_test = load_engine_telemetry(dataset_type="test", unit_number=1)
        assert isinstance(df_test, pd.DataFrame)
        assert len(df_test) == 31  # Engine 1 has 31 cycles in test_FD001
        assert (df_test["unit_number"] == 1).all()
        assert df_test["cycle"].is_monotonic_increasing

    def test_compute_sensor_summary_active_sensor(self):
        """Verify statistical metrics computation for active sensor."""
        df_train = load_engine_telemetry(dataset_type="train", unit_number=1)
        summary = compute_sensor_summary(df_train, "sensor_11")
        assert "initial_mean" in summary
        assert "final_mean" in summary
        assert "delta" in summary
        assert "min" in summary
        assert "max" in summary
        assert "std" in summary
        assert summary["total_cycles"] == 192
        assert summary["min"] <= summary["mean"] <= summary["max"]

    def test_compute_sensor_summary_invalid_column(self):
        """Verify compute_sensor_summary returns empty dict for unknown sensor."""
        df_train = load_engine_telemetry(dataset_type="train", unit_number=1)
        summary = compute_sensor_summary(df_train, "non_existent_sensor_99")
        assert summary == {}


# ─── 3. RUL Prediction Integration Tests ──────────────────────────────────────

class TestRULPredictionIntegration:
    """Test suite for Phase 2 model inference integration in dashboard."""

    def test_model_artifacts_available(self):
        """Verify feature pipeline and trained model pickle artifacts exist."""
        available, err = check_model_artifacts_available()
        assert available is True
        assert err == ""

    def test_predict_test_engine_rul_unit_1(self):
        """Verify prediction on Test Engine 1 matches Phase 2 recorded prediction."""
        result = predict_test_engine_rul(unit_number=1)
        assert result["unit_number"] == 1
        assert result["last_observed_cycle"] == 31
        # Predicted RUL should match the recorded ~42.3 cycles
        assert 40.0 <= result["predicted_rul"] <= 45.0
        assert result["true_rul"] == 112.0
        assert result["error"] < 0  # Underestimated RUL
        assert result["health_status"] == "Moderate Degradation"
        assert SYSTEM_DISCLAIMER in result["disclaimer"]

    def test_predict_test_engine_rul_unit_34(self):
        """Verify prediction on Test Engine 34."""
        result = predict_test_engine_rul(unit_number=34)
        assert result["unit_number"] == 34
        assert result["last_observed_cycle"] > 0
        assert isinstance(result["predicted_rul"], float)
        assert result["predicted_rul"] > 0
        assert result["health_status"] in [
            "Critical Degradation",
            "Moderate Degradation",
            "Nominal / Early Life",
        ]

    def test_predict_test_engine_invalid_unit_raises(self):
        """Verify invalid unit_number raises ValueError."""
        with pytest.raises(ValueError, match="not found in test dataset"):
            predict_test_engine_rul(unit_number=999)


# ─── 4. RAG Research Assistant Integration Tests ──────────────────────────────

class TestRAGAssistantIntegration:
    """Test suite for research assistant integration in dashboard."""

    def test_query_research_assistant_supported(self, shared_assistant):
        """Verify supported query returns answered status with valid citations."""
        query = "What is the health index definition and how is the failure criterion reached in the simulation?"
        res = query_research_assistant(shared_assistant, query=query, top_k=2)
        assert res["status"] == "answered"
        assert len(res["citations"]) > 0
        assert "Damage Propagation Modeling.pdf" in res["citations"][0]
        assert SYSTEM_DISCLAIMER in res["answer"]

    def test_query_research_assistant_refusal(self, shared_assistant):
        """Verify unsupported maintenance procedure returns refusal status."""
        query = "What are the FAA certified bolt torque limits for GE90 turbine casing maintenance?"
        res = query_research_assistant(shared_assistant, query=query)
        assert res["status"] == "insufficient_evidence"
        assert len(res["citations"]) == 0
        assert SYSTEM_DISCLAIMER in res["answer"]

    def test_query_research_assistant_with_engine_context(self, shared_assistant):
        """Verify engine context payload integrates into retrieval query cleanly."""
        context = {
            "unit_number": 42,
            "current_cycle": 160,
            "predicted_rul": 35.4,
            "health_status": "Degrading",
            "anomalous_sensors": ["sensor_11", "sensor_12"],
        }
        res = query_research_assistant(
            shared_assistant,
            query="What degradation mechanisms affect High Pressure Compressor HPC flow and stall margin?",
            top_k=2,
            engine_context_dict=context,
        )
        assert res["status"] == "answered"
        assert res["engine_context"] is not None
        assert res["engine_context"]["unit_number"] == 42


# ─── 5. Governance Details Tests ──────────────────────────────────────────────

class TestGovernanceDetails:
    """Test suite for governance and provenance service."""

    def test_governance_schema_and_content(self):
        """Verify governance dictionary includes all required compliance sections."""
        gov = get_governance_details()
        assert "dataset_provenance" in gov
        assert "model_methodology" in gov
        assert "performance_benchmark" in gov
        assert "known_limitations" in gov
        assert "regulatory_disclaimer" in gov

        assert "NASA" in gov["dataset_provenance"]["source"]
        assert len(gov["known_limitations"]) >= 3
        assert SYSTEM_DISCLAIMER in gov["regulatory_disclaimer"]
