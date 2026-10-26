"""AeroGuard AI — Integrated Streamlit Intelligence Dashboard (Phase 4).

Features:
  1. Overview: Validated dataset dimensions, Phase 2 test metrics, RAG knowledge base.
  2. Sensor Analytics: Trajectories, operational settings, statistics for train/test engines.
  3. RUL Prediction: Real Phase 2 model & feature pipeline inference on test engines.
  4. Ask AeroGuard AI: Grounded RAG research assistant with citations, scores, and refusal.
  5. Model & Governance: Provenance, methodology, limitations, and safety disclaimers.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
from src.rag_retriever import EmbeddingEngine


# ─── Streamlit Page Configuration ─────────────────────────────────────────────

st.set_page_config(
    page_title="AeroGuard AI — Turbofan Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS Styling ───────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Global Aerospace Dark Palette */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .aeroguard-hero {
        background: linear-gradient(135deg, #161b22 0%, #1f2937 50%, #0f172a 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #58a6ff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    
    /* Disclaimers & Badges */
    .disclaimer-banner {
        background-color: rgba(210, 153, 34, 0.12);
        border-left: 4px solid #d29922;
        padding: 10px 16px;
        border-radius: 4px;
        color: #e3b341;
        font-size: 0.88rem;
        line-height: 1.4;
        margin-bottom: 16px;
    }
    .historical-notice {
        background-color: rgba(56, 139, 253, 0.12);
        border-left: 4px solid #388bfd;
        padding: 8px 14px;
        border-radius: 4px;
        color: #79c0ff;
        font-size: 0.85rem;
        margin-bottom: 16px;
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #3fb950;
    }
    
    /* Status Badges */
    .badge-nominal {
        background-color: #238636;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-moderate {
        background-color: #9e6a03;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-critical {
        background-color: #da3633;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Citation chip */
    .citation-chip {
        display: inline-block;
        background-color: #21262d;
        border: 1px solid #388bfd;
        color: #79c0ff;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# ─── Cached Resource & Data Loaders ───────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing Sentence Transformers & FAISS Index...")
def get_cached_embedding_and_assistant():
    """Load and cache the embedding engine and Phase 3 research assistant once."""
    shared_engine = EmbeddingEngine()
    assistant = get_research_assistant_instance(shared_engine=shared_engine)
    return assistant


@st.cache_data(ttl=3600)
def get_cached_overview():
    """Cache the overview statistics."""
    return get_overview_statistics()


@st.cache_data(ttl=3600)
def get_cached_engine_data(dataset_type: str, unit_number: int):
    """Cache telemetry rows for a specific engine."""
    return load_engine_telemetry(dataset_type=dataset_type, unit_number=unit_number)


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## **AeroGuard AI**")
    st.caption("Turbofan Health & RAG Assistant | Phase 4")

    section = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Sensor Analytics",
            "3. RUL Prediction",
            "4. Ask AeroGuard AI (RAG)",
            "5. Model & Governance",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### **Operational Status**")
    model_ready, model_error = check_model_artifacts_available()
    if model_ready:
        st.success("ML model artifacts: available")
    else:
        st.warning(f"ML model artifacts: unavailable ({model_error})")
    from src.phase3_config import FAISS_INDEX_PATH, INDEX_METADATA_PATH
    if FAISS_INDEX_PATH.exists() and INDEX_METADATA_PATH.exists():
        st.success("FAISS knowledge base: available")
    else:
        st.warning("FAISS knowledge base: unavailable")
    st.info("● Subdataset: C-MAPSS FD001")

    st.markdown("---")
    st.caption("AeroGuard AI is an academic prototype. Not certified for airworthiness.")


# ─── Header Hero Section ──────────────────────────────────────────────────────

st.markdown(
    """
    <div class="aeroguard-hero">
        <div class="hero-title">✈️ AeroGuard AI — Turbofan Intelligence Platform</div>
        <div class="hero-subtitle">Integrated Engine Health Monitoring, Predictive RUL Regression & Evidence-Grounded Research Assistant</div>
        <div class="disclaimer-banner">
            <strong>RESEARCH PROTOTYPE NOTICE:</strong> AeroGuard AI is developed exclusively for academic and prognostics research using simulated NASA C-MAPSS data. It is <strong>NOT</strong> an FAA/EASA certified maintenance system. Predictions and citations must never authorize maintenance or operational flight decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if section == "1. Overview":
    st.subheader("1. Fleet Overview & Performance Benchmark")
    st.markdown(
        "A high-level summary of validated C-MAPSS FD001 engine datasets, Phase 2 predictive regression metrics, and the Phase 3 localized RAG knowledge base."
    )

    with st.spinner("Loading fleet overview statistics..."):
        stats = get_cached_overview()

    # Top KPI Metrics Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Training Fleet</div>
                <div class="metric-val">{stats['dataset']['train_engines']} Engines</div>
                <div class="metric-sub">{stats['dataset']['train_rows']:,} Run-to-Failure Cycles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Fleet</div>
                <div class="metric-val">{stats['dataset']['test_engines']} Engines</div>
                <div class="metric-sub">{stats['dataset']['test_rows']:,} Truncated Cycles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Phase 2 Test MAE</div>
                <div class="metric-val">{stats['phase2_models']['test_mae'] if stats['phase2_models']['test_mae'] is not None else 'Unavailable'} Cycles</div>
                <div class="metric-sub">RMSE: {stats['phase2_models']['test_rmse'] if stats['phase2_models']['test_rmse'] is not None else 'Unavailable'} Cycles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">RAG Knowledge Base</div>
                <div class="metric-val">{stats['rag_kb']['total_chunks'] if stats['rag_kb']['total_chunks'] is not None else 'Unavailable'} Chunks</div>
                <div class="metric-sub">NASA Paper ({stats['rag_kb']['total_pages'] if stats['rag_kb']['total_pages'] is not None else 'unknown'} Pages, {stats['rag_kb']['embedding_dim']} dimensions)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance comparison chart and dataset architecture
    chart_col, info_col = st.columns([3, 2])

    with chart_col:
        st.markdown("#### **Phase 2 Validation & Test Metric Comparison**")
        comparison_df = pd.DataFrame(
            {
                "Evaluation Split": [
                    "Val: Mean Baseline",
                    "Val: Random Forest",
                    "Test: Random Forest",
                ],
                "MAE (cycles)": [
                    stats["phase2_models"]["val_baseline_mae"],
                    stats["phase2_models"]["val_rf_mae"],
                    stats["phase2_models"]["test_mae"],
                ],
                "RMSE (cycles)": [stats["phase2_models"]["val_baseline_rmse"], stats["phase2_models"]["val_rf_rmse"], stats["phase2_models"]["test_rmse"]],
            }
        )

        fig_comp = px.bar(
            comparison_df,
            x="Evaluation Split",
            y=["MAE (cycles)", "RMSE (cycles)"],
            barmode="group",
            color_discrete_sequence=["#58a6ff", "#f0883e"],
            title="Model Error Metrics Across Evaluation Partitions",
        )
        fig_comp.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#c9d1d9",
            legend_title_text="Metric",
            height=340,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with info_col:
        st.markdown("#### **System & Dataset Attributes**")
        st.caption(f"Dataset validation report: {stats['dataset']['validation_status']}")
        if stats["dataset"]["validation_failures"]:
            st.error("; ".join(stats["dataset"]["validation_failures"]))
        st.markdown(
            f"""
            - **Dataset Standard**: NASA C-MAPSS Turbofan Degradation (FD001)
            - **Operating Conditions**: Sea-level single regime (Altitude 0 ft, Mach 0.0)
            - **Sensor Channels**: 21 raw channels (15 active variable, 6 constant removed)
            - **Feature Count**: 109 engineered features (rolling means & stds over 5, 10, 20 cycles)
            - **ML Model**: `RandomForestRegressor` (200 trees, max depth 15, max features 0.6)
            - **RAG Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (Local PyTorch)
            - **Similarity Metric**: Exact Inner Product (Cosine Similarity) with FAISS
            """
        )
        if stats["phase2_models"]["val_rf_mae"] is not None:
            st.info(f"Validation MAE: {stats['phase2_models']['val_rf_mae']:.2f} cycles ({stats['phase2_models']['improvement_pct']:.1f}% lower MAE than the mean baseline).")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SENSOR ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif section == "2. Sensor Analytics":
    st.subheader("2. Sensor Analytics & Trajectory Inspection")
    st.markdown(
        "Inspect observed sensor trajectories and operational settings across flight cycles for any training or test turbofan unit."
    )

    st.markdown(
        """
        <div class="historical-notice">
            <strong>⚠️ HISTORICAL BENCHMARK OBSERVATIONS:</strong> The time series shown below are historical numerical simulation cycles recorded from NASA C-MAPSS FD001. They represent bench simulation run-to-failure or truncated testing, <strong>NOT</strong> live streaming aircraft telemetry.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        dataset_type = st.selectbox(
            "Select Partition",
            ["Training (Run-to-Failure)", "Test (Truncated)"],
            index=0,
        )
        ds_code = "train" if "Training" in dataset_type else "test"

    with ctrl_col2:
        unit_number = st.number_input("Engine Unit ID", min_value=1, max_value=100, value=1, step=1)

    sensor_opts = get_sensor_options()
    sensor_map = {opt["label"]: opt["sensor_id"] for opt in sensor_opts}
    sensor_descriptions = {opt["sensor_id"]: opt["description"] for opt in sensor_opts}

    with ctrl_col3:
        # Default to sensor_11 (HPC static pressure) or sensor_2
        default_idx = 10  # sensor_11
        selected_sensor_label = st.selectbox(
            "Select Sensor Channel",
            list(sensor_map.keys()),
            index=default_idx,
        )
        selected_sensor_id = sensor_map[selected_sensor_label]

    # Load engine telemetry
    engine_df = get_cached_engine_data(ds_code, int(unit_number))

    if engine_df.empty:
        st.error(f"No telemetry records found for Engine Unit {unit_number}.")
    else:
        summary = compute_sensor_summary(engine_df, selected_sensor_id)

        # Summary Metric Bar
        s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
        with s_col1:
            st.metric("Total Observed Cycles", f"{summary.get('total_cycles', len(engine_df))}")
        with s_col2:
            st.metric("Initial 5-Cycle Mean", f"{summary.get('initial_mean', 0.0):.2f}")
        with s_col3:
            st.metric("Final 5-Cycle Mean", f"{summary.get('final_mean', 0.0):.2f}")
        with s_col4:
            st.metric("Net Sensor Drift (Δ)", f"{summary.get('delta', 0.0):+.2f}")
        with s_col5:
            st.metric("Std Dev", f"{summary.get('std', 0.0):.3f}")

        # Plotly Time-Series Trajectory
        fig_sensor = go.Figure()

        # Raw observed values
        fig_sensor.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=engine_df[selected_sensor_id],
                mode="lines+markers",
                name="Observed Trajectory",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=4),
            )
        )

        # 5-cycle rolling average
        rolling_5 = engine_df[selected_sensor_id].rolling(window=5, min_periods=1).mean()
        fig_sensor.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=rolling_5,
                mode="lines",
                name="5-Cycle Rolling Mean",
                line=dict(color="#f0883e", width=2, dash="dash"),
            )
        )

        desc = sensor_descriptions.get(selected_sensor_id, selected_sensor_id)
        fig_sensor.update_layout(
            title=f"Engine #{unit_number} — {selected_sensor_id}: {desc}",
            xaxis_title="Flight Cycles",
            yaxis_title=f"{selected_sensor_id} Measurement",
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font_color="#c9d1d9",
            height=400,
            hovermode="x unified",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sensor, use_container_width=True)

        # Operational Settings Trajectory
        with st.expander("Inspect Operational Settings (Settings 1, 2, and 3)"):
            fig_settings = go.Figure()
            colors = ["#388bfd", "#7ee787", "#d2a8ff"]
            for i, set_col in enumerate(["setting_1", "setting_2", "setting_3"]):
                fig_settings.add_trace(
                    go.Scatter(
                        x=engine_df["cycle"],
                        y=engine_df[set_col],
                        mode="lines",
                        name=set_col,
                        line=dict(color=colors[i], width=1.5),
                    )
                )
            fig_settings.update_layout(
                title=f"Engine #{unit_number} — Operational Settings across Cycles",
                xaxis_title="Flight Cycles",
                yaxis_title="Setting Value",
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                height=260,
                margin=dict(l=40, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_settings, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: RUL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

elif section == "3. RUL Prediction":
    st.subheader("3. Remaining Useful Life (RUL) Prediction")
    st.markdown(
        "Select an engine from `test_FD001.txt`. Evaluates remaining useful life using the exact Phase 2 feature-engineering scaler and trained `RandomForestRegressor`."
    )

    artifacts_ok, artifact_error = check_model_artifacts_available()
    if not artifacts_ok:
        st.error(f"⚠️ Model Artifacts Missing: {artifact_error}")
        st.info("Please run `python src/run_phase2.py` in PowerShell to generate model artifacts.")
    else:
        pred_col1, pred_col2 = st.columns([1, 2])

        with pred_col1:
            st.markdown("#### **Test Engine Selection**")
            test_engine_id = st.number_input(
                "Select Test Engine (1 to 100)",
                min_value=1,
                max_value=100,
                value=34,
                step=1,
            )

            predict_btn = st.button("Predict Remaining Useful Life", type="primary", use_container_width=True)

        pred_result = None
        if predict_btn:
            try:
                with st.spinner("Applying the saved Phase 2 feature pipeline and model..."):
                    pred_result = predict_test_engine_rul(int(test_engine_id))
                st.session_state["last_rul_prediction"] = pred_result
                st.session_state["last_rul_engine"] = int(test_engine_id)
            except Exception as e:
                st.error(f"Error executing prediction pipeline: {e}")
        elif st.session_state.get("last_rul_engine") == int(test_engine_id):
            pred_result = st.session_state.get("last_rul_prediction")

        if pred_result:
            with pred_col2:
                st.markdown("#### **Prognostic Output**")
                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    st.metric(
                        "Last Observed Cycle",
                        f"{pred_result['last_observed_cycle']} cycles",
                    )
                with res_col2:
                    st.metric(
                        "Predicted RUL",
                        f"{pred_result['predicted_rul']} cycles",
                        help="Exact regression prediction from Phase 2 RandomForest",
                    )
                with res_col3:
                    if pred_result["true_rul"] is not None:
                        st.metric(
                            "Ground Truth RUL",
                            f"{pred_result['true_rul']} cycles",
                            delta=f"{pred_result['error']:+.1f} cycles",
                            delta_color="inverse",
                        )

                st.caption("Research estimate only. This result does not establish airworthiness or authorize maintenance or operational decisions.")

            st.markdown("---")

            # Timeline visualization: observed history + projected horizon
            st.markdown("#### **Engine Degradation Horizon Timeline**")
            last_c = pred_result["last_observed_cycle"]
            pred_r = pred_result["predicted_rul"]
            true_r = pred_result.get("true_rul", 0)

            fig_timeline = go.Figure()

            # Observed historical span
            fig_timeline.add_trace(
                go.Scatter(
                    x=[1, last_c],
                    y=[0, 0],
                    mode="lines+markers",
                    name="Observed Cycles",
                    line=dict(color="#388bfd", width=8),
                    marker=dict(size=12, symbol="circle"),
                )
            )

            # Predicted remaining span
            fig_timeline.add_trace(
                go.Scatter(
                    x=[last_c, last_c + pred_r],
                    y=[0, 0],
                    mode="lines+markers",
                    name=f"Predicted Horizon (+{pred_r:.1f} cycles)",
                    line=dict(color="#f0883e", width=8, dash="dot"),
                    marker=dict(size=14, symbol="diamond"),
                )
            )

            # Ground truth point if available
            if true_r is not None:
                fig_timeline.add_trace(
                    go.Scatter(
                        x=[last_c + true_r],
                        y=[0],
                        mode="markers+text",
                        name=f"Ground Truth Failure (+{true_r:.1f} cycles)",
                        marker=dict(size=16, color="#da3633", symbol="x"),
                        text=["Actual Failure Point"],
                        textposition="top center",
                    )
                )

            fig_timeline.update_layout(
                xaxis_title="Flight Cycles (Historical + Projected)",
                yaxis=dict(visible=False),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                height=220,
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Mandatory Non-Airworthiness Notice
            st.warning(
                "⚠️ **REGULATORY DISCLAIMER**: Remaining Useful Life predictions are regression estimates for academic research. "
                "This value is **NOT** a certified failure probability or airworthiness guarantee. Do **NOT** use for operational dispatch or maintenance sign-off."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ASK AEROGUARD AI (RAG)
# ═══════════════════════════════════════════════════════════════════════════════

elif section == "4. Ask AeroGuard AI (RAG)":
    st.subheader("4. Ask AeroGuard AI — Evidence-Grounded Research Assistant")
    st.markdown(
        "Query the foundational NASA C-MAPSS literature (*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*). "
        "Strictly grounded in localized FAISS retrieved passages. **Refuses out-of-domain or unverified maintenance procedures.**"
    )

    try:
        assistant = get_cached_embedding_and_assistant()
    except Exception as e:
        st.error(f"Error loading Phase 3 RAG Assistant: {e}")
        assistant = None

    if assistant:
        # Sample benchmark query buttons
        st.markdown("##### **Quick Benchmark Queries**")
        q_cols = st.columns(4)

        query_input = ""
        with q_cols[0]:
            if st.button("Health index & failure criteria", use_container_width=True):
                st.session_state["query_text"] = (
                    "What is the health index definition and how is the failure criterion reached in the simulation?"
                )
        with q_cols[1]:
            if st.button("Damage propagation model", use_container_width=True):
                st.session_state["query_text"] = (
                    "How is damage propagation modeled within the modules of aircraft gas turbine engines?"
                )
        with q_cols[2]:
            if st.button("C-MAPSS sensor outputs", use_container_width=True):
                st.session_state["query_text"] = (
                    "What sensor response surfaces and simulation inputs are generated via the C-MAPSS model?"
                )
        with q_cols[3]:
            if st.button("FAA torque limits (Refusal test)", use_container_width=True):
                st.session_state["query_text"] = (
                    "What are the FAA certified bolt torque limits for GE90 turbine casing maintenance?"
                )

        current_query = st.session_state.get(
            "query_text",
            "What is the health index definition and how is the failure criterion reached in the simulation?",
        )

        user_query = st.text_input(
            "Enter research question or inquiry:",
            value=current_query,
            placeholder="Ask about damage propagation, C-MAPSS parameters, or health index...",
        )

        # Engine context integration option
        with st.expander("Attach Engine Context (Decoupled Telemetry Integration)"):
            attach_ctx = st.checkbox("Include selected engine context in research query", value=False)
            ctx_unit = st.number_input("Context Engine Unit", min_value=1, max_value=100, value=42)
            ctx_dict = None
            if attach_ctx:
                ctx_dict = {
                    "unit_number": int(ctx_unit),
                    "current_cycle": 160,
                    "predicted_rul": 35.4,
                    "health_status": "Degrading",
                    "anomalous_sensors": ["sensor_11", "sensor_12"],
                }
                st.caption(f"Context payload: {ctx_dict}")

        ask_btn = st.button("Ask Assistant", type="primary")

        if user_query and (ask_btn or "query_text" in st.session_state):
            with st.spinner("Searching FAISS vector index and verifying textual evidence..."):
                result = query_research_assistant(
                    assistant=assistant,
                    query=user_query,
                    top_k=3,
                    engine_context_dict=ctx_dict,
                )

            # Display status badge
            if result["status"] == "answered":
                st.success(
                    f"**Answer Status: Grounded in Literature** (Cosine Similarity: {result['top_similarity_score']:.3f})"
                )
            else:
                st.warning(
                    f"**Answer Status: Refused — Insufficient Evidence in Research Document** (Top Score: {result['top_similarity_score']:.3f})"
                )

            # Answer Text
            st.markdown("#### **Research Assistant Response:**")
            st.markdown(result["answer"])

            # Citations Chips
            if result["citations"]:
                st.markdown("#### **Citations & Sources:**")
                chips_html = "".join(
                    [f"<span class='citation-chip'>📄 {c}</span>" for c in result["citations"]]
                )
                st.markdown(chips_html, unsafe_allow_html=True)

            # Retrieved Passages Expander
            if result.get("retrieved_evidence"):
                with st.expander("Inspect Retrieved Supporting Passages (FAISS Top Chunks)"):
                    for i, p in enumerate(result["retrieved_evidence"], start=1):
                        st.markdown(
                            f"**Passage {i} — {p['citation']} (Cosine Score: {p['score']:.4f}, Chunk ID: `{p['chunk_id']}`):**"
                        )
                        st.info(p["text"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL & GOVERNANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif section == "5. Model & Governance":
    st.subheader("5. Model & Governance: Provenance, Methodology & Compliance")
    st.markdown(
        "Formal engineering documentation regarding dataset provenance, machine learning methodology, validation architecture, known limitations, and regulatory non-certification."
    )

    gov = get_governance_details()

    gov_tab1, gov_tab2, gov_tab3, gov_tab4 = st.tabs(
        [
            "Dataset Provenance",
            "ML Methodology & Leakage Prevention",
            "Validation & Test Benchmark",
            "Limitations & Airworthiness Notice",
        ]
    )

    with gov_tab1:
        st.markdown("#### **Dataset Provenance & Simulation Architecture**")
        st.markdown(
            f"""
            - **Origin**: {gov['dataset_provenance']['source']}
            - **Simulation Engine**: {gov['dataset_provenance']['simulator']}
            - **Subdataset Partition**: {gov['dataset_provenance']['subdataset']}
            - **Operating Envelope**: {gov['dataset_provenance']['operating_regime']}
            - **Injected Fault Mode**: {gov['dataset_provenance']['fault_mode']}
            - **Nature**: {gov['dataset_provenance']['synthetic_nature']}
            """
        )
        st.markdown(
            """
            The C-MAPSS simulation generates synthetic run-to-failure degradation trajectories by varying flow and efficiency 
            parameters of gas turbine modules (Fan, LPC, HPC, HPT, LPT). Each engine trajectory starts with an undefined 
            degree of initial wear and degrades exponentially until an operational safety margin reaches zero.
            """
        )

    with gov_tab2:
        st.markdown("#### **Feature Engineering & Anti-Leakage Protocol**")
        st.markdown(
            f"""
            - **Algorithm**: `{gov['model_methodology']['algorithm']}`
            - **Hyperparameters**: `n_estimators=200`, `max_depth=15`, `min_samples_leaf=4`, `max_features=0.6`
            - **Total Engineered Features**: `{gov['model_methodology']['feature_count']}`
            - **Feature Breakdown**: {gov['model_methodology']['feature_breakdown']}
            - **Validation Split**: {gov['model_methodology']['validation_split']}
            """
        )
        st.markdown(
            """
            **Strict Anti-Leakage Guarantees:**
            1. **Engine-ID Split**: Engines 1–80 form the training partition; Engines 81–100 form the validation partition. Zero engine ID overlap.
            2. **Scaler Isolation**: `StandardScaler` is `fit()` strictly on the training partition (Engines 1–80). Validation and test sets are transformed only.
            3. **Causal Rolling Statistics**: Rolling means and standard deviations (windows 5, 10, 20) are computed strictly engine-by-engine using backward observations (`min_periods=1`). No future cycles are ever accessed.
            4. **Test Set Sealing**: The test dataset (`test_FD001.txt`) is only evaluated once at the final cycle per engine after model selection is finalized.
            """
        )

    with gov_tab3:
        st.markdown("#### **Empirical Evaluation Benchmark**")
        perf = gov["performance_benchmark"]
        b_df = pd.DataFrame(
            [
                {
                    "Evaluation Partition": "Validation Set (Cycle-Level, 4,493 cycles)",
                    "Model": "Mean Baseline",
                    "MAE (cycles)": format_metric(perf["cycle_val_baseline_mae"]),
                    "RMSE (cycles)": format_metric(perf["cycle_val_baseline_rmse"]),
                },
                {
                    "Evaluation Partition": "Validation Set (Cycle-Level, 4,493 cycles)",
                    "Model": "Random Forest",
                    "MAE (cycles)": format_metric(perf["cycle_val_rf_mae"]),
                    "RMSE (cycles)": format_metric(perf["cycle_val_rf_rmse"]),
                },
                {
                    "Evaluation Partition": "Test Set (Engine-Level, 100 test engines)",
                    "Model": "Random Forest",
                    "MAE (cycles)": format_metric(perf["engine_test_mae"]),
                    "RMSE (cycles)": format_metric(perf["engine_test_rmse"]),
                },
            ]
        )
        st.table(b_df)
        st.caption(
            "Cycle-level validation evaluates RUL prediction across all engine cycles. "
            "Engine-level test evaluation evaluates RUL strictly at the final observed cycle of each test engine."
        )

    with gov_tab4:
        st.markdown("#### **Known System Limitations**")
        for lim in gov["known_limitations"]:
            st.markdown(f"- {lim}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="disclaimer-banner">
                <strong>REGULATORY GOVERNANCE NOTICE:</strong><br>
                {gov['regulatory_disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "AeroGuard AI Phase 4 | Academic Research Prototype | Grounded on NASA C-MAPSS Benchmark Data | PyTorch & FAISS Local Inference"
)
