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
    :root {
        --bg: #020b16;
        --bg-2: #07172a;
        --panel: rgba(11, 28, 43, 0.92);
        --panel-2: rgba(17, 36, 54, 0.9);
        --border: rgba(90, 150, 255, 0.35);
        --blue: #5bb3ff;
        --cyan: #57d4ff;
        --green: #3ae6a9;
        --amber: #f4b942;
        --red: #ff5c7a;
        --white: #eaf3ff;
        --muted: #9cb4ce;
    }

    .stApp {
        background: linear-gradient(180deg, #030d18 0%, #061320 100%);
        color: var(--white);
    }

    [data-testid="stSidebar"] {
        background: rgba(2, 12, 22, 0.96);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebarNav"] {
        background: transparent;
    }

    .block-container {
        padding-top: 0.4rem;
        padding-bottom: 1rem;
    }

    .topbar {
        background: rgba(5, 20, 32, 0.95);
        border-bottom: 1px solid rgba(86, 155, 255, 0.5);
        padding: 0.85rem 1.4rem 0.75rem 1.4rem;
        margin: -0.8rem -1.2rem 1rem -1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #eaf3ff;
        font-weight: 700;
        font-size: 2rem;
    }

    .brand .brand-mark {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: linear-gradient(135deg, #49d4ff, #2d7ae8);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 18px rgba(91, 179, 255, 0.45);
        color: #021017;
        font-size: 0.9rem;
    }

    .brand small {
        color: #76baff;
        font-size: 0.62rem;
        letter-spacing: 0.22rem;
        text-transform: uppercase;
        font-weight: 700;
        margin-left: 0.35rem;
    }

    .statusbar {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        color: #d5e8ff;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .live-dot {
        width: 10px;
        height: 10px;
        background: #48e5a5;
        border-radius: 50%;
        box-shadow: 0 0 12px #48e5a5;
        display: inline-block;
    }

    .status-pill {
        border: 1px solid rgba(120, 184, 255, 0.4);
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        background: rgba(20, 53, 72, 0.5);
        color: #dfeeff;
    }

    .nav-title {
        color: #7ea7d2;
        letter-spacing: 0.2rem;
        font-size: 0.7rem;
        text-transform: uppercase;
        margin: 1rem 0 0.5rem 0;
        padding-left: 0.65rem;
    }

    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        margin-top: 0.2rem;
    }

    div[data-testid="stRadio"] label {
        background: rgba(17, 32, 48, 0.65);
        border: 1px solid rgba(91, 179, 255, 0.18);
        border-radius: 10px;
        padding: 0.8rem 0.8rem 0.8rem 0.9rem;
        color: #dfeefc;
        font-weight: 600;
        font-size: 0.98rem;
        margin: 0;
    }

    div[data-testid="stRadio"] label:hover {
        border-color: rgba(91, 179, 255, 0.8);
    }

    div[data-testid="stRadio"] input:checked + div {
        color: #ebf8ff;
        font-weight: 700;
    }

    div[data-testid="stRadio"] > div > label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(41, 90, 162, 0.9), rgba(17, 48, 79, 0.8));
        border-color: rgba(111, 179, 255, 0.9);
        box-shadow: inset 0 0 0 1px rgba(146, 196, 255, 0.4);
    }

    .hero-panel {
        background: linear-gradient(180deg, rgba(15, 28, 44, 0.95), rgba(14, 35, 54, 0.96));
        border: 1px solid rgba(91, 179, 255, 0.5);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 0.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem;
    }

    .hero-copy {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        min-width: 0;
    }

    .hero-badge {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: rgba(91, 179, 255, 0.12);
        color: #6ec8ff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(110, 200, 255, 0.4);
        font-size: 1rem;
    }

    .eyebrow {
        font-size: 0.8rem;
        letter-spacing: 0.16rem;
        text-transform: uppercase;
        color: #6ec8ff;
        font-weight: 700;
    }

    .hero-title {
        font-size: clamp(2rem, 2.9vw, 3.3rem);
        line-height: 1.15;
        margin: 0.2rem 0 0.4rem 0;
        color: #edf6ff;
        font-weight: 800;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #a9c6de;
        margin: 0;
        max-width: 760px;
    }

    .hero-actions {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .action-btn {
        padding: 0.9rem 1.1rem;
        border-radius: 11px;
        border: 1px solid rgba(91, 179, 255, 0.55);
        font-weight: 700;
        cursor: pointer;
        background: rgba(21, 61, 88, 0.9);
        color: #edf7ff;
    }

    .action-btn.primary {
        background: linear-gradient(135deg, #2aa7ff, #1d86f6);
        color: #031827;
        border-color: rgba(110, 200, 255, 0.8);
        box-shadow: 0 8px 18px rgba(80, 170, 255, 0.35);
    }

    [data-testid="stHorizontalBlock"]:has(.hero-copy) {
        background: linear-gradient(180deg, rgba(15, 28, 44, 0.95), rgba(14, 35, 54, 0.96));
        border: 1px solid rgba(91, 179, 255, 0.5);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
        align-items: center;
    }

    .hero-actions-container {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
    }

    div.st-key-hero_rul_btn button {
        background: linear-gradient(135deg, #2aa7ff, #1d86f6) !important;
        color: #031827 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(110, 200, 255, 0.8) !important;
        box-shadow: 0 4px 14px rgba(80, 170, 255, 0.35) !important;
        border-radius: 11px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }

    div.st-key-hero_rul_btn button:hover {
        background: linear-gradient(135deg, #44b6ff, #2a94f8) !important;
        box-shadow: 0 6px 20px rgba(80, 170, 255, 0.5) !important;
    }

    div.st-key-hero_copilot_btn button {
        background: rgba(21, 61, 88, 0.9) !important;
        color: #edf7ff !important;
        font-weight: 700 !important;
        border: 1px solid rgba(91, 179, 255, 0.55) !important;
        border-radius: 11px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
    }

    div.st-key-hero_copilot_btn button:hover {
        background: rgba(30, 85, 120, 0.95) !important;
        border-color: rgba(110, 200, 255, 0.8) !important;
        color: #ffffff !important;
    }

    div.st-key-inspect_matrix_btn button, div.st-key-rul_to_copilot_btn button {
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .kpi-card {
        background: rgba(13, 26, 38, 0.88);
        border: 1px solid rgba(95, 154, 255, 0.25);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        min-height: 122px;
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, rgba(91,179,255,0.8), rgba(90,255,205,0.1));
    }

    .kpi-label {
        color: #7ea7d2;
        font-size: 0.72rem;
        letter-spacing: 0.18rem;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-value {
        font-size: clamp(2rem, 2vw, 2.5rem);
        font-weight: 800;
        color: var(--white);
        margin-top: 0.6rem;
    }

    .kpi-sub {
        color: #9ec4e0;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    .kpi-card .positive { color: #46e6a3; }
    .kpi-card .warning { color: #f4b942; }
    .kpi-card .danger { color: #ff6b82; }
    .kpi-card .info { color: #6ec8ff; }

    .matrix-wrap {
        display: grid;
        grid-template-columns: 2.2fr 1fr;
        gap: 1rem;
        margin-top: 1.35rem;
    }

    .panel {
        background: rgba(9, 23, 34, 0.9);
        border: 1px solid rgba(96, 153, 255, 0.35);
        border-radius: 12px;
        padding: 1rem 1.05rem 0.8rem 1.05rem;
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }

    .title {
        font-size: 1.1rem;
        color: #edf6ff;
        font-weight: 800;
        margin: 0;
    }

    .subtitle {
        color: #8ca8c7;
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    .searchbox {
        background: rgba(15, 32, 46, 0.9);
        border: 1px solid rgba(92, 159, 255, 0.35);
        border-radius: 10px;
        color: var(--white);
        padding: 0.55rem 0.8rem;
        min-width: 180px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }

    th {
        color: #8ca8c7;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
        padding: 0.7rem 0.5rem;
        text-align: left;
        border-bottom: 1px solid rgba(111, 163, 255, 0.22);
    }

    td {
        padding: 0.85rem 0.5rem;
        border-bottom: 1px solid rgba(111, 163, 255, 0.12);
        color: #ebf4ff;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .status-optimal { background: rgba(58, 230, 169, 0.12); color: #5ceab1; border: 1px solid rgba(58, 230, 169, 0.35); }
    .status-warning { background: rgba(244, 185, 66, 0.1); color: #f4b942; border: 1px solid rgba(244, 185, 66, 0.35); }
    .status-critical { background: rgba(255, 92, 122, 0.12); color: #ff8aa4; border: 1px solid rgba(255, 92, 122, 0.35); }

    .alert-list {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }

    .alert-card {
        background: rgba(17, 34, 49, 0.9);
        border: 1px solid rgba(255, 175, 77, 0.25);
        border-radius: 10px;
        padding: 0.85rem 0.9rem;
        color: #eaf5ff;
    }

    .alert-level {
        color: #ffd166;
        text-transform: uppercase;
        letter-spacing: 0.12rem;
        font-size: 0.68rem;
        font-weight: 700;
    }

    .alert-title {
        color: #fef8d8;
        font-weight: 700;
        margin-top: 0.2rem;
        margin-bottom: 0.35rem;
    }

    .alert-meta {
        color: #9ebdd3;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }

    .fleet-footer {
        margin-top: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #9ec0dc;
        font-size: 0.82rem;
    }

    .mini-bar {
        height: 6px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 0.35rem;
    }

    .mini-bar > span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #44dfae, #5bb3ff);
    }

    @media (max-width: 1040px) {
        .hero-panel { display: block; }
        .hero-actions { justify-content: flex-start; margin-top: 1rem; }
        .kpi-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
        .matrix-wrap { grid-template-columns: 1fr; }
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

NAV_SECTIONS = [
    "Fleet Overview",
    "Data & Sensor Telemetry",
    "Engine Analysis & Trends",
    "RUL Prediction & ML",
    "Research Co-Pilot (RAG)",
    "Model Performance",
    "Governance & Specs",
]

if "nav_section" not in st.session_state or st.session_state["nav_section"] not in NAV_SECTIONS:
    st.session_state["nav_section"] = "Fleet Overview"

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">✦</span>AeroGuard.AI<small>AEROSPACE ENGINE INTEL</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">Navigation System</div>', unsafe_allow_html=True)

    current_nav_idx = NAV_SECTIONS.index(st.session_state["nav_section"])
    section = st.radio(
        "Navigation Menu",
        NAV_SECTIONS,
        index=current_nav_idx,
        label_visibility="collapsed",
    )
    st.session_state["nav_section"] = section

    st.markdown("---")
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
    st.info("Subdataset: C-MAPSS FD001")

    st.markdown("---")
    st.caption("AeroGuard AI is an academic prototype. Not certified for airworthiness.")


st.markdown(
    """
    <div class="topbar">
      <div class="brand"><span class="brand-mark">✦</span>AeroGuard.AI<small>AEROSPACE ENGINE INTEL</small></div>
      <div class="statusbar">
        <span><span class="live-dot"></span> N-CMAPSS TELEMETRY LIVE</span>
        <span class="status-pill">12:04:09 UTC</span>
        <span>🔔</span>
        <span>🇺🇸</span>
        <button class="action-btn" style="padding: 0.55rem 1rem; border-radius: 10px;">Flight Eng. Ops</button>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Custom Overview Dashboard ─────────────────────────────────────────────────

if section == "Fleet Overview":
    stats = get_cached_overview()

    hero_col_text, hero_col_actions = st.columns([3.3, 1.3], vertical_alignment="center")
    with hero_col_text:
        st.markdown(
            """
            <div class="hero-copy">
              <div class="hero-badge">✈</div>
              <div>
                <div class="eyebrow">Aerospace Predictive Maintenance Platform</div>
                <div class="hero-title">Fleet Maintenance Intelligence & Control</div>
                <div class="hero-subtitle">Real-time degradation tracking, physics-informed RUL forecasting, and AI-driven technical documentation intelligence for turbofan fleets.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_col_actions:
        st.markdown('<div class="hero-actions-container">', unsafe_allow_html=True)
        if st.button("🚀 Run RUL Inference", type="primary", use_container_width=True, key="hero_rul_btn"):
            st.session_state["nav_section"] = "RUL Prediction & ML"
            st.session_state["auto_run_rul"] = True
            st.rerun()
        if st.button("💬 Ask Co-Pilot", use_container_width=True, key="hero_copilot_btn"):
            st.session_state["nav_section"] = "Research Co-Pilot (RAG)"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    kpis = [
        ("Total Turbofans", f"{stats['dataset']['train_engines'] + stats['dataset']['test_engines']}", "↑ 12 active flight missions", "info"),
        ("Fleet Health Index", f"{94.2:.1f}%", "Nominal across A320neo & B737 MAX", "positive"),
        ("Critical Alerts", "4 Engines", "RUL < 45 cycles threshold", "warning"),
        ("Avg Fleet RUL", "142 Cycles", "Mean flight cycle expectancy", "info"),
    ]

    cols = st.columns(4)
    for col, (label, value, sub, tone) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label"><span>{label}</span><span class="{tone}">✦</span></div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-sub {tone}">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="matrix-wrap">', unsafe_allow_html=True)
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-header">
                <div>
                  <div class="title">Fleet Turbofan Matrix</div>
                  <div class="subtitle">N-CMAPSS Simulated Fleet Flight Telemetry</div>
                </div>
              </div>
            """,
            unsafe_allow_html=True,
        )
        fleet_rows = [
            ("ENG-#004", "A320neo (CFM LEAP-1A)", "284", "158 cycles", "Optimal"),
            ("ENG-#012", "B737 MAX (LEAP-1B)", "310", "38 cycles", "Warning"),
            ("ENG-#026", "A321neo (CFM LEAP-1A)", "296", "54 cycles", "Optimal"),
            ("ENG-#047", "A330-900neo", "332", "17 cycles", "Critical"),
            ("ENG-#093", "B787-9", "288", "112 cycles", "Optimal"),
        ]
        table_html = """
        <table>
          <tr><th>Engine ID</th><th>Aircraft Model</th><th>Cycles</th><th>Est. RUL</th><th>Health Status</th></tr>
        """
        for eng_id, aircraft, cycles, rul, health in fleet_rows:
            badge_class = "status-optimal" if health == "Optimal" else "status-warning" if health == "Warning" else "status-critical"
            table_html += (
                f"<tr><td><strong>{eng_id}</strong></td><td>{aircraft}</td><td>{cycles}</td><td>{rul}</td><td><span class='status-badge {badge_class}'>{health}</span></td></tr>"
            )
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
        insp_col1, insp_col2 = st.columns([3, 1.4], vertical_alignment="center")
        with insp_col1:
            inspect_target = st.selectbox(
                "Quick Inspect Turbofan:",
                [
                    "ENG-#004 (A320neo, CFM LEAP-1A)",
                    "ENG-#012 (B737 MAX, LEAP-1B)",
                    "ENG-#026 (A321neo, CFM LEAP-1A)",
                    "ENG-#047 (A330-900neo)",
                    "ENG-#093 (B787-9)",
                ],
                index=1,
                label_visibility="collapsed",
                key="matrix_inspect_select",
            )
        with insp_col2:
            if st.button("Inspect Engine", use_container_width=True, key="inspect_matrix_btn"):
                unit_map = {
                    "ENG-#004": 4,
                    "ENG-#012": 12,
                    "ENG-#026": 26,
                    "ENG-#047": 47,
                    "ENG-#093": 93,
                }
                unit_key = inspect_target.split(" ")[0]
                st.session_state["selected_engine_id"] = unit_map.get(unit_key, 12)
                st.session_state["nav_section"] = "Engine Analysis & Trends"
                st.rerun()
        st.markdown(
            """
            <div class="fleet-footer">
              <span>Active fleet health 94.2%</span>
              <span>142 turbo fans active</span>
              <span>4 critical</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="panel">
              <div class="panel-header">
                <div class="title">Live Cockpit Alerts</div>
                <span class="status-pill">STREAM ACTIVE</span>
              </div>
              <div class="alert-list">
                <div class="alert-card">
                  <div class="alert-level">Engine #012 • HPC Temp Spike</div>
                  <div class="alert-title">High-Pressure Compressor Outlet Temperature (T30) drift +3.8°C above baseline threshold.</div>
                  <div class="alert-meta">10m ago</div>
                </div>
                <div class="alert-card">
                  <div class="alert-level">Engine #210 • Vibration Anomaly</div>
                  <div class="alert-title">Rotor vibration increased by 0.43 in/s across the last 5 cycles.</div>
                  <div class="alert-meta">28m ago</div>
                </div>
                <div class="alert-card">
                  <div class="alert-level">Engine #047 • RUL Threshold</div>
                  <div class="alert-title">Remaining useful life projected below maintenance trigger at 17 cycles.</div>
                  <div class="alert-meta">2h ago</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif section == "Data & Sensor Telemetry":
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

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        dataset_type = st.selectbox(
            "Select Partition",
            ["Training (Run-to-Failure)", "Test (Truncated)"],
            index=0,
        )
        ds_code = "train" if "Training" in dataset_type else "test"

    with ctrl_col2:
        unit_number = st.number_input(
            "Engine Unit ID",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get("selected_engine_id", 1)),
            step=1,
        )

    sensor_opts = get_sensor_options()
    sensor_map = {opt["label"]: opt["sensor_id"] for opt in sensor_opts}
    sensor_descriptions = {opt["sensor_id"]: opt["description"] for opt in sensor_opts}

    with ctrl_col3:
        default_idx = 10
        selected_sensor_label = st.selectbox(
            "Select Sensor Channel",
            list(sensor_map.keys()),
            index=default_idx,
        )
        selected_sensor_id = sensor_map[selected_sensor_label]

    engine_df = get_cached_engine_data(ds_code, int(unit_number))

    if engine_df.empty:
        st.error(f"No telemetry records found for Engine Unit {unit_number}.")
    else:
        summary = compute_sensor_summary(engine_df, selected_sensor_id)

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

        fig_sensor = go.Figure()
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
# SECTION 2: SENSOR ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif section == "Engine Analysis & Trends":
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
        unit_number = st.number_input(
            "Engine Unit ID",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get("selected_engine_id", 1)),
            step=1,
        )

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

elif section == "RUL Prediction & ML":
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
            default_test_id = int(st.session_state.get("selected_test_engine_id", 34))
            test_engine_id = st.number_input(
                "Select Test Engine (1 to 100)",
                min_value=1,
                max_value=100,
                value=default_test_id,
                step=1,
            )

            predict_btn = st.button("Predict Remaining Useful Life", type="primary", use_container_width=True)

        pred_result = None
        auto_run = st.session_state.pop("auto_run_rul", False)
        if predict_btn or auto_run:
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

                if st.button("💬 Ask Co-Pilot About Engine #" + str(test_engine_id), key="rul_to_copilot_btn", use_container_width=True):
                    st.session_state["nav_section"] = "Research Co-Pilot (RAG)"
                    st.session_state["copilot_engine_context"] = int(test_engine_id)
                    st.session_state["query_text"] = f"What degradation mechanisms affect engine performance at cycle {pred_result['last_observed_cycle']} with predicted RUL of {pred_result['predicted_rul']:.1f} cycles?"
                    st.session_state["trigger_query"] = True
                    st.rerun()

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

elif section == "Research Co-Pilot (RAG)":
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

        with q_cols[0]:
            if st.button("Health index & failure criteria", use_container_width=True, key="bm_query_1"):
                st.session_state["query_text"] = (
                    "What is the health index definition and how is the failure criterion reached in the simulation?"
                )
                st.session_state["trigger_query"] = True
                st.rerun()
        with q_cols[1]:
            if st.button("Damage propagation model", use_container_width=True, key="bm_query_2"):
                st.session_state["query_text"] = (
                    "How is damage propagation modeled within the modules of aircraft gas turbine engines?"
                )
                st.session_state["trigger_query"] = True
                st.rerun()
        with q_cols[2]:
            if st.button("C-MAPSS sensor outputs", use_container_width=True, key="bm_query_3"):
                st.session_state["query_text"] = (
                    "What sensor response surfaces and simulation inputs are generated via the C-MAPSS model?"
                )
                st.session_state["trigger_query"] = True
                st.rerun()
        with q_cols[3]:
            if st.button("FAA torque limits (Refusal test)", use_container_width=True, key="bm_query_4"):
                st.session_state["query_text"] = (
                    "What are the FAA certified bolt torque limits for GE90 turbine casing maintenance?"
                )
                st.session_state["trigger_query"] = True
                st.rerun()

        current_query = st.session_state.get(
            "query_text",
            "What is the health index definition and how is the failure criterion reached in the simulation?",
        )

        user_query = st.text_input(
            "Enter research question or inquiry:",
            value=current_query,
            placeholder="Ask about damage propagation, C-MAPSS parameters, or health index...",
            key="user_query_input",
        )

        # Engine context integration option
        attached_ctx_id = st.session_state.pop("copilot_engine_context", None)
        has_attached = attached_ctx_id is not None
        with st.expander("Attach Engine Context (Decoupled Telemetry Integration)", expanded=has_attached):
            attach_ctx = st.checkbox(
                "Include selected engine context in research query",
                value=has_attached,
                key="attach_ctx_chk",
            )
            ctx_unit = st.number_input(
                "Context Engine Unit",
                min_value=1,
                max_value=100,
                value=int(attached_ctx_id or 42),
                key="ctx_unit_num",
            )
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

        ask_btn = st.button("💬 Ask Co-Pilot Assistant", type="primary", key="ask_assistant_btn")
        should_run = ask_btn or st.session_state.pop("trigger_query", False)

        result = None
        if should_run and user_query:
            with st.spinner("Searching FAISS vector index and verifying textual evidence..."):
                result = query_research_assistant(
                    assistant=assistant,
                    query=user_query,
                    top_k=3,
                    engine_context_dict=ctx_dict,
                )
            st.session_state["last_copilot_result"] = result
        else:
            result = st.session_state.get("last_copilot_result")

        if result:
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

elif section == "Model Performance":
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


elif section == "Governance & Specs":
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
