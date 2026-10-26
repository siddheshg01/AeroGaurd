# AeroGuard AI - Phase 1: Ingestion, Validation & Exploratory Data Analysis

AeroGuard AI is an intelligent predictive maintenance and health monitoring platform for turbofan aircraft engines, built on NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan engine degradation dataset.

**Phase 1 Focus:**
- Ingest raw time-series sensor degradation data (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
- Standardize the 26 operational and sensor measurement attributes.
- Execute data validation checks (shapes, engine fleet counts, missing values, duplicates, cycle ranges).
- Perform exploratory data analysis and generate high-resolution sensor degradation visualizations and reports into a dedicated `outputs/` directory.

---

## Project Structure

```text
AeroGaurd/
├── CMAPSSData/                         # Raw NASA C-MAPSS dataset (Read-only, untouched)
│   ├── Damage Propagation Modeling.pdf
│   ├── readme.txt
│   ├── RUL_FD001.txt
│   ├── test_FD001.txt
│   └── train_FD001.txt
├── src/                                # Core Phase 1 source code
│   ├── __init__.py
│   ├── config.py                       # Project paths, column schemas, sensor descriptions
│   ├── data_loader.py                  # Dataset ingestion & training RUL computation
│   ├── validator.py                    # Quality & integrity validation suite
│   ├── eda.py                          # Descriptive statistics, plots & interactive dashboard
│   └── run_phase1.py                   # Primary Phase 1 execution runner
├── tests/                              # Automated pytest suite
│   ├── __init__.py
│   └── test_phase1.py                  # 6 automated validation tests
├── outputs/                            # Generated artifacts (never touches CMAPSSData)
│   ├── figures/                        # High-resolution figures and interactive dashboard
│   │   ├── sensor_degradation_trajectories.png
│   │   ├── engine_lifespan_distribution.png
│   │   ├── sensor_correlation_with_rul.png
│   │   └── interactive_dashboard.html
│   └── reports/                        # Machine-readable and markdown summary reports
│       ├── data_validation_report.json
│       ├── data_validation_summary.md
│       ├── sensor_summary_statistics.csv
│       └── sensor_summary_statistics.md
├── .gitignore                          # Clean repository hygiene
├── requirements.txt                    # Python package dependencies
└── README.md                           # Documentation and setup instructions
```

---

## Dataset Schema (26 Columns)

Each row is an operational cycle snapshot of a turbofan engine:
1. `unit_number`: Unique engine identifier (1 to 100)
2. `cycle`: Operational flight cycle (starts at 1, monotonically increasing)
3. `setting_1`: Operational resolver setting 1
4. `setting_2`: Operational resolver setting 2
5. `setting_3`: Operational resolver setting 3
6. `sensor_1` to `sensor_21`: 21 continuous sensor measurements (temperatures, pressures, fan/core speeds, fuel ratios, coolant bleed).

---

## Windows PowerShell Setup & Execution

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Windows PowerShell

### 2. Environment Setup (Optional Virtual Environment)
```powershell
# Open Windows PowerShell in the project directory
cd C:\Users\Lenovo\Desktop\AeroGaurd

# Optional: Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run the Phase 1 Pipeline
To load data, run all validation checks, and generate all figures and reports:
```powershell
python src\run_phase1.py
```

### 4. Run Automated Pytest Suite
To execute the automated regression test suite:
```powershell
pytest -v tests\test_phase1.py
```

---

## Phase 1 Deliverables & Visualizations

All generated outputs are saved cleanly inside `outputs/`:

1. **`outputs/figures/sensor_degradation_trajectories.png`**:
   - 8-panel high-resolution trajectory plot showing key degradation sensors (LPC/HPC/LPT temperatures, HPC outlet/static pressures, fuel flow ratio, bypass ratio, coolant bleed) across sample fleet units up to their points of failure.
2. **`outputs/figures/engine_lifespan_distribution.png`**:
   - Combined histogram, KDE distribution, and boxplot illustrating fleet operational longevity (min: 128 cycles, median: 199 cycles, max: 362 cycles).
3. **`outputs/figures/sensor_correlation_with_rul.png`**:
   - Pearson correlation analysis identifying active wear indicators versus flatline/constant sensors under sea-level conditions.
4. **`outputs/figures/interactive_dashboard.html`**:
   - Interactive Plotly dashboard with hover tooltips and zoom capabilities for engine degradation analysis. Open directly in any web browser.
5. **`outputs/reports/data_validation_report.json` & `.md`**:
   - Full audit report detailing dataset dimensions, fleet counts, zero missing values, zero duplicates, and cycle monotonicity.
6. **`outputs/reports/sensor_summary_statistics.csv` & `.md`**:
   - Statistical variance and mean profiles for all 21 sensors.

---

## What's Next (Phase 2 Roadmap)
- Feature engineering: rolling statistics, exponential moving averages, degradation index.
- Health index formulation and baseline remaining useful life (RUL) regression modeling.
- RAG pipeline integration for aircraft maintenance manuals and fault diagnostic reasoning.

---

# Phase 2: Feature Engineering, RUL Preparation & Machine Learning

## Project Structure (Updated for Phase 2)

```text
AeroGaurd/
├── CMAPSSData/                             # Original NASA C-MAPSS dataset (Untouched)
├── src/
│   ├── config.py                           # Phase 1 paths and schema
│   ├── data_loader.py                      # Phase 1 data loading
│   ├── validator.py                        # Phase 1 data validation
│   ├── eda.py                              # Phase 1 EDA and plots
│   ├── run_phase1.py                       # Phase 1 pipeline runner
│   ├── phase2_config.py                    # Phase 2 config (RUL cap, sensors, split, RF params)
│   ├── feature_engineering.py             # RUL computation, rolling features, split, scaler
│   ├── trainer.py                          # Baseline + RF training, evaluation, plots
│   └── run_phase2.py                       # Phase 2 pipeline runner (9 steps)
├── tests/
│   ├── test_phase1.py                      # Phase 1 tests (6 tests)
│   └── test_phase2.py                      # Phase 2 tests (31 tests)
├── outputs/
│   ├── figures/
│   │   ├── phase2/
│   │   │   ├── validation_model_comparison.png
│   │   │   ├── test_engine_rul_predictions.png
│   │   │   ├── feature_importances.png
│   │   │   └── test_error_distribution.png
│   │   └── ...                             # Phase 1 figures
│   ├── models/
│   │   ├── mean_baseline.pkl
│   │   ├── random_forest.pkl
│   │   └── feature_pipeline.pkl
│   └── reports/
│       ├── phase2/
│       │   ├── phase2_evaluation_report.md
│       │   ├── random_forest_metrics.json
│       │   └── random_forest_predictions.csv
│       └── ...                             # Phase 1 reports
├── requirements.txt
└── README.md
```

## Phase 2: Feature Engineering Design

### RUL Target Computation
- **Formula**: `RUL(engine u, cycle t) = max_cycle(u) − t`
- **Final cycle**: RUL = 0 (confirmed failure in training set)
- **RUL Cap**: Configurable in `src/phase2_config.py` (`RUL_CAP = None` by default = no cap)
- **Enabling a cap**: Set `RUL_CAP = 125` (or any positive int) to clip high-RUL early-life cycles

### Constant Sensor Removal (from Phase 1 EDA)
Six sensors with **zero variance** (std < 1e-6) under FD001 sea-level conditions are dropped:
`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`
→ **15 informative sensors** remain.

### Features Generated (109 total)
| Group | Count | Description |
|---|---|---|
| Raw active sensors | 15 | Variable sensor readings |
| Operational settings | 3 | Setting 1/2/3 (single-condition, low variance) |
| Cycle normalisation | 1 | `cycle / max_cycle_per_engine` ∈ (0, 1] |
| Rolling mean (windows 5, 10, 20) | 45 | Per-engine backward rolling mean |
| Rolling std (windows 5, 10, 20) | 45 | Per-engine backward rolling std |
| **Total** | **109** | |

### Anti-Leakage Guarantees
1. **Rolling features** are computed engine-by-engine (`groupby("unit_number")`) in sorted (unit, cycle) order, using `min_periods=1` — no future cycles are ever accessed.
2. **Validation split** is by engine ID: engines 1–80 train, engines 81–100 validate. Zero engine overlap guaranteed.
3. **StandardScaler** is `fit()` exclusively on training rows (engines 1–80), then `transform()` is applied to validation and test.
4. **Test set** (`test_FD001.txt`) is loaded after model selection is complete.

## Phase 2: Evaluation Design

### Cycle-Level Validation (engines 81–100)
Predict RUL for **every cycle** in the 20 validation engines. Measures how well the model fits temporal degradation patterns.

### Engine-Level Test Evaluation (test_FD001.txt)
For each of the 100 test engines, predict RUL at the **final observed cycle only**. Compare against `RUL_FD001.txt` ground truth. This is the operationally relevant metric.

**These are intentionally different numbers** — do not compare them directly.

## Phase 2 Results (Actual Execution Output)

### Cycle-Level Validation (engines 81–100, 4,493 cycles)
| Model | MAE (cycles) | RMSE (cycles) |
|---|---|---|
| Mean Baseline | 64.09 | 78.93 |
| **Random Forest** | **27.19** | **39.21** |

RF reduces MAE by 57.6% versus the mean baseline.

### Engine-Level Test Evaluation (100 test engines, final cycle each)
| Metric | Value |
|---|---|
| Test MAE | **47.89 cycles** |
| Test RMSE | **56.62 cycles** |

The test MAE (47.89) is higher than validation MAE (27.19) because the test set evaluates at a single cut-off point per engine rather than across all observed cycles, and the difficulty of predicting RUL at an unknown terminal point is inherently higher than predicting across the full trajectory.

## Windows PowerShell Setup & Execution (Phase 2)

```powershell
# Navigate to the project directory
cd C:\Users\Lenovo\Desktop\AeroGaurd

# (Optional) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies (Phase 1 + Phase 2)
pip install -r requirements.txt

# Run Phase 1 (data validation and EDA)
python src\run_phase1.py

# Run Phase 2 (feature engineering, training, evaluation)
python src\run_phase2.py

# Run all tests (Phase 1: 6 tests, Phase 2: 31 tests = 37 total)
python -m pytest -v tests\

# Run only Phase 2 tests
python -m pytest -v tests\test_phase2.py

# Open interactive Phase 1 EDA dashboard in browser
Start-Process "outputs\figures\interactive_dashboard.html"

# Open Phase 2 evaluation report
Get-Content "outputs\reports\phase2\phase2_evaluation_report.md"
```

---

# Phase 3 — RAG Knowledge Base & Evidence-Grounded Research Assistant

## Overview
Phase 3 extends AeroGuard AI with a localized Retrieval-Augmented Generation (RAG) architecture and research assistant grounded in the foundational NASA C-MAPSS literature:
- **Research Document**: `CMAPSSData\Damage Propagation Modeling.pdf` (*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, Saxena et al., 2008)
- **Local Embedding Engine**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic vectors, no paid API key required)
- **Vector Database**: FAISS `IndexFlatIP` (Exact Inner Product with $L_2$-normalized vectors = Cosine Similarity)
- **Strict Evidence Grounding**: The assistant synthesizes responses strictly from retrieved passages. If adequate evidence is absent, it refuses to answer.
- **Safety Policy**: Does not hallucinate maintenance procedures, service limits, fault diagnoses, or airworthiness recommendations.

## Phase 3 Architecture & Pipeline

```
CMAPSSData/Damage Propagation Modeling.pdf (Read-Only)
               │
               ▼
   [src/pdf_extractor.py]
         pypdf extraction with page numbers & title preservation
               │
               ▼
   outputs/rag/extracted_document.json (9 pages, 41,115 characters)
               │
               ▼
       [src/chunker.py]
         Boundary-aligned sliding character windows (600 chars, 150 overlap)
               │
               ▼
   outputs/rag/text_chunks.json (101 overlapping chunks with chunk_id, page, doc_name)
               │
               ▼
    [src/rag_retriever.py]
         SentenceTransformer('all-MiniLM-L6-v2') + FAISS IndexFlatIP
               │
               ▼
   outputs/rag/faiss_index.bin + outputs/rag/index_metadata.json
               │
               ▼
   [src/research_assistant.py]
         Multi-tier verification:
           1. Semantic Similarity Threshold (>= 0.45)
           2. Lexical Concept Grounding Check
           3. Maintenance Safety Guardrail
           4. Exact Document & Page Citation Generation
           5. EngineContext Protocol for Decoupled Telemetry Integration
               │
               ▼
   outputs/reports/phase3/phase3_rag_report.md
   outputs/reports/phase3/rag_sample_qa.json
```

## Source Code & Modules

| Module | Location | Purpose |
|---|---|---|
| `phase3_config.py` | [`src/phase3_config.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/src/phase3_config.py) | Paths, chunking dimensions, embedding settings, thresholds, and safety policies |
| `pdf_extractor.py` | [`src/pdf_extractor.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/src/pdf_extractor.py) | Robust PDF extraction preserving page numbers (1-indexed), title, and clean text |
| `chunker.py` | [`src/chunker.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/src/chunker.py) | Sentence-boundary-aligned overlapping chunker retaining complete metadata |
| `rag_retriever.py` | [`src/rag_retriever.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/src/rag_retriever.py) | SentenceTransformer wrapper and FAISS vector index with load/save roundtrip |
| `research_assistant.py` | [`src/research_assistant.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/src/research_assistant.py) | Evidence-grounded QA assistant with strict refusal and `EngineContext` interface |
| `run_phase3.py` | [`src/run_phase3.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/src/run_phase3.py) | End-to-end executable pipeline with empirical benchmark evaluation |
| `test_phase3.py` | [`tests/test_phase3.py`](file:///c:/Users/Lenovo/Desktop/AeroGaurd/tests/test_phase3.py) | 26 unit and integration tests covering extraction, chunking, FAISS, and safety |

## Safety & Governance Policies
1. **Epistemic Refusal**: If a question cannot be answered using the research document (e.g. out-of-domain topics, unverified torque limits, emergency flight manuals), the system explicitly states:
   > *"The research document does not provide sufficient evidence to answer this question."*
2. **Prohibition on Maintenance Advice**: The system will not invent certified aircraft maintenance procedures, service limits, or airworthiness recommendations.
3. **Mandatory Disclaimer**: Every response attaches the following governance notice:
   > *DISCLAIMER: AeroGuard AI is an academic research prototype and does not provide certified aircraft maintenance procedures, service limits, fault diagnoses, or airworthiness recommendations. Always consult FAA/EASA certified Aircraft Maintenance Manuals (AMM) and authorized OEM engineering instructions.*
4. **Verifiable Citations**: Answers include exact citations: `[Damage Propagation Modeling.pdf, Page X]`.

## Decoupled Telemetry Integration Interface (`EngineContext`)
Phase 3 keeps the RAG pipeline completely independent from the Phase 2 machine learning predictive model, while providing a clean integration protocol:
```python
from src.research_assistant import EngineContext, ResearchAssistant

# Optional engine context populated from Phase 2 predictions / telemetry
context = EngineContext(
    unit_number=42,
    current_cycle=160,
    predicted_rul=35.4,
    health_status="Degrading",
    anomalous_sensors=["sensor_11", "sensor_12"],
)

response = assistant.ask(
    query="What degradation mechanisms affect High Pressure Compressor HPC flow and stall margin?",
    top_k=3,
    engine_context=context,
)
```

## Windows PowerShell Execution (Phase 3)

```powershell
cd C:\Users\Lenovo\Desktop\AeroGaurd

# Install dependencies (including pypdf, sentence-transformers, faiss-cpu, torch)
pip install -r requirements.txt

# Run the Phase 3 pipeline (extracts PDF, chunks text, builds FAISS index, tests QA)
python src\run_phase3.py

# Run Phase 3 tests (26 passed in ~19s)
python -m pytest -v tests\test_phase3.py

# Run the complete test suite across all phases (Phase 1: 6, Phase 2: 31, Phase 3: 26 = 63 tests)
python -m pytest -v tests\

# Inspect the Phase 3 RAG evaluation report
Get-Content "outputs\reports\phase3\phase3_rag_report.md"

# Inspect the sample QA evaluation JSON
Get-Content "outputs\reports\phase3\rag_sample_qa.json"
```

---

# Phase 4 — Integrated Streamlit Intelligence Dashboard

The dashboard combines validated FD001 dataset summaries, historical sensor analytics, inference from the saved Phase 2 model and feature pipeline, Phase 3 evidence-grounded research QA, and model governance details. It uses local artifacts and does not require a paid API. Sensor plots show historical NASA C-MAPSS simulation observations, never live aircraft telemetry. RUL outputs are research estimates and must not authorize maintenance or operational decisions.

## Launch on Windows PowerShell

```powershell
cd C:\Users\Lenovo\Desktop\AeroGaurd
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If model or RAG artifacts are absent, generate them with the existing phase runners before launching:

```powershell
python src\run_phase2.py
python src\run_phase3.py
```

Run the complete regression suite:

```powershell
python -m pytest -v tests\
```

Phase 4 keeps presentation in `app.py` and dashboard integration services in `src\dashboard_service.py`. It loads the existing Phase 2 model and preprocessing pipeline and the existing Phase 3 SentenceTransformer/FAISS index; it does not create substitute predictions, metrics, or evidence. The application is an academic research prototype, not a certified aircraft maintenance or airworthiness system.

