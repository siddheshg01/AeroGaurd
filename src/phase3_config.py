"""Phase 3 configuration: document paths, chunking, embeddings, FAISS and retrieval settings."""

from pathlib import Path
from src.config import DATA_DIR, OUTPUT_DIR

# ─── File & Directory Paths ──────────────────────────────────────────────────
# Raw research document (in CMAPSSData, read-only, never modified)
RESEARCH_DOC_PATH: Path = DATA_DIR / "Damage Propagation Modeling.pdf"

# Generated RAG artifacts (strictly outside CMAPSSData)
RAG_OUTPUT_DIR: Path = OUTPUT_DIR / "rag"
PHASE3_REPORTS_DIR: Path = OUTPUT_DIR / "reports" / "phase3"

# Extracted texts and chunk files
EXTRACTED_DATA_PATH: Path = RAG_OUTPUT_DIR / "extracted_document.json"
CHUNKS_DATA_PATH: Path = RAG_OUTPUT_DIR / "text_chunks.json"

# FAISS Vector Index and metadata mapping
FAISS_INDEX_PATH: Path = RAG_OUTPUT_DIR / "faiss_index.bin"
INDEX_METADATA_PATH: Path = RAG_OUTPUT_DIR / "index_metadata.json"

# Phase 3 evaluation reports
PHASE3_REPORT_PATH: Path = PHASE3_REPORTS_DIR / "phase3_rag_report.md"
PHASE3_QA_SAMPLES_PATH: Path = PHASE3_REPORTS_DIR / "rag_sample_qa.json"

# ─── Document Metadata Defaults ──────────────────────────────────────────────
DEFAULT_DOC_TITLE: str = (
    "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
)

# ─── Chunking Settings ───────────────────────────────────────────────────────
# Sliding character window with sentence/whitespace boundary awareness
CHUNK_SIZE: int = 600
CHUNK_OVERLAP: int = 150

# ─── Embedding & Retrieval Settings ──────────────────────────────────────────
# Local lightweight SentenceTransformer model (~90MB, 384 dimensions)
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# Default top-k retrieved chunks
DEFAULT_TOP_K: int = 3

# Minimum cosine similarity threshold (range [-1, 1], normalized dot product)
# Queries with top relevance score below this threshold are refused as unsupported
SIMILARITY_THRESHOLD: float = 0.45

# ─── Safety & Governance Policies ────────────────────────────────────────────
SYSTEM_DISCLAIMER: str = (
    "DISCLAIMER: AeroGuard AI is an academic research prototype and does not provide "
    "certified aircraft maintenance procedures, service limits, fault diagnoses, or "
    "airworthiness recommendations. Always consult FAA/EASA certified Aircraft Maintenance "
    "Manuals (AMM) and authorized OEM engineering instructions."
)

UNSUPPORTED_ANSWER_MESSAGE: str = (
    "The research document does not provide sufficient evidence to answer this question."
)
