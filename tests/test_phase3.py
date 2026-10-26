"""Unit and integration test suite for AeroGuard AI Phase 3 RAG Knowledge Base."""

import json
from pathlib import Path
import pytest
import numpy as np

from src.phase3_config import (
    CHUNKS_DATA_PATH,
    DEFAULT_DOC_TITLE,
    EMBEDDING_MODEL_NAME,
    EXTRACTED_DATA_PATH,
    FAISS_INDEX_PATH,
    INDEX_METADATA_PATH,
    RESEARCH_DOC_PATH,
    SIMILARITY_THRESHOLD,
    SYSTEM_DISCLAIMER,
    UNSUPPORTED_ANSWER_MESSAGE,
)
from src.pdf_extractor import (
    clean_page_text,
    extract_pdf_pages,
    load_extracted_data,
    save_extracted_data,
)
from src.chunker import (
    chunk_document,
    chunk_page_text,
    load_chunks,
    save_chunks,
)
from src.rag_retriever import EmbeddingEngine, RAGRetriever
from src.research_assistant import EngineContext, ResearchAssistant


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shared_embedding_engine():
    """Load EmbeddingEngine once per test session to optimize test run time."""
    return EmbeddingEngine(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def extracted_doc():
    """Extract research document once for inspection tests."""
    return extract_pdf_pages(RESEARCH_DOC_PATH, default_title=DEFAULT_DOC_TITLE)


@pytest.fixture(scope="module")
def document_chunks(extracted_doc):
    """Chunk research document once for chunking and retrieval tests."""
    return chunk_document(extracted_doc)


@pytest.fixture(scope="module")
def live_retriever(shared_embedding_engine):
    """Load or build the FAISS retriever for the real document."""
    if FAISS_INDEX_PATH.exists() and INDEX_METADATA_PATH.exists():
        return RAGRetriever.load(
            index_path=FAISS_INDEX_PATH,
            metadata_path=INDEX_METADATA_PATH,
            embedding_model=shared_embedding_engine,
        )
    # Build dynamically if not yet on disk
    doc_data = extract_pdf_pages(RESEARCH_DOC_PATH)
    chunks = chunk_document(doc_data)
    r = RAGRetriever(embedding_model=shared_embedding_engine)
    r.build_from_chunks(chunks)
    return r


@pytest.fixture(scope="module")
def assistant(live_retriever):
    """Instantiate ResearchAssistant once per test session."""
    return ResearchAssistant(
        retriever=live_retriever,
        similarity_threshold=SIMILARITY_THRESHOLD,
        disclaimer=SYSTEM_DISCLAIMER,
    )


# ─── 1. PDF Extraction Tests ─────────────────────────────────────────────────

class TestPDFExtraction:
    """Test suite for PDF loading, text cleaning, page extraction, and metadata."""

    def test_pdf_file_exists_and_non_empty(self):
        """Verify the raw research PDF exists in CMAPSSData and has non-zero size."""
        assert RESEARCH_DOC_PATH.exists(), f"Missing PDF at: {RESEARCH_DOC_PATH}"
        assert RESEARCH_DOC_PATH.stat().st_size > 0, "PDF file is empty"

    def test_extract_pdf_pages_structure(self, extracted_doc):
        """Verify extract_pdf_pages returns expected metadata schema."""
        assert extracted_doc["doc_name"] == "Damage Propagation Modeling.pdf"
        assert extracted_doc["doc_title"] == DEFAULT_DOC_TITLE
        assert extracted_doc["total_pages"] == 9
        assert len(extracted_doc["pages"]) == 9

    def test_page_numbers_are_one_indexed_and_sequential(self, extracted_doc):
        """Verify pages are 1-indexed from 1 to 9 without gaps."""
        page_numbers = [p["page_number"] for p in extracted_doc["pages"]]
        assert page_numbers == list(range(1, 10))

    def test_extracted_pages_content(self, extracted_doc):
        """Verify all pages contain non-trivial extracted text."""
        for page in extracted_doc["pages"]:
            assert page["char_count"] > 500, f"Page {page['page_number']} text too short"
            assert page["word_count"] > 50, f"Page {page['page_number']} word count too low"
            assert len(page["text"]) == page["char_count"]

    def test_text_cleaning_normalizes_hyphens_and_whitespace(self):
        """Verify clean_page_text handles hyphens across newlines and multiple spaces."""
        raw = "degra-\ndation of tur-\nbofan engines   with    spaces.\r\n\r\n\r\n\r\nNew paragraph."
        cleaned = clean_page_text(raw)
        assert "degradation" in cleaned
        assert "turbofan" in cleaned
        assert "   " not in cleaned
        assert "\r" not in cleaned
        assert "\n\n\n" not in cleaned

    def test_save_and_load_extracted_data_roundtrip(self, extracted_doc, tmp_path):
        """Verify extracted document JSON serialization roundtrip."""
        out_file = tmp_path / "test_extracted.json"
        saved_path = save_extracted_data(extracted_doc, out_file)
        assert saved_path.exists()

        loaded = load_extracted_data(saved_path)
        assert loaded["doc_name"] == extracted_doc["doc_name"]
        assert loaded["total_pages"] == extracted_doc["total_pages"]
        assert len(loaded["pages"]) == len(extracted_doc["pages"])

    def test_missing_pdf_raises_error(self):
        """Verify FileNotFoundError is raised when file does not exist."""
        with pytest.raises(FileNotFoundError):
            extract_pdf_pages("non_existent_document_12345.pdf")


# ─── 2. Text Chunking Tests ──────────────────────────────────────────────────

class TestTextChunking:
    """Test suite for overlapping chunking, metadata retention, and boundary alignment."""

    def test_chunk_attributes_present(self, document_chunks):
        """Verify every chunk retains required metadata: doc_name, page_number, chunk_id, etc."""
        assert len(document_chunks) > 0
        for chunk in document_chunks:
            assert "chunk_id" in chunk
            assert "doc_name" in chunk
            assert "doc_title" in chunk
            assert "page_number" in chunk
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "char_start" in chunk
            assert "char_end" in chunk
            assert chunk["doc_name"] == "Damage Propagation Modeling.pdf"
            assert 1 <= chunk["page_number"] <= 9
            assert len(chunk["text"]) > 0

    def test_chunk_ids_are_unique(self, document_chunks):
        """Verify every chunk has a distinct deterministic chunk_id."""
        ids = [c["chunk_id"] for c in document_chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_overlap_synthetic(self):
        """Verify consecutive synthetic chunks exhibit proper overlapping content."""
        synthetic_text = (
            "Sentence one describes aircraft turbofan engines. "
            "Sentence two discusses High Pressure Compressor degradation. "
            "Sentence three models flow capacity loss and efficiency reduction. "
            "Sentence four evaluates the remaining useful life margins. "
            "Sentence five concludes the damage propagation simulation methodology."
        )
        chunks = chunk_page_text(
            page_text=synthetic_text,
            page_number=1,
            doc_name="synthetic.pdf",
            doc_title="Synthetic Test Document",
            chunk_size=120,
            chunk_overlap=40,
        )
        assert len(chunks) >= 2
        # Check that consecutive chunks share text
        for i in range(len(chunks) - 1):
            curr_text = chunks[i]["text"]
            next_text = chunks[i + 1]["text"]
            # Look for words from the end of curr_text inside next_text
            curr_words = curr_text.split()[-3:]
            shared = [w for w in curr_words if w in next_text]
            assert len(shared) > 0, f"Chunks {i} and {i+1} do not appear to overlap"

    def test_short_text_produces_single_chunk(self):
        """Verify text smaller than chunk_size produces exactly one chunk."""
        short_text = "Brief turbofan note."
        chunks = chunk_page_text(
            page_text=short_text,
            page_number=1,
            doc_name="test.pdf",
            doc_title="Test",
            chunk_size=500,
            chunk_overlap=100,
        )
        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text
        assert chunks[0]["chunk_index"] == 1

    def test_empty_text_produces_zero_chunks(self):
        """Verify empty page produces an empty list."""
        chunks = chunk_page_text(
            page_text="",
            page_number=1,
            doc_name="test.pdf",
            doc_title="Test",
        )
        assert len(chunks) == 0

    def test_invalid_chunk_arguments_raise_value_error(self):
        """Verify validation errors for bad chunk_size or chunk_overlap."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_page_text("test", 1, "doc.pdf", "Title", chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap .* must be non-negative"):
            chunk_page_text("test", 1, "doc.pdf", "Title", chunk_size=100, chunk_overlap=120)

    def test_save_and_load_chunks_roundtrip(self, document_chunks, tmp_path):
        """Verify chunks JSON serialization roundtrip."""
        out_file = tmp_path / "test_chunks.json"
        saved = save_chunks(document_chunks, out_file)
        assert saved.exists()

        loaded = load_chunks(saved)
        assert len(loaded) == len(document_chunks)
        assert loaded[0]["chunk_id"] == document_chunks[0]["chunk_id"]


# ─── 3. Semantic Retrieval & Vector Store Tests ──────────────────────────────

class TestRAGRetriever:
    """Test suite for Sentence Transformers embeddings, FAISS indexing, and retrieval."""

    def test_synthetic_corpus_retrieval(self, shared_embedding_engine):
        """Verify semantic retrieval ranks the most relevant passage highest on synthetic inputs."""
        synthetic_chunks = [
            {
                "chunk_id": "c1",
                "doc_name": "propulsion.pdf",
                "doc_title": "Jet Propulsion",
                "page_number": 2,
                "text": "The high pressure compressor suffers from blade tip clearance wear and flow loss.",
            },
            {
                "chunk_id": "c2",
                "doc_name": "avionics.pdf",
                "doc_title": "Avionics Overview",
                "page_number": 5,
                "text": "Fly-by-wire digital flight control computers manage aileron and rudder surfaces.",
            },
            {
                "chunk_id": "c3",
                "doc_name": "cabin.pdf",
                "doc_title": "Cabin Systems",
                "page_number": 1,
                "text": "Environmental air conditioning systems regulate passenger cabin temperature and oxygen.",
            },
        ]
        retriever = RAGRetriever(embedding_model=shared_embedding_engine)
        retriever.build_from_chunks(synthetic_chunks)

        results = retriever.search("compressor blade wear and efficiency", top_k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["doc_name"] == "propulsion.pdf"
        assert results[0]["page_number"] == 2
        assert results[0]["score"] > results[1]["score"]

    def test_empty_query_returns_empty_results(self, live_retriever):
        """Verify empty and whitespace queries return empty lists without crashing."""
        assert live_retriever.search("") == []
        assert live_retriever.search("   ") == []
        assert live_retriever.search(None) == []

    def test_citations_present_in_retrieval_output(self, live_retriever):
        """Verify citation fields are properly formatted with document name and page number."""
        results = live_retriever.search("damage propagation in turbofan engines", top_k=3)
        assert len(results) > 0
        for r in results:
            assert "citation" in r
            assert "full_citation" in r
            assert "Damage Propagation Modeling.pdf" in r["citation"]
            assert "Page" in r["citation"]
            assert 0.0 <= r["score"] <= 1.0

    def test_top_k_parameter_enforced(self, live_retriever):
        """Verify top_k parameter strictly controls number of returned passages."""
        for k in [1, 2, 4]:
            results = live_retriever.search("health index", top_k=k)
            assert len(results) <= k

    def test_retriever_save_and_load_roundtrip(self, shared_embedding_engine, tmp_path):
        """Verify FAISS binary index and metadata JSON save and load correctly."""
        mini_chunks = [
            {
                "chunk_id": "m1",
                "doc_name": "test.pdf",
                "doc_title": "Test Doc",
                "page_number": 1,
                "text": "C-MAPSS turbofan simulation generates run to failure datasets.",
            }
        ]
        r1 = RAGRetriever(embedding_model=shared_embedding_engine)
        r1.build_from_chunks(mini_chunks)

        idx_path = tmp_path / "faiss.bin"
        meta_path = tmp_path / "meta.json"
        r1.save(idx_path, meta_path)

        assert idx_path.exists()
        assert meta_path.exists()

        r2 = RAGRetriever.load(idx_path, meta_path, embedding_model=shared_embedding_engine)
        assert len(r2.chunks) == 1
        assert r2.chunks[0]["chunk_id"] == "m1"
        res = r2.search("run to failure data", top_k=1)
        assert len(res) == 1
        assert res[0]["chunk_id"] == "m1"


# ─── 4. Research Assistant & Epistemic Grounding Tests ───────────────────────

class TestResearchAssistant:
    """Test suite for question answering, grounded synthesis, refusal, and safety policies."""

    def test_supported_domain_question_is_answered(self, assistant):
        """Verify domain questions covered by the research paper are answered with citations."""
        query = "What is the health index definition and how is the failure criterion reached in the simulation?"
        res = assistant.ask(query, top_k=3)

        assert res["status"] == "answered"
        assert res["answer_mode"] == "retrieval_grounded_extractive"
        assert res["top_similarity_score"] >= SIMILARITY_THRESHOLD
        assert len(res["citations"]) > 0
        assert len(res["retrieved_evidence"]) > 0
        assert "Damage Propagation Modeling.pdf" in res["citations"][0]
        # Verify answer text includes citation reference and key facts
        assert "health index" in res["answer"].lower()
        assert SYSTEM_DISCLAIMER in res["answer"]

    def test_unsupported_maintenance_procedure_is_refused(self, assistant):
        """Verify unverified maintenance procedures / bolt torque specifications are cleanly refused."""
        unsupported_query = "What are the FAA certified bolt torque limits for GE90 turbine casing maintenance?"
        res = assistant.ask(unsupported_query)

        assert res["status"] == "insufficient_evidence"
        assert UNSUPPORTED_ANSWER_MESSAGE in res["answer"]
        assert len(res["citations"]) == 0
        assert len(res["retrieved_evidence"]) == 0
        assert SYSTEM_DISCLAIMER in res["answer"]

    def test_unsupported_emergency_procedure_is_refused(self, assistant):
        """Verify operational emergency procedures not in the paper are refused."""
        emergency_query = "Provide step-by-step emergency instructions for an in-flight engine shutdown and restart."
        res = assistant.ask(emergency_query)

        assert res["status"] == "insufficient_evidence"
        assert UNSUPPORTED_ANSWER_MESSAGE in res["answer"]
        assert len(res["citations"]) == 0

    def test_unrelated_out_of_domain_query_is_refused(self, assistant):
        """Verify completely unrelated query is refused."""
        unrelated_query = "What is the recipe for baking chocolate chip cookies?"
        res = assistant.ask(unrelated_query)

        assert res["status"] == "insufficient_evidence"
        assert res["top_similarity_score"] < SIMILARITY_THRESHOLD
        assert len(res["citations"]) == 0

    def test_empty_query_handling(self, assistant):
        """Verify empty and whitespace inputs return clean insufficient_evidence status."""
        res = assistant.ask("")
        assert res["status"] == "insufficient_evidence"
        assert "No question was provided." in res["answer"]

    def test_disclaimer_present_in_every_response(self, assistant):
        """Verify the mandatory research prototype disclaimer is always present."""
        q1_res = assistant.ask("How is damage propagation modeled?")
        assert SYSTEM_DISCLAIMER in q1_res["answer"]

        q2_res = assistant.ask("Invalid query 1234567890?")
        assert SYSTEM_DISCLAIMER in q2_res["answer"]

    def test_engine_context_decoupled_integration(self, assistant):
        """Verify EngineContext interface accepts telemetry without coupling to model internals."""
        context = EngineContext(
            unit_number=10,
            current_cycle=120,
            predicted_rul=45.2,
            health_status="Degrading",
            anomalous_sensors=["sensor_11", "sensor_12"],
        )
        res = assistant.ask(
            query="How does High Pressure Compressor HPC degradation affect stall margin?",
            top_k=2,
            engine_context=context,
        )

        assert res["engine_context"] is not None
        assert res["engine_context"]["unit_number"] == 10
        assert res["engine_context"]["current_cycle"] == 120
        assert res["engine_context"]["predicted_rul"] == 45.2
        assert res["engine_context"]["anomalous_sensors"] == ["sensor_11", "sensor_12"]
        assert res["status"] == "answered"
