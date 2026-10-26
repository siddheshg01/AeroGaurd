"""Semantic retrieval module using SentenceTransformers and FAISS index with source/page citations."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.phase3_config import (
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
)


class EmbeddingEngine:
    """Wrapper around SentenceTransformer with L2 normalization for cosine similarity."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode text or list of texts into L2-normalized float32 vectors.

        Args:
            texts: Single string or list of text passages.

        Returns:
            2D numpy array of shape (N, embedding_dim) with float32 dtype.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class RAGRetriever:
    """FAISS-backed vector retrieval engine for document chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingEngine | None = None,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self.embedding_engine = embedding_model or EmbeddingEngine(model_name)
        self.chunks: list[dict[str, Any]] = []
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int = EMBEDDING_DIM

    def build_from_chunks(self, chunks: list[dict[str, Any]]) -> "RAGRetriever":
        """Index a list of chunks into a FAISS Inner Product index.

        Args:
            chunks: List of chunk dictionaries containing 'text', 'doc_name', 'page_number', etc.

        Returns:
            self
        """
        if not chunks:
            raise ValueError("Cannot build RAG index from empty chunk list.")

        self.chunks = list(chunks)
        texts = [chunk["text"] for chunk in self.chunks]

        # Generate L2-normalized vectors
        vectors = self.embedding_engine.encode(texts)
        self.dimension = vectors.shape[1]

        # Use IndexFlatIP for exact cosine similarity of L2-normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        return self

    def save(self, index_path: Path | str, metadata_path: Path | str) -> None:
        """Serialize FAISS index and chunk metadata to disk.

        Args:
            index_path: File path for the binary FAISS index.
            metadata_path: File path for JSON chunk metadata.
        """
        if self.index is None:
            raise RuntimeError("Cannot save uninitialized index. Call build_from_chunks first.")

        idx_p = Path(index_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(idx_p))

        meta_p = Path(metadata_path)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "total_chunks": len(self.chunks),
            "chunks": self.chunks,
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        index_path: Path | str,
        metadata_path: Path | str,
        embedding_model: EmbeddingEngine | None = None,
    ) -> "RAGRetriever":
        """Load an indexed FAISS vector store and chunk metadata from disk.

        Args:
            index_path: Path to FAISS binary file.
            metadata_path: Path to JSON metadata file.
            embedding_model: Optional existing EmbeddingEngine to avoid re-instantiation.

        Returns:
            Reconstructed RAGRetriever instance.
        """
        idx_p = Path(index_path)
        meta_p = Path(metadata_path)

        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found: {idx_p}")
        if not meta_p.exists():
            raise FileNotFoundError(f"Index metadata file not found: {meta_p}")

        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_name = meta.get("model_name", EMBEDDING_MODEL_NAME)
        retriever = cls(
            embedding_model=embedding_model,
            model_name=model_name,
        )
        retriever.dimension = meta.get("dimension", EMBEDDING_DIM)
        retriever.chunks = meta.get("chunks", [])
        retriever.index = faiss.read_index(str(idx_p))

        return retriever

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Retrieve top relevant passages matching the query with citations.

        Args:
            query: User's search query or question.
            top_k: Number of highest-ranked passages to return.
            min_score: Minimum cosine similarity score threshold (0.0 to 1.0).

        Returns:
            List of retrieved passage dictionaries with source/page citations and scores.
        """
        cleaned_query = (query or "").strip()
        if not cleaned_query or self.index is None or len(self.chunks) == 0:
            return []

        # Encode single query vector (normalized)
        q_vec = self.embedding_engine.encode(cleaned_query)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            cos_score = float(score)
            if cos_score < min_score:
                continue

            chunk = self.chunks[idx]
            citation = f"[{chunk['doc_name']}, Page {chunk['page_number']}]"
            full_citation = (
                f"{chunk.get('doc_title', 'Damage Propagation Modeling')} "
                f"({chunk['doc_name']}, Page {chunk['page_number']}, Chunk {chunk['chunk_id']})"
            )

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_name": chunk["doc_name"],
                    "doc_title": chunk.get("doc_title", "Damage Propagation Modeling"),
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk.get("chunk_index", 1),
                    "text": chunk["text"],
                    "score": round(cos_score, 4),
                    "citation": citation,
                    "full_citation": full_citation,
                }
            )

        return results
