"""Evidence-grounded research assistant providing retrieval-based QA with strict citations and safety disclaimers."""

import re
from dataclasses import dataclass, field
from typing import Any

from src.config import SENSOR_DESCRIPTIONS
from src.phase3_config import (
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    SYSTEM_DISCLAIMER,
    UNSUPPORTED_ANSWER_MESSAGE,
)
from src.rag_retriever import RAGRetriever


@dataclass
class EngineContext:
    """Optional telemetry/prediction context for integrating with predictive models."""

    unit_number: int | None = None
    current_cycle: int | None = None
    predicted_rul: float | None = None
    health_status: str | None = None
    anomalous_sensors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "unit_number": self.unit_number,
            "current_cycle": self.current_cycle,
            "predicted_rul": self.predicted_rul,
            "health_status": self.health_status,
            "anomalous_sensors": list(self.anomalous_sensors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EngineContext":
        """Instantiate EngineContext safely from dict."""
        if not data:
            return cls()
        return cls(
            unit_number=data.get("unit_number"),
            current_cycle=data.get("current_cycle"),
            predicted_rul=data.get("predicted_rul"),
            health_status=data.get("health_status"),
            anomalous_sensors=data.get("anomalous_sensors") or [],
        )


class ResearchAssistant:
    """Evidence-grounded research assistant adhering to strict refusal and citation rules."""

    def __init__(
        self,
        retriever: RAGRetriever,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        disclaimer: str = SYSTEM_DISCLAIMER,
    ) -> None:
        """Initialize assistant with retriever and safety configuration.

        Args:
            retriever: Configured RAGRetriever instance.
            similarity_threshold: Minimum cosine score to consider evidence adequate.
            disclaimer: Mandated safety disclaimer.
        """
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.disclaimer = disclaimer

    def _extract_key_sentences(self, query: str, passages: list[dict[str, Any]], max_sentences: int = 3) -> list[str]:
        """Extract the most relevant sentences from retrieved passages matching query keywords.

        Args:
            query: Question asked.
            passages: Retrieved passage dictionaries.
            max_sentences: Maximum number of sentences to extract.

        Returns:
            List of relevant sentence strings.
        """
        # Tokenize query keywords (excluding common stop words)
        stop_words = {
            "what", "is", "the", "and", "or", "in", "of", "for", "a", "an", "to", "how",
            "are", "do", "does", "did", "can", "could", "would", "should", "which", "where",
            "when", "why", "who", "with", "at", "by", "from", "on", "this", "that", "these",
        }
        tokens = set(re.findall(r"\w+", query.lower())) - stop_words

        candidate_sentences: list[tuple[float, str]] = []
        seen = set()

        for passage in passages:
            text = passage.get("text", "")
            # Split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                clean_s = s.strip()
                if len(clean_s) < 25 or clean_s in seen:
                    continue
                seen.add(clean_s)
                # Compute token overlap score
                s_tokens = set(re.findall(r"\w+", clean_s.lower()))
                overlap = len(tokens & s_tokens)
                if overlap > 0:
                    candidate_sentences.append((overlap, clean_s))

        # Sort by overlap descending
        candidate_sentences.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in candidate_sentences[:max_sentences]]

    def _has_adequate_lexical_grounding(self, query: str, passages: list[dict[str, Any]]) -> bool:
        """Verify that informative query concepts are actually grounded in retrieved evidence text.

        Dense embeddings can produce moderate similarity on generic domain words (e.g. 'turbine', 'engine').
        This check ensures that specific query nouns/predicates (e.g. 'torque', 'bolt', 'restart')
        actually exist in the candidate text before an answer is formulated.
        """
        stop_words = {
            "what", "is", "the", "and", "or", "in", "of", "for", "a", "an", "to", "how",
            "are", "do", "does", "did", "can", "could", "would", "should", "which", "where",
            "when", "why", "who", "with", "at", "by", "from", "on", "this", "that", "these",
            "provide", "give", "tell", "explain", "about", "describe", "step", "steps",
        }
        tokens = {t.lower() for t in re.findall(r"\b[a-zA-Z]{3,}\b", query)} - stop_words
        if not tokens:
            return True

        combined_text = " ".join(p.get("text", "").lower() for p in passages)
        matched = {t for t in tokens if t in combined_text}

        # If query has 3 or more distinctive keywords, require at least 2 distinct matches
        # and at least 35% overlap
        if len(tokens) >= 3:
            return len(matched) >= 2 and (len(matched) / len(tokens)) >= 0.35
        return len(matched) >= 1

    def ask(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        engine_context: EngineContext | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer a question strictly using evidence retrieved from the research document.

        If no adequate evidence is retrieved or the query seeks unverified maintenance
        actions/procedures, clearly states that the document does not provide enough information.

        Args:
            query: User's query or question.
            top_k: Number of candidate passages to retrieve.
            engine_context: Optional EngineContext or dict for model integration.

        Returns:
            Structured answer dictionary containing:
              - question
              - status: "answered" or "insufficient_evidence"
              - answer: grounded synthesis or refusal message
              - answer_mode: "retrieval_grounded_extractive"
              - top_similarity_score: float
              - citations: list of unique human-readable citations
              - retrieved_evidence: list of retrieved chunk passages
              - disclaimer: research prototype disclaimer
              - engine_context: context if supplied
        """
        clean_query = (query or "").strip()
        context_obj = (
            engine_context
            if isinstance(engine_context, EngineContext)
            else EngineContext.from_dict(engine_context)
        )

        if not clean_query:
            return {
                "question": query,
                "status": "insufficient_evidence",
                "answer": "No question was provided.",
                "answer_mode": "retrieval_grounded_extractive",
                "top_similarity_score": 0.0,
                "citations": [],
                "retrieved_evidence": [],
                "disclaimer": self.disclaimer,
                "engine_context": context_obj.to_dict() if context_obj.unit_number else None,
            }

        # Formulate search query (optionally incorporating anomalous sensor context if relevant)
        retrieval_query = clean_query
        grounding_query = clean_query
        if context_obj.anomalous_sensors:
            hints = []
            for s in context_obj.anomalous_sensors[:3]:
                desc = SENSOR_DESCRIPTIONS.get(s, s)
                clean_desc = re.sub(r"\[.*?\]", "", desc).strip()
                hints.append(clean_desc)
            if hints:
                sensor_hint = " ".join(hints)
                retrieval_query = f"{clean_query} {sensor_hint}"
                grounding_query = f"{clean_query} {sensor_hint}"

        passages = self.retriever.search(
            query=retrieval_query,
            top_k=top_k,
            min_score=0.0,
        )

        # Check if we have any passages and if the top passage meets the threshold
        top_score = passages[0]["score"] if passages else 0.0
        has_adequate_grounding = self._has_adequate_lexical_grounding(grounding_query, passages)

        if not passages or top_score < self.similarity_threshold or not has_adequate_grounding:
            return {
                "question": clean_query,
                "status": "insufficient_evidence",
                "answer": f"{UNSUPPORTED_ANSWER_MESSAGE}\n\n{self.disclaimer}",
                "answer_mode": "retrieval_grounded_extractive",
                "top_similarity_score": top_score,
                "citations": [],
                "retrieved_evidence": [],
                "disclaimer": self.disclaimer,
                "engine_context": context_obj.to_dict() if context_obj.unit_number else None,
            }

        # Filter passages that meet threshold
        valid_passages = [p for p in passages if p["score"] >= self.similarity_threshold]
        if not valid_passages:
            valid_passages = passages[:1]

        # Extract distinct citations
        citations = []
        for p in valid_passages:
            cite = p["citation"]
            if cite not in citations:
                citations.append(cite)

        # Extract key sentences from evidence
        key_sentences = self._extract_key_sentences(clean_query, valid_passages, max_sentences=3)

        # Construct grounded answer
        answer_parts = []
        if key_sentences:
            evidence_summary = " ".join(key_sentences)
            answer_parts.append(
                f"Based on the research document ({', '.join(citations)}):\n\n"
                f"{evidence_summary}"
            )
        else:
            # Fall back to top passage primary excerpt
            top_passage_snippet = valid_passages[0]["text"][:350].strip()
            answer_parts.append(
                f"According to {valid_passages[0]['citation']}:\n\n"
                f'"{top_passage_snippet}..."'
            )

        # Explicitly quote primary evidence source
        primary_chunk = valid_passages[0]
        answer_parts.append(
            f"\n\n**Primary Evidence [{primary_chunk['doc_name']}, Page {primary_chunk['page_number']} "
            f"(Relevance: {primary_chunk['score']:.2f})]:**\n"
            f"> {primary_chunk['text'][:400]}..."
        )

        answer_parts.append(f"\n\n{self.disclaimer}")
        full_answer = "".join(answer_parts)

        return {
            "question": clean_query,
            "status": "answered",
            "answer": full_answer,
            "answer_mode": "retrieval_grounded_extractive",
            "top_similarity_score": top_score,
            "citations": citations,
            "retrieved_evidence": valid_passages,
            "disclaimer": self.disclaimer,
            "engine_context": context_obj.to_dict() if context_obj.unit_number else None,
        }
