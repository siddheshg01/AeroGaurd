"""Text chunking module with metadata preservation, overlapping windows, and boundary alignment."""

import json
import re
from pathlib import Path
from typing import Any

from src.phase3_config import CHUNK_OVERLAP, CHUNK_SIZE


def _find_clean_boundary(text: str, target_pos: int, window: int = 60) -> int:
    """Find a natural word or sentence boundary near target_pos to avoid cutting words.

    Args:
        text: Entire text block.
        target_pos: Desired cut position.
        window: Maximum characters to search backwards/forwards for a whitespace.

    Returns:
        Refined cut position.
    """
    if target_pos >= len(text):
        return len(text)

    # First look backwards for a period + space or newline (sentence boundary)
    search_start = max(0, target_pos - window)
    sentence_break = -1
    for sep in [". ", ".\n", "\n\n", "\n"]:
        pos = text.rfind(sep, search_start, target_pos)
        if pos != -1:
            sentence_break = max(sentence_break, pos + len(sep))

    if sentence_break > search_start:
        return sentence_break

    # Otherwise look backwards for any whitespace boundary
    space_pos = text.rfind(" ", search_start, target_pos)
    if space_pos > search_start:
        return space_pos + 1

    # Fallback: exact target position
    return target_pos


def chunk_page_text(
    page_text: str,
    page_number: int,
    doc_name: str,
    doc_title: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    start_chunk_idx: int = 1,
) -> list[dict[str, Any]]:
    """Chunk the text of a single page using overlapping windows.

    Args:
        page_text: Cleaned text of the page.
        page_number: 1-indexed page number.
        doc_name: Filename of the source document.
        doc_title: Human-readable document title.
        chunk_size: Approximate maximum character size per chunk.
        chunk_overlap: Number of characters to overlap between successive chunks.
        start_chunk_idx: Starting index for chunk numbering.

    Returns:
        List of chunk dictionaries with complete metadata.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be non-negative and less than chunk_size ({chunk_size})"
        )

    text = page_text.strip()
    if not text:
        return []

    # Clean document slug for chunk ID
    doc_slug = re.sub(r"[^a-zA-Z0-9]+", "_", doc_name.replace(".pdf", "")).strip("_").lower()

    # If entire page is smaller than or equal to chunk_size, return single chunk
    if len(text) <= chunk_size:
        chunk_id = f"{doc_slug}_p{page_number:02d}_c01"
        return [
            {
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "doc_title": doc_title,
                "page_number": page_number,
                "chunk_index": 1,
                "text": text,
                "char_start": 0,
                "char_end": len(text),
                "char_count": len(text),
                "word_count": len(text.split()),
            }
        ]

    chunks: list[dict[str, Any]] = []
    step = chunk_size - chunk_overlap
    start = 0
    chunk_idx = start_chunk_idx

    while start < len(text):
        ideal_end = start + chunk_size
        actual_end = _find_clean_boundary(text, ideal_end)

        chunk_str = text[start:actual_end].strip()
        if chunk_str:
            chunk_id = f"{doc_slug}_p{page_number:02d}_c{chunk_idx:02d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_name": doc_name,
                    "doc_title": doc_title,
                    "page_number": page_number,
                    "chunk_index": chunk_idx,
                    "text": chunk_str,
                    "char_start": start,
                    "char_end": actual_end,
                    "char_count": len(chunk_str),
                    "word_count": len(chunk_str.split()),
                }
            )
            chunk_idx += 1

        if actual_end >= len(text):
            break

        # Move forward so next chunk starts before actual_end by approximately chunk_overlap
        target_next_start = max(start + 1, actual_end - chunk_overlap)
        clean_next_start = _find_clean_boundary(
            text, target_next_start, window=max(15, chunk_overlap // 2)
        )
        if clean_next_start <= start or clean_next_start >= actual_end:
            start = target_next_start
        else:
            start = clean_next_start

    return chunks


def chunk_document(
    doc_data: dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split an entire extracted document dictionary into overlapping chunks.

    Args:
        doc_data: Dictionary returned by extract_pdf_pages().
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of all chunk dictionaries across all pages.
    """
    doc_name = doc_data.get("doc_name", "unknown_doc.pdf")
    doc_title = doc_data.get("doc_title", "Unknown Title")
    pages = doc_data.get("pages", [])

    all_chunks: list[dict[str, Any]] = []
    for page in pages:
        page_num = page.get("page_number", 1)
        page_text = page.get("text", "")
        page_chunks = chunk_page_text(
            page_text=page_text,
            page_number=page_num,
            doc_name=doc_name,
            doc_title=doc_title,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


def save_chunks(chunks: list[dict[str, Any]], output_path: Path | str) -> Path:
    """Save chunk list to JSON file.

    Args:
        chunks: List of chunk dictionaries.
        output_path: Target path.

    Returns:
        Resolved Path to the saved file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return out


def load_chunks(input_path: Path | str) -> list[dict[str, Any]]:
    """Load chunks from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        List of chunk dictionaries.
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Chunks file not found: {inp}")
    with open(inp, "r", encoding="utf-8") as f:
        return json.load(f)
