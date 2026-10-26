"""PDF text extraction module with page numbering and document metadata preservation."""

import json
import re
from pathlib import Path
from typing import Any
import pypdf

from src.phase3_config import DEFAULT_DOC_TITLE


def clean_page_text(raw_text: str) -> str:
    """Normalize extracted page text while preserving sentence and paragraph structure.

    Args:
        raw_text: Raw string extracted from PDF page.

    Returns:
        Cleaned string with unified newlines and normalized whitespace.
    """
    if not raw_text:
        return ""

    # Replace non-standard whitespace and special control characters
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace weird form feeds or null characters
    text = text.replace("\x0c", "\n").replace("\x00", "")

    # Fix broken hyphenations across linebreaks (e.g. "degra-\ndation" -> "degradation")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Normalize multiple spaces per line but keep single newlines
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    # Rejoin lines, collapsing excessive blank lines (>2) to a single paragraph break
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def extract_pdf_pages(
    pdf_path: Path | str,
    default_title: str | None = None,
) -> dict[str, Any]:
    """Extract all text pages from a PDF document, preserving page numbers and title.

    Args:
        pdf_path: Path to the PDF file.
        default_title: Optional fallback title if PDF metadata lacks a valid title.

    Returns:
        Dictionary containing document name, title, total page count, and page list.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF or has 0 pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Specified path is not a file: {pdf_path}")

    fallback_title = default_title or DEFAULT_DOC_TITLE or pdf_path.stem

    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF document contains 0 pages: {pdf_path}")

    # Extract or infer document title
    doc_title = fallback_title
    if reader.metadata:
        meta_title = reader.metadata.title
        if meta_title and meta_title.strip() and meta_title.strip().lower() != "untitled":
            doc_title = meta_title.strip()

    pages: list[dict[str, Any]] = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-indexed page number
        raw_text = page.extract_text() or ""
        cleaned = clean_page_text(raw_text)

        pages.append(
            {
                "page_number": page_num,
                "text": cleaned,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()) if cleaned else 0,
            }
        )

    return {
        "doc_name": pdf_path.name,
        "doc_title": doc_title,
        "total_pages": total_pages,
        "source_path": str(pdf_path.resolve()),
        "pages": pages,
    }


def save_extracted_data(data: dict[str, Any], output_path: Path | str) -> Path:
    """Save extracted PDF data and metadata to a JSON file.

    Args:
        data: Extracted document dictionary.
        output_path: Target JSON file path.

    Returns:
        Resolved Path to the saved file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


def load_extracted_data(input_path: Path | str) -> dict[str, Any]:
    """Load extracted PDF data from a JSON file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Extracted document dictionary.
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Extracted data file not found: {inp}")
    with open(inp, "r", encoding="utf-8") as f:
        return json.load(f)
