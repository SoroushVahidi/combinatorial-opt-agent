"""
PDF text extraction utilities for the combinatorial-opt-agent.

This module is intentionally dependency-light (only ``pypdf``) so it can be
imported and tested without pulling in Gradio or FastAPI.
"""
from __future__ import annotations

import re


def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a PDF file and return it as a single string.

    Pages are joined with a single space and runs of whitespace are collapsed
    so the result is suitable as a search query.  Returns an empty string when
    *file_path* is falsy.  Returns a descriptive error string (starting with
    ``"(Could not extract PDF text:"`` ) if the file cannot be parsed.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the PDF file on disk.  This is the
        value provided by the Gradio ``gr.File`` component when
        ``type="filepath"`` is used.
    """
    if not file_path:
        return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
        combined = " ".join(p for p in pages if p)
        # Collapse runs of whitespace (tabs, newlines, multiple spaces) to a single space.
        return re.sub(r"\s+", " ", combined).strip()
    except Exception as exc:
        return f"(Could not extract PDF text: {exc})"
