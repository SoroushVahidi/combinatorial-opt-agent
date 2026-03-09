"""
Tests for PDF text extraction (retrieval/pdf_utils.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pdf_utils import extract_text_from_pdf


def test_extract_text_from_pdf_none():
    assert extract_text_from_pdf(None) == ""
    assert extract_text_from_pdf("") == ""


def test_extract_text_from_pdf_missing_file():
    result = extract_text_from_pdf("/tmp/this_file_does_not_exist_xyz.pdf")
    assert "(Could not extract PDF text:" in result

