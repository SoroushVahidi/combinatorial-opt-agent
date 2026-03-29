"""
Tests for PDF text extraction (retrieval/pdf_utils.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pdf_utils import extract_text_from_pdf


def test_extract_text_from_pdf_none():
    assert extract_text_from_pdf(None) == ""
    assert extract_text_from_pdf("") == ""


def test_extract_text_from_pdf_missing_file():
    result = extract_text_from_pdf("/tmp/this_file_does_not_exist_xyz.pdf")
    assert "(Could not extract PDF text:" in result


def test_whitespace_collapsing():
    """Regression: extract_text_from_pdf must collapse whitespace (tabs, newlines, multiple
    spaces) down to a single space.  A bug where r'\\\\s+' was used instead of r'\\s+'
    caused the collapsing to silently do nothing."""
    pytest.importorskip("pypdf", reason="pypdf required to patch PdfReader for this test")
    # Build a fake PDF whose pages contain multi-whitespace text.
    page1 = MagicMock()
    page1.extract_text.return_value = "hello\t\tworld"
    page2 = MagicMock()
    page2.extract_text.return_value = "foo  \n  bar"

    fake_reader = MagicMock()
    fake_reader.pages = [page1, page2]

    with patch("pypdf.PdfReader", return_value=fake_reader):
        result = extract_text_from_pdf("any_path.pdf")

    # Tabs, newlines and runs of spaces must all collapse to a single space.
    assert result == "hello world foo bar"

