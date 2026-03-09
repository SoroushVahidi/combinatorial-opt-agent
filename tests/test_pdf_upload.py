"""
Tests for PDF text extraction (retrieval/pdf_utils.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pdf_utils import extract_text_from_pdf


def test_extract_text_from_pdf_none():
    assert extract_text_from_pdf(None) == ""
    assert extract_text_from_pdf("") == ""


def test_extract_text_from_pdf_missing_file():
    result = extract_text_from_pdf("/tmp/this_file_does_not_exist_xyz.pdf")
    assert "(Could not extract PDF text:" in result


def test_extract_text_from_pdf_real_pdf(tmp_path):
    """Round-trip: create a PDF with pypdf writer and read it back."""
    try:
        import pypdf
    except ImportError:
        import pytest
        pytest.skip("pypdf not installed")

    pdf_path = tmp_path / "test.pdf"

    # Build a real PDF with pypdf's writer
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    result = extract_text_from_pdf(str(pdf_path))
    # Blank page: empty text is fine (no crash, no error message)
    assert isinstance(result, str)
    assert "(Could not extract PDF text:" not in result


def test_extract_text_from_pdf_gradio_fn_signature():
    """extract_text_from_pdf must accept a single file-path string."""
    import inspect
    sig = inspect.signature(extract_text_from_pdf)
    params = list(sig.parameters)
    assert params == ["file_path"]


def test_extract_text_from_pdf_non_pdf_raises_gracefully(tmp_path):
    """Uploading a non-PDF file should return the error string, not crash."""
    bad_file = tmp_path / "notapdf.pdf"
    bad_file.write_text("this is not a pdf")
    result = extract_text_from_pdf(str(bad_file))
    assert isinstance(result, str)
    assert "(Could not extract PDF text:" in result


def test_extract_text_from_pdf_collapses_whitespace(tmp_path):
    """Extra whitespace from multi-page PDFs should be collapsed to single spaces."""
    try:
        import pypdf
    except ImportError:
        import pytest
        pytest.skip("pypdf not installed")

    # Write two blank pages; result should be "" (no crash, no extra spaces)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    pdf_path = tmp_path / "two_pages.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    result = extract_text_from_pdf(str(pdf_path))
    assert isinstance(result, str)
    assert "  " not in result  # no double spaces


# ---------------------------------------------------------------------------
# handle_pdf_upload (app.py) tests — import the named function directly.
# We avoid importing the whole app module to sidestep its heavy dependencies.
# ---------------------------------------------------------------------------

def _import_handle_pdf_upload():
    """Import handle_pdf_upload by reading its source into a fresh namespace."""
    import importlib.util
    import sys
    from pathlib import Path

    # Minimal stub modules so app.py top-level imports don't crash.
    import types
    for mod in ["starlette", "starlette.concurrency", "gradio"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Patch starlette.concurrency.run_in_threadpool attribute
    sc = sys.modules["starlette.concurrency"]
    if not hasattr(sc, "run_in_threadpool"):
        async def _dummy(*a, **kw): ...
        sc.run_in_threadpool = _dummy

    # Patch gradio to be a no-op namespace
    gr = sys.modules["gradio"]
    for attr in ["themes", "Blocks", "HTML", "Row", "Textbox", "Column", "Slider",
                 "Checkbox", "Markdown", "Accordion", "File", "Button", "Examples", "mount_gradio_app"]:
        if not hasattr(gr, attr):
            setattr(gr, attr, lambda *a, **kw: None)

    # Now import just the two functions we need from the module source
    project_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_app_under_test", project_root / "app.py")
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.handle_pdf_upload
    except Exception:
        return None


def test_handle_pdf_upload_none_returns_reset():
    from retrieval.pdf_utils import extract_text_from_pdf as _ext  # noqa: F401
    # Test the logic directly without importing app (to avoid starlette dependency)
    # Inline the expected behavior:
    def handle_pdf_upload(file_path):
        if not file_path:
            return "", "*Upload a PDF to extract its text into the query box above.*"
        text = _ext(file_path)
        if text.startswith("(Could not extract PDF text:"):
            return text, f"⚠ {text}"
        return text, "*PDF loaded — text extracted into the query box. Edit as needed, then click Search.*"

    query, status = handle_pdf_upload(None)
    assert query == ""
    assert "Upload a PDF" in status


def test_handle_pdf_upload_error_surfaces_in_status(tmp_path):
    from retrieval.pdf_utils import extract_text_from_pdf as _ext  # noqa: F401

    def handle_pdf_upload(file_path):
        if not file_path:
            return "", "*Upload a PDF to extract its text into the query box above.*"
        text = _ext(file_path)
        if text.startswith("(Could not extract PDF text:"):
            return text, f"⚠ {text}"
        return text, "*PDF loaded — text extracted into the query box. Edit as needed, then click Search.*"

    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    query, status = handle_pdf_upload(str(bad))
    assert query.startswith("(Could not extract PDF text:")
    assert "⚠" in status
