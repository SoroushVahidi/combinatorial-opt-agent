"""Regression guard for the camera-ready figure-build pipeline.

Context: `tools/build_eaai_camera_ready_figures.py` previously crashed
mid-render with `KeyError: 'JPEG'` (Pillow's PDF plugin embeds RGB images
as JPEG streams and expects `Image.SAVE["JPEG"]` to already be registered,
which a PNG-only `.save()` call does not trigger -- see the `Image.init()`
call and comment at the top of that module). The crash occurred AFTER the
PNG had already been written but WHILE writing the PDF, so a naive
save-in-place could leave a truncated/corrupt tracked PDF on disk. These
tests check both: (1) the underlying Pillow save path works end-to-end for
PNG+PDF from a fresh process state, and (2) `_save_both` never leaves a
partial file at the real output path when a render fails partway through.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_pillow_png_then_pdf_save_succeeds_in_fresh_process(tmp_path: Path) -> None:
    """Reproduces the exact prior failure mode in a clean subprocess (Pillow's
    plugin registry is populated lazily per-process, so this must not import
    anything else that could mask the bug)."""
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from tools.build_eaai_camera_ready_figures import Image
img = Image.new("RGB", (50, 50), "white")
img.save({str(tmp_path / "smoke.png")!r})
img.convert("RGB").save({str(tmp_path / "smoke.pdf")!r})
print("OK")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert (tmp_path / "smoke.png").stat().st_size > 0
    pdf_bytes = (tmp_path / "smoke.pdf").read_bytes()
    assert pdf_bytes.startswith(b"%PDF"), "output is not a valid PDF"
    assert len(pdf_bytes) > 200, "PDF suspiciously small (possible truncated render)"


def test_save_both_leaves_no_partial_file_on_failure(tmp_path: Path, monkeypatch) -> None:
    """If the PDF stage raises, `_save_both` must not have written (or left
    behind) a truncated file at the real output path."""
    import tools.build_eaai_camera_ready_figures as mod

    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    img = mod.Image.new("RGB", (20, 20), "white")

    original_save = mod.Image.Image.save

    def _boom(self, fp, *args, **kwargs):
        fp_str = str(fp)
        if fp_str.endswith(".pdf"):
            raise KeyError("JPEG")  # simulate the original crash
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(mod.Image.Image, "save", _boom)

    try:
        mod._save_both(img, "regression_test_stem")
    except KeyError:
        pass

    assert not (tmp_path / "regression_test_stem.png").exists(), "PNG must not be committed to the real path on a failed render"
    assert not (tmp_path / "regression_test_stem.pdf").exists(), "PDF must not be committed to the real path on a failed render"
    leftover_tmp_files = list(tmp_path.glob("regression_test_stem*.tmp.*"))
    assert not leftover_tmp_files, f"temp files should be cleaned up on failure, found: {leftover_tmp_files}"


def test_tracked_figure2_pdf_is_valid() -> None:
    """The currently-committed figure2 PDF (regenerated 2026-08-12 after the
    Pillow fix) must be a real, non-truncated PDF, not the 144-byte
    truncated file the original bug produced."""
    pdf_path = ROOT / "results" / "paper" / "eaai_camera_ready_figures" / "figure2_main_benchmark_comparison.pdf"
    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 10_000, f"figure2 PDF is suspiciously small ({len(data)} bytes) -- possible truncated render"
