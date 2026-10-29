import subprocess
from scripts.check_docs_integrity import check

def test_tracked_links_and_private_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "doc.md").write_text("[ok](a%20b.md#heading) ![image](/image.svg) [remote](https://example.invalid/private)\n[ref]: a%20b.md\n```md\n[example](missing-example)\n```\n=======\n")
    (tmp_path / "a b.md").write_text("# heading\n")
    (tmp_path / "image.svg").write_text("<svg/>")
    subprocess.run(["git", "add", "doc.md", "a b.md", "image.svg"], cwd=tmp_path, check=True)
    (tmp_path / "private.md").write_text("[bad](missing)\n<<<<<<< local")
    assert check(tmp_path) == (3, [])
    (tmp_path / "doc.md").write_text("[missing](private.md)\n[ref]: gone.md\n<<<<<<< HEAD\n")
    count, errors = check(tmp_path)
    assert count == 2 and len(errors) == 3
