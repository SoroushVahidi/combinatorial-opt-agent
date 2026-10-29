"""Package the already-built corrected manuscript; never run scientific code."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = 'Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf'
ARCHIVE = 'JDSA_Stage3_Submission_Source.zip'
SUFFIXES = {'.tex', '.bib', '.bbl', '.cls', '.bst', '.sty'}
EXTRA = {'Fig1.pdf', MANUSCRIPT, 'ESM_1.pdf', 'README.md', 'build.sh', 'package_source.py'}

def main():
    files = sorted(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in SUFFIXES or p.name in EXTRA))
    for name in EXTRA | {'main.tex', 'supplement.tex', 'main.bbl', 'supplement.bbl', 'references.bib', 'sn-jnl.cls'}:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    with zipfile.ZipFile(ROOT / ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            entry = zipfile.ZipInfo(path.name, (2026, 10, 29, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    checksums = []
    for name in [MANUSCRIPT, 'ESM_1.pdf', ARCHIVE]:
        checksums.append(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + '  ' + name)
    (ROOT / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n')
    print('Packaged', len(files), 'flat source/artifact files; SHA256SUMS.txt updated.')

if __name__ == '__main__':
    main()
