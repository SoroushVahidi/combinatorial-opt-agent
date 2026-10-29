"""Build a flat, source-only journal-review archive. No experimental execution."""
from pathlib import Path
import re,zipfile
R=Path(__file__).resolve().parents[2];P=R/'manuscript/jdsa-phase4'
def expand(p):
 s=p.read_text()
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(p.parent/(m[1]+('.tex' if not m[1].endswith('.tex') else ''))),s)
files={name:(P/name).read_bytes() for name in ['references.bib','sn-jnl.cls','sn-basic.bst','orcidlink.sty','build.sh']}
for name in ['main.tex','supplement.tex']:
 files[name]=expand(P/name).replace('figures/Fig1.pdf','Fig1.pdf').encode()
files['Fig1.pdf']=(P/'figures/Fig1.pdf').read_bytes()
files['Fig1.tex']=expand(P/'figures/Fig1.tex').encode()
files['README.md']=b'''# Flat JDSA Phase 4 source package (author review)

Extract into an empty directory. Install TeX Live with pdfLaTeX, BibTeX,
amsmath, graphicx, tabularx, booktabs, algorithm/algorithmicx/algpseudocode,
xurl, microtype, xr-hyper, hyperref, multirow, TikZ and placeins, plus dependencies
of the supplied Springer Nature class. Run `sh build.sh`.

This independently builds the complete main article and Online Resource 1,
resolves cross-document citations, and produces descriptively named PDFs.
All manuscript inputs are inlined: the source upload has no subfolders.
The supplied vector Fig1.pdf is used directly. Fig1.tex is its editable
TikZ/standalone source; regenerating it is optional and document-only.
Keep both generated PDFs in the same directory for relative PDF links.
Some browser viewers do not follow cross-document PDF links.

Main article: main.tex; supplementary information: supplement.tex.
The journal requires an editable source package and a compiled PDF.
Online Resource 1 is the generated ESM_1_..._Supplement.pdf.
No missing historical experimental outputs can be recovered from this ZIP.
No dataset, private review log, local M5 evidence or credentials are included.

Original Springer class/style notices and third-party rights are retained.
No new license is assigned. Established author information is preserved.
This Phase 4 revision awaits independent manuscript review.
Do not submit until withdrawal of the previous overlapping submission is
confirmed. This package is not a journal submission or author approval.
'''
out=P/'Retrieval_Assisted_Scalar_Grounding_JDSA_Phase4_Submission_Source.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,data in sorted(files.items()):
  info=zipfile.ZipInfo(name,(2026,10,29,12,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16;z.writestr(info,data)
print(f'Created {out.name}: {len(files)} flat source files')
