#!/bin/sh
# Build both documents from clean source using the established pdfLaTeX/BibTeX procedure.
set -eu
# Fixed document timestamps make independent clean builds byte-comparable.
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1793313920}
FORCE_SOURCE_DATE=1
export SOURCE_DATE_EPOCH FORCE_SOURCE_DATE
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
source_dir=$PWD
output_dir=${OUTPUT_DIR:-$source_dir}
mkdir -p "$output_dir"
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/jdsa-build.XXXXXX")
if [ "${KEEP_BUILD:-0}" = 1 ]; then
    echo "Build directory: $build_dir"
else
    trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
fi
cp ./*.tex ./*.bib ./*.cls ./*.bst ./*.sty "$build_dir/"
cd "$build_dir"
# Regenerate the only figure from its editable TikZ source.
pdflatex -interaction=nonstopmode -halt-on-error Fig1.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
bibtex main
bibtex supplement
for pass in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
done
if grep -E 'undefined|Rerun to get cross-references right|There were multiply-defined labels|Missing character:|Overfull|between bookmark levels' main.log supplement.log; then
    echo 'Unresolved LaTeX diagnostics; PDFs were not published.' >&2
    exit 1
fi
cp main.pdf "$output_dir/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_A_Diagnostic_Study.pdf"
cp main.bbl supplement.bbl Fig1.pdf "$output_dir/"
python3 "$source_dir/export_crossrefs.py" "$build_dir" "$output_dir"
cp supplement.pdf "$output_dir/ESM_1_Retrieval_Assisted_Scalar_Grounding_Supplement.pdf"

printf "%s\n" "PASS: final main/supplement logs have no unresolved citations, references, missing characters or overfull boxes."
