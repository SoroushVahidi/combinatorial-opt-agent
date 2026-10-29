#!/bin/sh
# Build both documents from clean source using the established pdfLaTeX/BibTeX procedure.
set -eu
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
if grep -E 'undefined|Rerun to get cross-references right|There were multiply-defined labels|Missing character:|Overfull' main.log supplement.log; then
    echo 'Unresolved LaTeX diagnostics; PDFs were not published.' >&2
    exit 1
fi
cp main.pdf "$output_dir/manuscript.pdf"
cp supplement.pdf "$output_dir/ESM_1.pdf"

printf "%s\n" "PASS: final main/supplement logs have no unresolved citations, references, missing characters or overfull boxes."
