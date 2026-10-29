#!/bin/sh
# Build both documents from clean source using the established pdfLaTeX/BibTeX procedure.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
source_dir=$PWD
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/jdsa-phase1-build.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
cp ./*.tex ./*.bib ./*.cls ./*.bst "$build_dir/"
cp -R figures tables "$build_dir/"
cd "$build_dir"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
bibtex main
bibtex supplement
for pass in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
done
if grep -E 'undefined|Rerun to get cross-references right|There were multiply-defined labels|Missing character:' main.log supplement.log; then
    echo 'Unresolved LaTeX diagnostics; PDFs were not published.' >&2
    exit 1
fi
cp main.pdf "$source_dir/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase1.pdf"
cp supplement.pdf "$source_dir/Retrieval_Assisted_Scalar_Grounding_JDSA_Phase1_Supplement.pdf"
