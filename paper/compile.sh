#!/bin/bash
set -e

FILE="${1:-paper}"

pdflatex -interaction=nonstopmode "$FILE.tex"
bibtex "$FILE"
pdflatex -interaction=nonstopmode "$FILE.tex"
pdflatex -interaction=nonstopmode "$FILE.tex"

echo "Done: $FILE.pdf"
