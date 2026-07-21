#! /usr/bin/env bash

find DESIGN-DOC/ -type f -name "*.md" | sort | xargs cat > /tmp/DESIGN-DOC.md
pandoc /tmp/DESIGN-DOC.md -o "DESIGN-DOC/image-classifier-dojo refactor design-doc.pdf" \
    --pdf-engine=xelatex -V geometry:margin=0.7in -V geometry:bottom=1in
