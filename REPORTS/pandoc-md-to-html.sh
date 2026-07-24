#!/usr/bin/env bash
set -euo pipefail

input=$1
output=${input%.md}.html

pandoc -f gfm -t html5 "$input" \
  -o "$output" \
  --standalone \
  --metadata pagetitle="P2 Training Findings" \
  --include-in-header <(printf '<style>body { max-width: 54em; }</style>\n')
