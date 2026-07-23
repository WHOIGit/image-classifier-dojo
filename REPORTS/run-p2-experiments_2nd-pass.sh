#!/usr/bin/env bash
#
# P2 experiment campaign — 2nd pass.
#
# Reruns the 8 experiments that failed in the 1st pass, plus the 4 already-
# passing 07a transfer-miniset experiments re-run against the corrected
# checkpoint (venomous-panda/epoch=15 vs the old neon-wapiti/epoch=7).
#
# Fixes applied before this script (see REPORTS/P2-TRAINING-PLANS.md):
#   1. backbone.py  — convnext classifier strip preserves Flatten
#                     (01_architecture/nes_convnext_tiny)
#   2. parquet_images.py — strings_can_be_null=True so blank Type cells
#                     are read as None → dropped by missing_policy
#                     (07b, 08/type_only, 08/multihead_type*)
#   3. 07b/_base.yaml + 08/type_only.yaml — preflight warns (not errors) on
#                     the 7 val-only Type classes not present in train
#   4. 07a+07b _base.yaml URIs repointed to venomous-panda/epoch=15
#
# Experiment matrix / rationale: REPORTS/P2-TRAINING-PLANS.md
# Record final numbers into:     REPORTS/P2-report.md
#
# Usage:
#   REPORTS/run-p2-experiments_2nd-pass.sh
#   WAIT_SECONDS=120 REPORTS/run-p2-experiments_2nd-pass.sh
#   DRY_RUN=1 REPORTS/run-p2-experiments_2nd-pass.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAIT_SECONDS="${WAIT_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] && ! command -v dojo >/dev/null 2>&1; then
  echo "error: 'dojo' not found on PATH — enter the dev shell first (nix develop)" >&2
  exit 1
fi

# 8 failed from 1st pass + 4 07a transfer runs with corrected checkpoint.
# Order: failures first (convnext, then 07b, then multihead type variants),
# then 07a re-runs at the end.
CONFIGS=(
  p2/01_architecture/nes_convnext_tiny
  p2/07b_transfer-denticle-type/frozen_linear
  p2/07b_transfer-denticle-type/frozen_mlp
  p2/07b_transfer-denticle-type/unfrozen_linear
  p2/07b_transfer-denticle-type/unfrozen_mlp
  p2/08_multihead/multihead_type_weighted
  p2/08_multihead/multihead_type
  p2/08_multihead/type_only
  p2/07a_transfer-miniset/frozen_linear
  p2/07a_transfer-miniset/frozen_mlp
  p2/07a_transfer-miniset/frozen
  p2/07a_transfer-miniset/unfrozen
)

total="${#CONFIGS[@]}"
echo "P2 2nd-pass campaign: $total configs, ${WAIT_SECONDS}s between runs"
[ "$DRY_RUN" = "1" ] && echo "(dry run — nothing will be executed)"
echo

declare -a PASSED=() FAILED=()
i=0
for sel in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo "[$i/$total] RUN   dojo train experiment=$sel"
  if [ "$DRY_RUN" = "1" ]; then
    PASSED+=("$sel")
  elif dojo train "experiment=$sel"; then
    PASSED+=("$sel")
  else
    echo "  !! failed: $sel"
    FAILED+=("$sel")
  fi

  if [ "$i" -lt "$total" ] && [ "$DRY_RUN" != "1" ]; then
    echo "  ...waiting ${WAIT_SECONDS}s before next run..."
    sleep "$WAIT_SECONDS"
  fi
  echo
done

echo "=== 2nd-pass summary ==="
echo "passed  (${#PASSED[@]}): ${PASSED[*]:-}"
echo "failed  (${#FAILED[@]}): ${FAILED[*]:-}"

[ "${#FAILED[@]}" -eq 0 ]
