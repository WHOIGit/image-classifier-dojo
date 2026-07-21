#!/usr/bin/env bash
#
# Run every configured P2 experiment in sequence, pausing between runs.
#
# Experiment matrix / rationale: REPORTS/P2-TRAINING-PLANS.md
# Record final numbers into:     REPORTS/P2-report.md
#
# Prerequisites:
#   - Run inside the dev shell so `dojo` is on PATH:  nix develop
#   - Datasets present under ./datasets (nes-hf, ichthyoliths, plankton-miniset)
#
# Usage:
#   REPORTS/run-p2-experiments.sh              # run all, 60s between runs
#   WAIT_SECONDS=120 REPORTS/run-p2-experiments.sh
#   DRY_RUN=1 REPORTS/run-p2-experiments.sh    # print the plan, run nothing
#
# Transfer groups (07a/07b) carry a placeholder checkpoint URI
# (REPLACE_WITH_NES_RUN); those configs are skipped until you point
# backbone.weights.uri at a real NES checkpoint. `_base.yaml` include files
# are not runnable on their own and are excluded automatically.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXP_DIR="configs/experiment/p2"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ] && ! command -v dojo >/dev/null 2>&1; then
  echo "error: 'dojo' not found on PATH — enter the dev shell first (nix develop)" >&2
  exit 1
fi

# Sorted so groups run 00 -> 08; exclude `_*.yaml` include bases.
mapfile -t CONFIGS < <(find "$EXP_DIR" -name '*.yaml' ! -name '_*' | sort)
total="${#CONFIGS[@]}"

if [ "$total" -eq 0 ]; then
  echo "error: no experiment configs found under $EXP_DIR" >&2
  exit 1
fi

echo "P2 experiment campaign: $total configs, ${WAIT_SECONDS}s between runs"
[ "$DRY_RUN" = "1" ] && echo "(dry run — nothing will be executed)"
echo

declare -a PASSED=() FAILED=() SKIPPED=()
i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  sel="${cfg#configs/experiment/}"; sel="${sel%.yaml}"   # e.g. p2/00_baseline/nes_baseline

  # The placeholder URI may live in the config or in the sibling `_base.yaml`
  # it inherits from (transfer groups), so check both.
  check_files=("$cfg")
  sibling_base="$(dirname "$cfg")/_base.yaml"
  [ -f "$sibling_base" ] && check_files+=("$sibling_base")
  if grep -q "REPLACE_WITH_NES_RUN" "${check_files[@]}"; then
    echo "[$i/$total] SKIP  $sel  (checkpoint URI not set — edit backbone.weights.uri)"
    SKIPPED+=("$sel")
    continue
  fi

  echo "[$i/$total] RUN   dojo train experiment=$sel"
  if [ "$DRY_RUN" = "1" ]; then
    PASSED+=("$sel")
  elif dojo train "experiment=$sel"; then
    PASSED+=("$sel")
  else
    echo "  !! failed: $sel"
    FAILED+=("$sel")
  fi

  # Pause between commands, but not after the final config.
  if [ "$i" -lt "$total" ] && [ "$DRY_RUN" != "1" ]; then
    echo "  ...waiting ${WAIT_SECONDS}s before next run..."
    sleep "$WAIT_SECONDS"
  fi
  echo
done

echo "=== campaign summary ==="
echo "passed  (${#PASSED[@]}): ${PASSED[*]:-}"
echo "failed  (${#FAILED[@]}): ${FAILED[*]:-}"
echo "skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-}"

[ "${#FAILED[@]}" -eq 0 ]
