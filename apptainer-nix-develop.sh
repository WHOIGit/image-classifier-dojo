#!/usr/bin/env bash
set -euo pipefail

# Always operate relative to the directory containing this script.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

sif="nixos.sif"
overlay="nix-overlay.img"

if [[ ! -f "$sif" ]]; then
  echo "Creating $sif ..."
  apptainer pull "$sif" docker://nixos/nix:latest
fi

if [[ ! -f "$overlay" ]]; then
  echo "Creating $overlay (4 GiB) ..."
  apptainer overlay create --size 4096 "$overlay"
fi

export NIX_CONFIG='experimental-features = nix-command flakes'

exec apptainer exec \
  --nv \
  --overlay "$overlay" \
  "$sif" \
  nix develop

#apptainer exec --env NIX_CONFIG='experimental-features = nix-command flakes' --nv --overlay nix-overlay.img nix_latest.sif nix develop
