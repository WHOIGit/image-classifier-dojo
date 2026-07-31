#!/usr/bin/env bash
set -euo pipefail

exec srun \
  -p gpu \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=8G \
  --pty \
  ./apptainer-nix-develop.sh
