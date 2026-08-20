#!/usr/bin/env bash
set -euo pipefail

# Unique job name so we can find this exact job in sacct afterwards.
job_name="dojo-$USER-$$-$(date +%s)"

srun \
  -p gpu \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=8G \
  --pty \
  --job-name="$job_name" \
  ./apptainer-nix-develop.sh || true

echo
echo "=== Slurm session stats ==="
sleep 5 # let slurmdbd catch up
sacct --name="$job_name" --units=M \
  --format=JobID%18,Elapsed,MaxRSS,MaxVMSize,TotalCPU,AllocTRES%40
echo "(MaxRSS = peak RAM; see the .0 step row. VRAM is not tracked by Slurm.)"
