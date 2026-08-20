# Proposal: record-peak-vram

## Why

Runs on shared GPU nodes (WHOI HPC via Slurm/Apptainer) have no durable
record of peak GPU memory, so sizing future jobs (batch size, `--gres`
requests) means re-running or eyeballing `nvidia-smi`. Slurm accounting
captures peak RAM (`MaxRSS`) but not VRAM. Torch already tracks an exact
allocator high-water mark (`torch.cuda.max_memory_reserved()`) — we just
never persist it. Extends the run-record outputs from
`DESIGN-DOC/06-results-artifacts-and-metadata.md` (P2.5a, shipped);
backlog-priority follow-on, not scheduled in the current workplan.

## What Changes

- After training (and optionally inference), record the process-level peak
  GPU memory as reported by `torch.cuda.max_memory_reserved()` into the
  run's `_metadata.json` sidecar (proposed: a `resource_stats` block, e.g.
  `{"peak_vram_reserved_bytes": ..., "device": "cuda:0"}`).
- Reset the peak counter (`torch.cuda.reset_peak_memory_stats()`) at run
  start so the value is scoped to the run.
- On non-CUDA devices the field is omitted (strict-schema convention: no
  stubs, no nulls).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `results-and-artifacts`: the `_metadata.json` sidecar gains an optional
  `resource_stats` block recording peak reserved VRAM for CUDA runs.

## Impact

- `_metadata.json` sidecar schema (additive, optional block).
- Training entry point (and possibly eval/inference) gains a
  reset-at-start / read-at-end hook.
- No config surface changes anticipated; no effect on content hashes
  (resource stats are outcomes, not identity).

## Open Questions

Deferred for future exploration before implementation:

- **Reserved vs allocated**: record `max_memory_reserved` only (matches
  nvidia-smi footprint), or also `max_memory_allocated` (true tensor
  demand, better for batch-size reasoning)?
- **Scope/stages**: single end-of-run peak, or per-stage peaks
  (fit vs validate vs test vs predict) via resets between stages?
- **Time series**: is Lightning's `DeviceStatsMonitor` (per-step logging
  to the experiment logger) wanted in addition to the sidecar peak?
- **Multi-GPU**: per-device peaks vs max-across-devices once DDP lands.
- **Non-CUDA devices**: MPS/ROCm equivalents, or CUDA-only forever?
- **Placement**: is `_metadata.json` the right home, or should resource
  stats live in a separate run-report artifact?
- **Context overhead**: the CUDA context (~hundreds of MiB) is invisible
  to torch's allocator — document the discrepancy vs nvidia-smi, or try
  to capture it (e.g. one `nvidia-smi`/NVML query at end of run)?
