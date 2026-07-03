# Questions for Sidney

## 2026-07-03 09:47 EDT — NES Training Execution Hardware

- Decision: earlier CPU-only P2 smoke checks were bounded to pilot runs; after
  Sidney launched a GPU-enabled full NES baseline train, process inspection
  showed active CUDA use by the training PID.
- Rationale: the four requested full NES experiments should be judged from GPU
  runs, not the earlier CPU pilot timings.
- Review: rerun the same four committed experiment configs without pilot
  overrides on the GPU host for final experiment numbers.
